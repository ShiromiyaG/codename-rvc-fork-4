"""
VITS Mod modules: ConvNeXt-based Posterior Encoder and Normalizing Flow.

Replaces the WaveNet backbone used in VITS v1/v2 with ConvNeXt 1D blocks
and FiLM conditioning. These modules are drop-in replacements with the same
input/output signatures as the original PosteriorEncoder and ResidualCouplingBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from rvc.lib.algorithm.commons import sequence_mask


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvNeXtLayerNorm1D(nn.Module):
    """Channel-first LayerNorm for 1D sequences (B, C, T)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        needs_upcast = orig_dtype in (torch.bfloat16, torch.float16)
        
        if needs_upcast:
            x = x.float()
        
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        
        if needs_upcast:
            x = self.weight[:, None].float() * x + self.bias[:, None].float()
            return x.to(orig_dtype)
        else:
            return self.weight[:, None] * x + self.bias[:, None]


class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""

    def __init__(self, gin_channels: int, channels: int):
        super().__init__()
        self.proj = nn.Conv1d(gin_channels, channels * 2, 1)
        
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias[:channels], 0.0)
        nn.init.constant_(self.proj.bias[channels:], 0.0)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(g)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return (1.0 + gamma) * x + beta


class CAM(nn.Module):
    """Context Aware Module - large-kernel depthwise convolution."""

    def __init__(self, channels: int, kernel_size: int = 31):
        super().__init__()
        self.dwconv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.norm = ConvNeXtLayerNorm1D(channels)
        
        nn.init.constant_(self.dwconv.weight, 0.0)
        nn.init.constant_(self.dwconv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.dwconv(x))


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt V2 block with optional FiLM conditioning."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        dilation: int = 1,
        layer_scale_init: float = 1e-6,
        gin_channels: int = 0,
    ):
        super().__init__()
        mlp_dim = int(channels * mlp_ratio)

        self.dwconv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            dilation=dilation,
            groups=channels,
            bias=True,
        )
        self.norm = ConvNeXtLayerNorm1D(channels)
        self.pwconv1 = nn.Conv1d(channels, mlp_dim, 1)
        self.act = nn.GELU()
        self.grn = GRN(mlp_dim)
        self.pwconv2 = nn.Conv1d(mlp_dim, channels, 1)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(1, channels, 1))
            if layer_scale_init > 0 else None
        )

        self.film = FiLM(gin_channels, channels) if gin_channels > 0 else None
        
        # Initialize weights (will be re-initialized by parent's apply())
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.dwconv.weight, std=0.02)
        nn.init.constant_(self.dwconv.bias, 0)
        
        nn.init.trunc_normal_(self.pwconv1.weight, std=0.02)
        nn.init.constant_(self.pwconv1.bias, 0)
        
        nn.init.trunc_normal_(self.pwconv2.weight, std=0.02)
        nn.init.constant_(self.pwconv2.bias, 0)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x * x_mask)
        x = self.norm(x)

        if self.film is not None and g is not None:
            x = self.film(x, g)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        return (residual + x) * x_mask


# ---------------------------------------------------------------------------
# Posterior Encoder v3
# ---------------------------------------------------------------------------

class PosteriorEncoderMod(nn.Module):
    """Posterior Encoder using ConvNeXt blocks with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 7,
        n_layers: int = 8,
        gin_channels: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # Exponential dilation for larger receptive field
        dilation_cycle = [1, 2, 4, 8]

        self.blocks = nn.ModuleList([
            ConvNeXtBlock1D(
                channels=hidden_channels,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                dilation=dilation_cycle[i % len(dilation_cycle)],
                gin_channels=gin_channels,
            )
            for i in range(n_layers)
        ])
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        self.apply(self._init_weights)
        
        nn.init.trunc_normal_(self.proj.weight, std=0.01)
        nn.init.constant_(self.proj.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ):
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)

        x = self.pre(x) * x_mask

        for block in self.blocks:
            x = block(x, x_mask, g=g)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        
        logs = torch.clamp(logs, min=-10.0, max=2.0)
        
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        # No weight_norm used in v3 modules
        pass


# ---------------------------------------------------------------------------
# Normalizing Flow v3
# ---------------------------------------------------------------------------

class Flip(nn.Module):
    """Channel flip for coupling layers (same as original VITS)."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.flip(x, [1])
        # Flip is volume-preserving, so logdet is always zero
        logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        return x, logdet


class ResidualCouplingLayerMod(nn.Module):
    """Affine coupling layer with ConvNeXt + CAM backbone."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 7,
        n_layers: int = 4,
        gin_channels: int = 0,
        mean_only: bool = False,
        cam_kernel_size: int = 31,
        mlp_ratio: float = 4.0,
    ):
        assert channels % 2 == 0
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.cam = CAM(hidden_channels, kernel_size=cam_kernel_size)

        # Exponential dilation for larger receptive field
        dilation_cycle = [1, 2, 4, 8]

        self.blocks = nn.ModuleList([
            ConvNeXtBlock1D(
                channels=hidden_channels,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                dilation=dilation_cycle[i % len(dilation_cycle)],
                gin_channels=gin_channels,
            )
            for i in range(n_layers)
        ])
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1
        )

        self.apply(self._init_weights)
        
        # Re-zero layers that need identity init
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        nn.init.constant_(self.cam.dwconv.weight, 0.0)
        nn.init.constant_(self.cam.dwconv.bias, 0.0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.cam(h) * x_mask

        for block in self.blocks:
            h = block(h, x_mask, g=g)

        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            logs = torch.clamp(logs, min=-10.0, max=0.5)
        else:
            # mean_only mode: logs are always zero (no scale transformation)
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            # Return consistent shape with forward pass
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet

    def remove_weight_norm(self):
        # No weight_norm used in v3 modules
        pass


class ResidualCouplingBlockMod(nn.Module):
    """Normalizing flow with ConvNeXt + CAM coupling layers."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 7,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 0,
        cam_kernel_size: int = 31,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayerMod(
                    channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                    cam_kernel_size=cam_kernel_size,
                    mlp_ratio=mlp_ratio,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        logdet_total = torch.zeros(x.size(0), device=x.device)
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, x_mask, g=g, reverse=reverse)
                logdet_total = logdet_total + logdet
        else:
            for flow in self.flows[::-1]:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x, logdet_total

    def remove_weight_norm(self):
        # No weight_norm used in v3 modules
        pass
