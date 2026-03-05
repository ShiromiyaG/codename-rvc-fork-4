"""
v3 modules: ConvNeXt-based Posterior Encoder and Normalizing Flow.

Replaces the WaveNet backbone used in VITS v1/v2 with:
  - ConvNeXt 1D blocks (depthwise + LayerNorm + GELU + pointwise)
  - FiLM conditioning (scale + shift) instead of WaveNet-style additive
  - CAM (Context Aware Module) for expanded receptive field in the flow

These modules are drop-in replacements: same input/output signatures
as the original PosteriorEncoder and ResidualCouplingBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

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
        # Upcast to FP32 for numerical stability (BF16-safe)
        orig_dtype = x.dtype
        x = x.float()
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None].float() * x + self.bias[:, None].float()
        return x.to(orig_dtype)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    
    Projects a global conditioning vector g (B, gin_channels, 1) into
    per-channel scale and shift for the target features: y = scale * x + shift.
    """

    def __init__(self, gin_channels: int, channels: int):
        super().__init__()
        self.proj = nn.Conv1d(gin_channels, channels * 2, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(g)  # (B, 2*C, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # each (B, C, 1)
        return gamma * x + beta


class CAM(nn.Module):
    """
    Context Aware Module.
    
    A large-kernel depthwise convolution that cheaply expands the receptive
    field.  Placed before or after ConvNeXt blocks inside each coupling layer.
    
    CAM(x) = LayerNorm(DwConv_large(x)) + x
    """

    def __init__(self, channels: int, kernel_size: int = 31):
        super().__init__()
        self.dwconv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = ConvNeXtLayerNorm1D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.dwconv(x))


class DropPath(nn.Module):
    """Stochastic Depth — drops the entire residual branch per-sample.

    At eval time this is a no-op, so inference is deterministic.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2).

    Encourages feature diversity by normalizing each channel's response
    relative to the global aggregate across channels.  Initialised as
    identity (gamma=0, beta=0) so it is safe to add to a pretrained model.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        gx = torch.norm(x, p=2, dim=2, keepdim=True)          # (B, C, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)       # (B, C, 1)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock1D(nn.Module):
    """
    1D ConvNeXt **V2** block with optional FiLM conditioning.

    DwConv -> LayerNorm -> FiLM(opt) -> Conv1d↑ -> GELU -> GRN -> Conv1d↓
         -> layer_scale -> DropPath -> + residual

    Changes vs V1:
      - GRN after GELU for feature diversity
      - Conv1d(1) instead of Linear (eliminates 2 permute ops per block)
      - DropPath for stochastic depth regularisation
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        dilation: int = 1,
        layer_scale_init: float = 1e-6,
        gin_channels: int = 0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        mlp_dim = int(channels * mlp_ratio)

        self.dwconv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            dilation=dilation,
            groups=channels,
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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.film = FiLM(gin_channels, channels) if gin_channels > 0 else None

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

        x = self.drop_path(x)
        return (residual + x) * x_mask


# ---------------------------------------------------------------------------
# Posterior Encoder v3
# ---------------------------------------------------------------------------

class PosteriorEncoder_v3(nn.Module):
    """
    Posterior Encoder using ConvNeXt blocks with FiLM conditioning.
    
    Replaces the WaveNet-based PosteriorEncoder.  Same interface:
        forward(x, x_lengths, g=None) -> (z, m, logs, x_mask)
    
    Architecture:
        Conv1d_pre -> N x ConvNeXtBlock1D (with FiLM) -> Conv1d_proj -> (m, logs)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 7,
        n_layers: int = 8,
        gin_channels: int = 0,
        mlp_ratio: float = 4.0,
        checkpointing: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.checkpointing = checkpointing

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # Linearly increasing drop-path rates (0 → drop_path_rate)
        dp_rates = (
            [drop_path_rate * i / (n_layers - 1) for i in range(n_layers)]
            if n_layers > 1 else [0.0]
        )
        # Cycling dilation schedule for expanded receptive field at zero cost
        dilation_cycle = [1, 2, 4, 8]

        self.blocks = nn.ModuleList([
            ConvNeXtBlock1D(
                channels=hidden_channels,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                dilation=dilation_cycle[i % len(dilation_cycle)],
                gin_channels=gin_channels,
                drop_path=dp_rates[i],
            )
            for i in range(n_layers)
        ])
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        self.apply(self._init_weights)

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
            if self.training and self.checkpointing:
                x = checkpoint(block, x, x_mask, g, use_reentrant=False)
            else:
                x = block(x, x_mask, g=g)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        """No-op: ConvNeXt uses LayerNorm, not weight_norm."""
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x, torch.zeros([1], device=x.device)


class ResidualCouplingLayer_v3(nn.Module):
    """
    Affine coupling layer with ConvNeXt + CAM backbone.
    
    Replaces the WaveNet-based ResidualCouplingLayer.  Same interface:
        forward(x, x_mask, g=None, reverse=False)
    
    Architecture:
        split(x) -> pre -> CAM -> N x ConvNeXtBlock1D (with FiLM) -> post -> affine transform
    """

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
        checkpointing: bool = False,
    ):
        assert channels % 2 == 0
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.checkpointing = checkpointing

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.cam = CAM(hidden_channels, kernel_size=cam_kernel_size)

        # Ascending dilation schedule for expanded receptive field
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
        # Zero-init the output projection so the flow starts as identity
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

        self.apply(self._init_weights)
        # Re-zero post after apply
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

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
            if self.training and self.checkpointing:
                h = checkpoint(block, h, x_mask, g, use_reentrant=False)
            else:
                h = block(h, x_mask, g=g)

        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
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
            return x, torch.zeros([1], device=x.device)

    def remove_weight_norm(self):
        """No-op: ConvNeXt uses LayerNorm, not weight_norm."""
        pass


class ResidualCouplingBlock_v3(nn.Module):
    """
    Normalizing flow with ConvNeXt + CAM coupling layers.
    
    Replaces ResidualCouplingBlock.  Same interface:
        forward(x, x_mask, g=None, reverse=False) -> x
    """

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
        checkpointing: bool = False,
    ):
        super().__init__()
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer_v3(
                    channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                    cam_kernel_size=cam_kernel_size,
                    mlp_ratio=mlp_ratio,
                    checkpointing=checkpointing,
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
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows[::-1]:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        """No-op: ConvNeXt uses LayerNorm, not weight_norm."""
        pass
