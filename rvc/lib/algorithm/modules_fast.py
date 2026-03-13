"""
VITS Fast: Optimized posterior encoder and normalizing flow.

Replaces WaveNet with depthwise-separable convolutions for significantly
faster training and inference while maintaining quality parity with VITS v1.

Key differences from v1 (WaveNet-based):
- Depthwise-separable convs instead of dense dilated convs (~2.7x fewer FLOPs/layer)
- GELU instead of gated tanh·sigmoid activation
- No weight_norm (uses RMSNorm + careful init instead)
- Exponential dilation [1,2,4,8] for much larger receptive field (RF 91 vs 9)

Key differences from VITS mod (ConvNeXt-based):
- No LayerNorm, FiLM, GRN, or CAM (much less overhead per block)
- expansion=2 instead of mlp_ratio=4 (lighter pointwise path)
- Simple additive conditioning instead of FiLM
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from rvc.lib.algorithm.commons import sequence_mask


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class FastRMSNorm1D(nn.Module):
    """Lightweight channel-first RMSNorm for (B, C, T) tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps) * self.scale


class FastConvBlock(nn.Module):
    """
    Lightweight depthwise-separable conv block.

    DWConv(k, dilated) → RMSNorm → PWConv(expand 2x) → GELU → PWConv(contract) → residual

    Per-layer FLOPs (channels=192, k=7):
      DWConv: 192×7 = 1,344
      PWConv1: 192×384 = 73,728
      PWConv2: 384×192 = 73,728
      Total: ~149K per timestep  (vs WaveNet ~405K = 2.7x less)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        expansion: int = 2,
    ):
        super().__init__()
        expanded = channels * expansion
        padding = (kernel_size * dilation - dilation) // 2

        self.dwconv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation, groups=channels,
        )
        self.norm = FastRMSNorm1D(channels)
        self.pwconv1 = nn.Conv1d(channels, expanded, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(expanded, channels, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.dwconv.weight, std=0.02)
        nn.init.constant_(self.dwconv.bias, 0)
        nn.init.trunc_normal_(self.pwconv1.weight, std=0.02)
        nn.init.constant_(self.pwconv1.bias, 0)
        nn.init.trunc_normal_(self.pwconv2.weight, std=0.02)
        nn.init.constant_(self.pwconv2.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x * x_mask)
        if g_cond is not None:
            x = x + g_cond
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return (residual + x) * x_mask


# ---------------------------------------------------------------------------
# Normalizing Flow
# ---------------------------------------------------------------------------

class Flip(nn.Module):
    """Channel flip for coupling layers."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        return x, logdet


class FastCouplingLayer(nn.Module):
    """
    Lightweight affine coupling layer using depthwise-separable convolutions.

    Replaces WaveNet's dense dilated convs + gated activation with
    FastConvBlocks. Speaker conditioning is applied per-block (additive),
    matching WaveNet's per-layer conditioning strategy.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 7,
        n_layers: int = 4,
        gin_channels: int = 0,
        mean_only: bool = True,
    ):
        assert channels % 2 == 0
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        dilation_cycle = [1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            FastConvBlock(
                hidden_channels, kernel_size,
                dilation=dilation_cycle[i % len(dilation_cycle)],
            )
            for i in range(n_layers)
        ])

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels * n_layers, 1)

        out_ch = self.half_channels * (2 - mean_only)
        self.post = nn.Conv1d(hidden_channels, out_ch, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask

        g_proj = None
        if g is not None and hasattr(self, 'cond'):
            g_proj = self.cond(g)

        for i, block in enumerate(self.blocks):
            g_l = None
            if g_proj is not None:
                offset = i * self.hidden_channels
                g_l = g_proj[:, offset:offset + self.hidden_channels, :]
            h = block(h, x_mask, g_cond=g_l)

        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            logs = torch.clamp(logs, min=-10.0, max=2.0)
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
        pass  # No weight_norm used


class FastCouplingBlock(nn.Module):
    """Normalizing flow with lightweight depthwise-separable coupling layers."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 7,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                FastCouplingLayer(
                    channels, hidden_channels, kernel_size,
                    n_layers=n_layers, gin_channels=gin_channels,
                    mean_only=True,
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
        pass  # No weight_norm used


# ---------------------------------------------------------------------------
# Posterior Encoder
# ---------------------------------------------------------------------------

class FastPosteriorEncoder(nn.Module):
    """
    Lightweight posterior encoder using depthwise-separable convolutions.

    Compared to v1's WaveNet PosteriorEncoder (16 layers, weight_norm):
    - 8 layers of FastConvBlocks (half the depth)
    - ~5.4x fewer parameters
    - No weight_norm overhead
    - Larger receptive field via exponential dilation

    Only used during training (discarded at inference), so lighter is better.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 7,
        n_layers: int = 8,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        dilation_cycle = [1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            FastConvBlock(
                hidden_channels, kernel_size,
                dilation=dilation_cycle[i % len(dilation_cycle)],
            )
            for i in range(n_layers)
        ])

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels * n_layers, 1)

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.proj.weight, std=0.01)
        nn.init.constant_(self.proj.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ):
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)

        x = self.pre(x) * x_mask

        g_proj = None
        if g is not None and hasattr(self, 'cond'):
            g_proj = self.cond(g)

        for i, block in enumerate(self.blocks):
            g_l = None
            if g_proj is not None:
                offset = i * self.hidden_channels
                g_l = g_proj[:, offset:offset + self.hidden_channels, :]
            x = block(x, x_mask, g_cond=g_l)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        logs = torch.clamp(logs, min=-10.0, max=2.0)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        pass  # No weight_norm used
