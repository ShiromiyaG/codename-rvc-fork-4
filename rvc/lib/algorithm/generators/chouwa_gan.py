"""
ChouwaGAN-PCPH Generator for RVC.

A ConvNeXt-backbone + HiFiGAN-head vocoder with Pseudo-Constant-Power Harmonic
(PCPH) source injection for pitch-conditioned waveform synthesis. Adapted from
the fish-vocoder FireflyGAN architecture. The NSF source (single sine at f0)
has been replaced by the PCPH band-limited harmonic source — which sums all
harmonics up to Nyquist by construction, eliminating the aliasing (spectral
mirrors / frequency-line artefacts) produced by NSF's unconstrained approach.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock_SnakeBeta
from rvc.lib.algorithm.conformer.activations import SnakeBeta


def remove_weight_norm_legacy_safe(module):
    if is_parametrized(module, "weight"):
        remove_parametrizations(module, "weight", leave_parametrized=True)
    else:
        remove_weight_norm(module)


# ---------------------------------------------------------------------------
# ConvNeXt Backbone
# ---------------------------------------------------------------------------


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2) in channels-first format (B, C, T).
    Normalises each channel's response relative to the global aggregate.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — channels-first
        gx = torch.linalg.vector_norm(x, ord=2, dim=2, keepdim=True)  # (B, C, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)           # (B, C, 1)
        return self.gamma * (x * nx) + self.beta + x


class RMSNorm1D(nn.Module):
    """Root Mean Square Normalization (channels-first, 1D)."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — fused rsqrt avoids separate sqrt + div
        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps) * self.weight[:, None]
        return x.to(orig_dtype) if orig_dtype != torch.float32 else x


class ChannelsFirstLayerNorm(nn.Module):
    """LayerNorm operating on channels-first (B, C, T) without transpose."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            x = x.float()
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x.to(orig_dtype) if orig_dtype != torch.float32 else x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (channels-first throughout — no transpose).
    DwConv -> LayerNorm -> Conv1d(1x1) -> Gated (tanh*sigmoid) -> GRN -> Conv1d(1x1)

    The GELU activation is replaced by a gated unit: pwconv1 projects to
    2*hidden_dim, one half goes through tanh and the other through sigmoid,
    their product forms the gated output. This is the same audio inductive
    bias as WaveNet, which significantly accelerates convergence without
    changing parameter count or architectural complexity.
    """

    def __init__(self, dim: int, layer_scale_init_value: float = 0.0,
                 mlp_ratio: float = 3.0, kernel_size: int = 13, dilation: int = 1):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, dilation=dilation,
            padding=(kernel_size * dilation - dilation) // 2,
            groups=dim, bias=True,
        )
        self.norm = ChannelsFirstLayerNorm(dim)

        hidden_dim = int(mlp_ratio * dim)
        # 2*hidden_dim: one half for tanh gate, one half for sigmoid gate
        self.pwconv1 = nn.Conv1d(dim, hidden_dim * 2, 1)
        self.grn = GlobalResponseNorm(hidden_dim)
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, 1)

        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )

    def forward(self, x, apply_residual: bool = True):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        # Gated activation: tanh(x_a) * sigmoid(x_b)
        x_a, x_b = x.chunk(2, dim=1)
        x = torch.tanh(x_a) * torch.sigmoid(x_b)
        x = self.grn(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        if apply_residual:
            x = residual + x
        return x


class ConvNeXtEncoder(nn.Module):
    """Multi-stage ConvNeXt encoder."""

    def __init__(
        self,
        input_channels: int = 192,
        depths: list = [3, 3, 9, 4],
        dims: list = [128, 256, 384, 512],
        dilations: list = [[1, 1, 1], [1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1]],
        layer_scale_init_value: float = 0.0,
        kernel_size: int = 13,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode="zeros"),
            RMSNorm1D(dims[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                RMSNorm1D(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.downsample_layers.append(mid_layer)

        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    layer_scale_init_value=layer_scale_init_value,
                    kernel_size=kernel_size,
                    dilation=dilations[i][j],
                    mlp_ratio=mlp_ratio,
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage)

        self.norm = RMSNorm1D(dims[-1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# HiFiGAN-PCPH Head
# ---------------------------------------------------------------------------

def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class UpsampleConvTranspose1d(nn.Module):
    """
    Transposed-convolution upsampler (BigVGAN approach).

    ConvTranspose1d with stride == upsample_factor.  The kernel size is
    chosen so that ``padding = (kernel_size - stride) // 2``, which produces
    an output whose length is exactly ``input_length * stride``.

    Unlike nearest-neighbor + Conv1d, each output position has independent
    weight paths through the transposed kernel, avoiding the piecewise-
    constant gradient problem that can cause mode collapse to silence.
    """
    def __init__(self, in_channels, out_channels, upsample_factor, kernel_size=None):
        super().__init__()
        stride = upsample_factor
        if kernel_size is None:
            kernel_size = upsample_factor * 2          # BigVGAN default: k = 2*stride

        # Ensure (kernel_size - stride) is even for symmetric padding
        if (kernel_size - stride) % 2 != 0:
            kernel_size += 1

        padding = (kernel_size - stride) // 2

        _conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        nn.init.kaiming_normal_(_conv.weight)
        if _conv.bias is not None:
            nn.init.zeros_(_conv.bias)
        self.conv = weight_norm(_conv)

    def forward(self, x):
        return self.conv(x)


class LearnableHarmonicSource(nn.Module):
    """
    Generates a matrix of individual harmonic sine waves and noise.
    The generator's har_convs (1×1 convolutions) act as learnable 
    mixers at each upsampling stage.
    
    Note: This module has NO learnable parameters.
    Output: [B, n_harmonics + 1, T_audio]
    """
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        n_harmonics: int = 8,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        random_init_phase: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_harmonics = n_harmonics
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.random_init_phase = random_init_phase

        # Pre-register k as buffer — avoid recreating every forward
        k = torch.arange(1, n_harmonics + 1, dtype=torch.float32).view(1, -1, 1)
        self.register_buffer("k", k)

    def forward(self, f0: torch.Tensor, upsample_factor: Optional[int] = None):
        hop = upsample_factor if upsample_factor is not None else self.hop_length
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
            
        device_type = f0.device.type
        with torch.amp.autocast(device_type, enabled=False):
            f0 = f0.float()
            
            # Use linear interpolation in both training and inference to avoid
            # train/test distribution shift in the harmonic excitation signal.
            f0_up = F.interpolate(
                f0, scale_factor=float(hop), 
                mode='linear', align_corners=False
            )
                
            # ── Phase computation in FLOAT32 ──
            # Safe for training segments (≤2s ≈ 96k samples).
            # Modular arithmetic prevents accumulation drift.
            phase_increment = f0_up / self.sample_rate  # cycles per sample
            
            if self.random_init_phase and self.training:
                init_phase = torch.rand(
                    (f0.shape[0], 1, 1), device=f0.device
                )
                phase_increment = phase_increment.clone()
                phase_increment[..., :1] = phase_increment[..., :1] + init_phase
                
            # Float32 cumsum + periodic wrap every 256 samples
            # Prevents phase drift without FP64 cost
            raw_phase = self._wrapped_cumsum(phase_increment)
            phase = raw_phase * (2.0 * math.pi)
            
            # ── Vectorized harmonics ──
            harmonics = torch.sin(self.k * phase) * self.sine_amp  # [B, H, T]
            
            # Anti-aliasing + voiced masks (fused)
            nyquist = self.sample_rate / 2.0
            voiced_mask = (f0_up > 1.0).float()
            aa_mask = (self.k * f0_up < nyquist).float()
            harmonics = harmonics * (aa_mask * voiced_mask)
            
            # Noise channel
            noise = torch.randn_like(f0_up)
            noise_amp = (
                voiced_mask * self.noise_std 
                + (1.0 - voiced_mask) * 0.003
            )
            noise = noise * noise_amp
            
            har_matrix = torch.cat([harmonics, noise], dim=1)
            
        return har_matrix

    @staticmethod
    def _wrapped_cumsum(x, chunk_size=256):
        """
        Float32 cumsum with periodic wrapping to prevent drift.
        Wraps phase to [0, 1) every chunk_size samples.
        ~30× faster than FP64 cumsum on consumer GPUs.
        """
        T = x.shape[-1]
        if T <= chunk_size:
            return torch.fmod(torch.cumsum(x, dim=-1), 1.0)

        chunks = x.split(chunk_size, dim=-1)
        results = []
        carry = torch.zeros_like(x[..., :1])
        for chunk in chunks:
            chunk_cum = torch.cumsum(chunk, dim=-1) + carry
            carry = chunk_cum[..., -1:]
            # Wrap to [0, 1) — prevents float32 from losing LSBs
            chunk_cum = torch.fmod(chunk_cum, 1.0)
            carry = torch.fmod(carry, 1.0)
            results.append(chunk_cum)
        return torch.cat(results, dim=-1)


class HiFiGANPCPHHead(nn.Module):
    """
    HiFiGAN generator head with PCPH harmonic matrix injection
    and SnakeBeta activations. Uses ConvTranspose1d (BigVGAN approach)
    for upsampling.
    """

    def __init__(
        self,
        input_channels: int = 512,
        upsample_rates: list = [8, 8, 2, 2, 2],
        upsample_kernel_sizes: list = [16, 16, 4, 4, 4],
        resblock_kernel_sizes: list = [3, 7, 11],
        resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel: int = 512,
        gin_channels: int = 256,
        sr: int = 48000,
        pre_conv_kernel_size: int = 7,
        post_conv_kernel_size: int = 7,
        n_harmonics: int = 8,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        self.upp = math.prod(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(
            input_channels, upsample_initial_channel,
            pre_conv_kernel_size, 1,
            padding=get_padding(pre_conv_kernel_size),
        ))

        self.pre_snake = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.har_convs = nn.ModuleList()

        self.channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        self.stride_f0s = [
            math.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.pre_snake.append(
                SnakeBeta(upsample_initial_channel // (2 ** i),
                          alpha_trainable=True, alpha_logscale=True)
            )

            self.ups.append(
                UpsampleConvTranspose1d(
                    in_channels=upsample_initial_channel // (2 ** i),
                    out_channels=self.channels[i],
                    upsample_factor=u,
                    kernel_size=k,
                )
            )

            stride = self.stride_f0s[i]
            if stride > 1:
                kernel = stride * 2 - stride % 2
                pad = (kernel - stride) // 2
                _har_conv = nn.Conv1d(
                    n_harmonics + 1, self.channels[i],
                    kernel_size=kernel, stride=stride, padding=pad
                )
            else:
                _har_conv = nn.Conv1d(
                    n_harmonics + 1, self.channels[i], kernel_size=1
                )
            nn.init.normal_(_har_conv.weight, std=0.01)
            if _har_conv.bias is not None:
                nn.init.zeros_(_har_conv.bias)
            self.har_convs.append(weight_norm(_har_conv))

        self.resblocks = nn.ModuleList([
            ResBlock_SnakeBeta(self.channels[i], k, d, post_act=False)
            for i in range(len(self.ups))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        self.post_snake = SnakeBeta(
            self.channels[-1], alpha_trainable=True, alpha_logscale=True
        )

        # conv_post: small init for near-zero output at start (stability)
        # Must init BEFORE weight_norm — parametrized .weight is read-only
        _conv_post = nn.Conv1d(
            self.channels[-1], 1,
            post_conv_kernel_size, 1,
            padding=get_padding(post_conv_kernel_size),
        )
        nn.init.normal_(_conv_post.weight, std=0.001)
        nn.init.zeros_(_conv_post.bias)
        self.conv_post = weight_norm(_conv_post)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
            nn.init.normal_(self.cond.weight, mean=0.0, std=0.01)
            if self.cond.bias is not None:
                nn.init.constant_(self.cond.bias, 0.0)

    def forward(self, x, har_source, g=None):
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = self.pre_snake[i](x)
            x = self.ups[i](x)
            
            har_out = self.har_convs[i](har_source)
            min_len = min(x.shape[-1], har_out.shape[-1])
            x = x[..., :min_len] + har_out[..., :min_len]
            
            start_idx = i * self.num_kernels
            if self.checkpointing and self.training:
                import torch.utils.checkpoint as tuc
                xs = sum(
                    tuc.checkpoint(
                        self.resblocks[start_idx + j], x,
                        use_reentrant=False
                    )
                    for j in range(self.num_kernels)
                )
            else:
                xs = sum(
                    self.resblocks[start_idx + j](x) 
                    for j in range(self.num_kernels)
                )
            x = xs / self.num_kernels

        x = self.post_snake(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm_legacy_safe(self.conv_pre)
        for l in self.ups:
            remove_weight_norm_legacy_safe(l.conv)
        for l in self.har_convs:
            remove_weight_norm_legacy_safe(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm_legacy_safe(self.conv_post)


# ---------------------------------------------------------------------------
# ChouwaGAN: Full Generator
# ---------------------------------------------------------------------------

class ChouwaGANGenerator(nn.Module):
    """
    ChouwaGAN: HiFiGAN with SnakeBeta activations, ConvTranspose1d upsampling,
    and band-limited harmonic source injection.

    The multi-harmonic source (LearnableHarmonicSource) individually anti-aliases
    each harmonic at Nyquist, then all harmonics are merged into a single channel
    before injection -- preserving anti-aliasing quality while maintaining the same
    information bottleneck as standard NSF. This forces the decoder to depend on
    the VITS latent z for spectral detail, preventing KL mean collapse.

    Optional ConvNeXt backbone can be enabled via use_backbone=True once KL
    stability is confirmed.

    Interface: forward(x, f0, g=None) -- same as HiFiGANNSFGenerator.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        n_harmonics: int = 8,
        use_backbone: bool = False,
        backbone_depths: list = [1, 1, 1, 1],
        backbone_dims: list = [192, 256, 384, 384],
        backbone_dilations: list = [
            [1],
            [1],
            [1],
            [1]
        ],
        backbone_kernel_size: int = 1,
        backbone_mlp_ratio: float = 3.0,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.use_backbone = use_backbone

        if use_backbone:
            self.backbone = ConvNeXtEncoder(
                input_channels=initial_channel,
                depths=backbone_depths,
                dims=backbone_dims,
                dilations=backbone_dilations,
                kernel_size=backbone_kernel_size,
                mlp_ratio=backbone_mlp_ratio,
            )
            head_input_channels = backbone_dims[-1]
        else:
            head_input_channels = initial_channel

        self.upp = math.prod(upsample_rates)

        # Band-limited harmonic source: each harmonic individually anti-aliased
        self.m_source = LearnableHarmonicSource(
            sample_rate=sr,
            hop_length=self.upp,
            n_harmonics=n_harmonics,
        )

        # Merge all harmonics + noise into 1 channel — LINEAR ONLY.
        # A learnable linear combination of harmonics preserves the 1-channel
        # information bottleneck: the merged signal contains energy ONLY at the
        # input harmonic frequencies (f0 … n_harmonics*f0), so the decoder must
        # rely on the VITS latent z for spectral detail above that range.
        #
        # NOTE: Do NOT add a nonlinearity (Tanh, etc.) after this conv.
        # Any pointwise nonlinearity creates intermodulation products that fill
        # the spectrum well beyond the input harmonics, defeating the bottleneck
        # and causing KL collapse (the decoder no longer needs z).
        self.harmonic_merge = weight_norm(nn.Conv1d(n_harmonics + 1, 1, 1))

        # HiFiGAN-PCPH head: SnakeBeta + ConvTranspose1d + 1-channel harmonic injection
        self.head = HiFiGANPCPHHead(
            input_channels=head_input_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            gin_channels=gin_channels,
            sr=sr,
            n_harmonics=0,  # 1 channel after merge
            checkpointing=checkpointing,
        )


    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        if x.shape[-1] != f0.shape[-1]:
            raise ValueError(
                f"Temporal dimension mismatch: x has {x.shape[-1]} frames, "
                f"f0 has {f0.shape[-1]} frames. They must match."
            )

        # Generate anti-aliased harmonics and merge to single channel
        har_source = self.m_source(f0, self.upp)       # [B, n_harmonics+1, T_audio]
        har_source = self.harmonic_merge(har_source)    # [B, 1, T_audio]

        if self.use_backbone:
            x = self.backbone(x)

        x = self.head(x, har_source, g=g)
        return x

    def remove_weight_norm(self):
        remove_weight_norm_legacy_safe(self.harmonic_merge)
        self.head.remove_weight_norm()
