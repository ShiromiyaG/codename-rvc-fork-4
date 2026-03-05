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
from torch.amp import autocast
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock
from rvc.lib.algorithm.conformer.activations import SnakeBeta
from rvc.lib.algorithm.generators.pcph_gan import SourceModulePCPH


# ---------------------------------------------------------------------------
# ConvNeXt Backbone (adapted from fish-vocoder)
# ---------------------------------------------------------------------------

def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ConvNeXtLayerNorm(nn.Module):
    """LayerNorm supporting channels_first and channels_last data formats."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # F.layer_norm upcasts to FP32 internally — safe for BF16/FP16.
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Manual implementation: upcast to FP32 for numerical stability
            # when running in BF16 (BF16 has only 7 mantissa bits; variance
            # computed in BF16 loses precision and destabilises training).
            orig_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None].float() * x + self.bias[:, None].float()
            return x.to(orig_dtype)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block: DwConv -> Permute -> LayerNorm -> Linear -> GELU -> Linear -> Permute -> DropPath
    """

    def __init__(self, dim: int, drop_path_rate: float = 0.0,
                 layer_scale_init_value: float = 1e-6, mlp_ratio: float = 4.0,
                 kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
        )
        self.norm = ConvNeXtLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, apply_residual: bool = True):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = self.drop_path(x)
        if apply_residual:
            x = residual + x
        return x


class ConvNeXtEncoder(nn.Module):
    """
    Multi-stage ConvNeXt encoder. Processes mel/latent features into a deeper
    representation for the HiFiGAN head.
    """

    def __init__(
        self,
        input_channels: int = 192,
        depths: list = [3, 3, 9, 3],
        dims: list = [128, 256, 384, 512],
        drop_path_rate: float = 0.2,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
        checkpointing: bool = False,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.downsample_layers = nn.ModuleList()
        # Stem
        stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode="zeros"),
            ConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # Intermediate transition layers
        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                ConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.downsample_layers.append(mid_layer)

        # ConvNeXt stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path_rate=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    kernel_size=kernel_size,
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = ConvNeXtLayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.checkpointing = checkpointing
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.downsample_layers)):
            if self.training and self.checkpointing:
                x = checkpoint(self.downsample_layers[i], x, use_reentrant=False)
                x = checkpoint(self.stages[i], x, use_reentrant=False)
            else:
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# HiFiGAN-PCPH Head (processes ConvNeXt backbone output with PCPH injection)
# ---------------------------------------------------------------------------

def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def _make_sinc_lowpass(channels: int, kernel_size: int, cutoff: float) -> nn.Conv1d:
    """
    Create a depthwise Conv1d initialized as a Hamming-windowed sinc low-pass filter.

    Args:
        channels:    number of input/output channels (depthwise: groups=channels)
        kernel_size: filter length (odd recommended; 5 gives better rolloff than 3)
        cutoff:      normalized cutoff frequency in (0, 0.5], where 0.5 = Nyquist.
                     For an upsampling stage with rate u, pass cutoff = 0.5 / u.
    """
    conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                     padding=kernel_size // 2, groups=channels, bias=False)

    k = kernel_size
    center = (k - 1) / 2.0
    n = torch.arange(k, dtype=torch.float64) - center  # [-c, ..., 0, ..., c]

    # Sinc kernel: h[n] = 2*fc * sinc(2*pi*fc*n)
    eps = 1e-8
    h = torch.where(
        n.abs() < eps,
        torch.full_like(n, 2.0 * cutoff),
        torch.sin(2.0 * math.pi * cutoff * n) / (math.pi * n + eps)
    )

    # Hamming window: better sidelobe attenuation than Hann for short kernels
    hamming = 0.54 - 0.46 * torch.cos(2.0 * math.pi * torch.arange(k, dtype=torch.float64) / (k - 1))
    h = h * hamming
    h = h / h.sum()  # unity DC gain

    # Broadcast to all channels (depthwise: out_ch=in_ch, each with kernel [1, k])
    h_init = h.float().view(1, 1, k).expand(channels, 1, k).contiguous()

    with torch.no_grad():
        conv.weight.copy_(h_init)

    return conv


class HiFiGANNSFHead(nn.Module):
    """
    HiFiGAN generator head with NSF source injection and SnakeBeta activations.
    Takes backbone features + f0 and generates audio waveform.

    SnakeBeta replaces LeakyReLU in both the upsample loop and the residual
    blocks. The periodic nature of Snake (x + 1/β · sin²(αx)) is fundamentally
    better at modelling audio sinusoidal components than a piecewise-linear
    activation; this is the core architectural improvement from BigVGAN.
    Cost is negligible: two trainable scalars (α, β) per channel per layer.
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
        checkpointing: bool = False,
        pre_conv_kernel_size: int = 13,
        post_conv_kernel_size: int = 13,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        self.lrelu_slope = LRELU_SLOPE

        self.upp = math.prod(upsample_rates)
        # PCPH source: sums all harmonics up to Nyquist — band-limited by construction,
        # no aliasing vs the old single-sine NSF source.
        self.m_source = SourceModulePCPH(
            sample_rate=sr,
            hop_length=self.upp,
            random_init_phase=True,
            power_factor=0.1,
            add_noise_std=0.003,
            use_pchip=True,
        )

        # Pre-conv to map backbone output channels to upsample_initial_channel
        self.conv_pre = weight_norm(nn.Conv1d(
            input_channels, upsample_initial_channel,
            pre_conv_kernel_size, 1,
            padding=get_padding(pre_conv_kernel_size),
        ))

        # SnakeBeta activations before each upsample stage.
        # One per stage, each with its own learnable α/β per channel.
        self.pre_snake = nn.ModuleList()

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        # Learned depthwise anti-aliasing filters after each upsample stage.
        # Acts as a trainable low-pass filter suppressing aliased components
        # introduced by ConvTranspose1d (mirrors/reflections in spectrogram).
        self.anti_alias_convs = nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        stride_f0s = [
            math.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # SnakeBeta activation for this upsample stage's input channels
            self.pre_snake.append(
                SnakeBeta(upsample_initial_channel // (2 ** i),
                          alpha_trainable=True, alpha_logscale=True)
            )

            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    channels[i],
                    k, u,
                    padding=padding,
                    output_padding=u % 2,
                ))
            )

            # Depthwise sinc low-pass filter to suppress aliasing from ConvTranspose1d.
            # Initialized as a Hamming-windowed sinc with cutoff = 0.5/u so it
            # attenuates the aliased spectral copies introduced by upsampling rate u.
            # Initialized BEFORE weight_norm so the parametrization starts from the
            # correct low-pass direction (prevents early convergence to spurious
            # frequencies such as sr/4 = 8 kHz at 32 kHz training).
            aa_kernel = 5  # 5-tap gives ~40 dB rolloff vs ~20 dB for 3-tap
            aa_cutoff = 0.5 / u   # ideal LP cutoff for this upsample stage
            aa_conv = _make_sinc_lowpass(channels[i], aa_kernel, aa_cutoff)
            self.anti_alias_convs.append(weight_norm(aa_conv))

            # NSF source injection convs
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            pad = 0 if stride == 1 else (kernel - stride) // 2
            self.noise_convs.append(
                nn.Conv1d(1, channels[i], kernel_size=kernel, stride=stride, padding=pad)
            )

        # Residual blocks with SnakeBeta activations (BigVGAN-style).
        # ResBlock_SnakeBeta uses per-layer learnable periodic activations
        # instead of LeakyReLU, capturing harmonic structure much better.
        from rvc.lib.algorithm.residuals import ResBlock_SnakeBeta
        self.resblocks = nn.ModuleList([
            ResBlock_SnakeBeta(channels[i], k, d)
            for i in range(len(self.ups))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        # Final SnakeBeta before conv_post
        self.post_snake = SnakeBeta(
            channels[-1], alpha_trainable=True, alpha_logscale=True
        )

        # Post-conv
        self.conv_post = weight_norm(nn.Conv1d(
            channels[-1], 1,
            post_conv_kernel_size, 1,
            padding=get_padding(post_conv_kernel_size),
        ))

        self.ups.apply(init_weights)
        # anti_alias_convs are already initialized as sinc; no random re-init.
        self.conv_post.apply(init_weights)

        # Speaker conditioning
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, har_source: torch.Tensor,
                g: Optional[torch.Tensor] = None):
        """
        Args:
            x: Backbone output [B, C, T]
            har_source: Harmonic source signal [B, 1, T_audio]
            g: Speaker embedding [B, gin_channels, 1]
        """
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = self.pre_snake[i](x)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = checkpoint(self.anti_alias_convs[i], x, use_reentrant=False)
                x = x + noise_convs(har_source)
                xs = sum([
                    checkpoint(resblock, x, use_reentrant=False)
                    for j, resblock in enumerate(self.resblocks)
                    if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                ])
            else:
                x = ups(x)
                x = self.anti_alias_convs[i](x)
                x = x + noise_convs(har_source)
                xs = sum([
                    resblock(x)
                    for j, resblock in enumerate(self.resblocks)
                    if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                ])
            x = xs / self.num_kernels

        x = self.post_snake(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_parametrizations(l)
        for l in self.anti_alias_convs:
            remove_parametrizations(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_parametrizations(self.conv_pre)
        remove_parametrizations(self.conv_post)


# ---------------------------------------------------------------------------
# ChouwaGAN: Full Generator (ConvNeXt backbone + HiFiGAN-NSF head)
# ---------------------------------------------------------------------------

class ChouwaGANGenerator(nn.Module):
    """
    ChouwaGAN with PCPH (Pseudo-Constant-Power Harmonic) source.

    Architecture:
      1. ConvNeXt backbone encodes VITS latent features into a deep representation
      2. HiFiGAN head with PCPH harmonic injection synthesizes the audio waveform
      3. f0 (pitch) drives a band-limited Dirichlet harmonic source (all harmonics
         up to Nyquist) injected at each upsampling stage \u2014 eliminating aliasing
         by construction, unlike the old NSF single-sine approach.

    Interface: forward(x, f0, g=None) \u2014 same as HiFiGANNSFGenerator for RVC compatibility.
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
        checkpointing: bool = False,
        # ConvNeXt backbone config
        backbone_depths: list = [3, 3, 9, 3],
        backbone_dims: list = [128, 256, 384, 512],
        backbone_drop_path_rate: float = 0.2,
        backbone_kernel_size: int = 7,
    ):
        super().__init__()
        self.checkpointing = checkpointing

        # ConvNeXt backbone: maps VITS latent (initial_channel) -> backbone output (backbone_dims[-1])
        self.backbone = ConvNeXtEncoder(
            input_channels=initial_channel,
            depths=backbone_depths,
            dims=backbone_dims,
            drop_path_rate=backbone_drop_path_rate,
            kernel_size=backbone_kernel_size,
            checkpointing=checkpointing,
        )

        backbone_out_channels = backbone_dims[-1]

        # HiFiGAN-NSF head: maps backbone features + f0 harmonic source -> audio
        self.head = HiFiGANNSFHead(
            input_channels=backbone_out_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            gin_channels=gin_channels,
            sr=sr,
            checkpointing=checkpointing,
        )

        # Store total upsampling factor and share source module reference
        self.upp = math.prod(upsample_rates)
        self.m_source = self.head.m_source

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: VITS latent features [B, initial_channel, T]
            f0: Fundamental frequency [B, T_f0]
            g: Speaker embedding [B, gin_channels, 1]

        Returns:
            Audio waveform [B, 1, T_audio]
        """
        # Generate band-limited PCPH harmonic source from f0.
        # SourceModulePCPH returns [B, 1, T_audio] directly (no transpose needed),
        # unlike the old NSF source which returned [B, T_audio, 1].
        har_source = self.m_source(f0, self.upp)  # [B, 1, T_audio]

        # Encode through ConvNeXt backbone
        x = self.backbone(x)

        # Synthesize through HiFiGAN-NSF head
        x = self.head(x, har_source, g=g)

        return x

    def remove_weight_norm(self):
        self.head.remove_weight_norm()
