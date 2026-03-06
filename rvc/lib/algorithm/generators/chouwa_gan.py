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
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock_SnakeBeta
from rvc.lib.algorithm.conformer.activations import SnakeBeta
from rvc.lib.algorithm.generators.pcph_gan import SourceModulePCPH


# ---------------------------------------------------------------------------
# ConvNeXt Backbone
# ---------------------------------------------------------------------------


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2) in channels-last format (B, T, C).
    Normalises each channel's response relative to the global aggregate.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, channels))
        self.beta  = nn.Parameter(torch.zeros(1, 1, channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=2, keepdim=True) + self.eps)
        return x * nx * self.gamma + self.beta


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
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            orig_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None].float() * x + self.bias[:, None].float()
            return x.to(orig_dtype)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block: DwConv -> Permute -> LayerNorm -> Linear -> GELU -> GRN -> Linear -> Permute
    """

    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6, 
                 mlp_ratio: float = 4.0, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim, bias=True,
        )
        self.norm = ConvNeXtLayerNorm(dim, eps=1e-6)
        
        hidden_dim = int(mlp_ratio * dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )

    def forward(self, x, apply_residual: bool = True):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.transpose(1, 2)
        
        if apply_residual:
            x = residual + x
        return x


class ConvNeXtEncoder(nn.Module):
    """Multi-stage ConvNeXt encoder."""

    def __init__(
        self,
        input_channels: int = 192,
        depths: list = [3, 3, 9, 3],
        dims: list = [128, 256, 384, 512],
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode="zeros"),
            ConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                ConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
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
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage)

        self.norm = ConvNeXtLayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
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


def init_weights(m, mean=0.0, std=0.01):
    """Initialize weights with normal distribution."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _make_sinc_lowpass(channels: int, kernel_size: int, cutoff: float, freeze: bool = True) -> nn.Conv1d:
    """
    Create a depthwise Conv1d initialized as a Blackman-windowed sinc low-pass filter.
    Uses Blackman window for better stopband attenuation (-74dB vs -43dB Hamming).
    
    Args:
        channels: Number of channels
        kernel_size: Filter kernel size
        cutoff: Normalized cutoff frequency (0 to 0.5)
        freeze: If True, freeze filter weights to preserve anti-aliasing properties
    """
    conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                     padding=kernel_size // 2, groups=channels, bias=False)

    k = kernel_size
    center = (k - 1) / 2.0
    n = torch.arange(k, dtype=torch.float64) - center

    eps = 1e-8
    h = torch.where(
        n.abs() < eps,
        torch.full_like(n, 2.0 * cutoff),
        torch.sin(2.0 * math.pi * cutoff * n) / (math.pi * n)
    )

    blackman_n = torch.arange(k, dtype=torch.float64)
    blackman = (0.42 
                - 0.5 * torch.cos(2.0 * math.pi * blackman_n / (k - 1))
                + 0.08 * torch.cos(4.0 * math.pi * blackman_n / (k - 1)))
    
    h = h * blackman
    h = h / h.sum()

    h_init = h.float().view(1, 1, k).expand(channels, 1, k).contiguous()

    with torch.no_grad():
        conv.weight.copy_(h_init)
    
    if freeze:
        conv.weight.requires_grad_(False)

    return conv


class HiFiGANNSFHead(nn.Module):
    """
    HiFiGAN generator head with NSF source injection and SnakeBeta activations.
    Takes backbone features + f0 and generates audio waveform.
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
        pre_conv_kernel_size: int = 13,
        post_conv_kernel_size: int = 13,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.upp = math.prod(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(
            input_channels, upsample_initial_channel,
            pre_conv_kernel_size, 1,
            padding=get_padding(pre_conv_kernel_size),
        ))

        self.pre_snake = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.har_convs = nn.ModuleList()
        self.anti_alias_convs = nn.ModuleList()

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

            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    self.channels[i],
                    k, u,
                    padding=padding,
                    output_padding=u % 2,
                ))
            )

            # Scaled anti-alias kernel: larger for high upsampling rates, smaller for low
            # u=8 → 17 taps (excellent stopband), u=2 → 5 taps (sufficient, no overhead)
            aa_kernel = max(5, u * 2 + 1)
            if u <= 2:
                aa_cutoff = 0.45 / u
            else:
                aa_cutoff = 0.5 / u
                
            aa_conv = _make_sinc_lowpass(self.channels[i], aa_kernel, aa_cutoff, freeze=True)
            self.anti_alias_convs.append(aa_conv)

            stride = self.stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            pad = 0 if stride == 1 else (kernel - stride) // 2
            
            self.har_convs.append(
                nn.Conv1d(1, self.channels[i], kernel_size=kernel, stride=stride, padding=pad)
            )

        self.resblocks = nn.ModuleList([
            ResBlock_SnakeBeta(self.channels[i], k, d, post_act=False)
            for i in range(len(self.ups))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        self.post_snake = SnakeBeta(
            self.channels[-1], alpha_trainable=True, alpha_logscale=True
        )

        self.conv_post = weight_norm(nn.Conv1d(
            self.channels[-1], 1,
            post_conv_kernel_size, 1,
            padding=get_padding(post_conv_kernel_size),
        ))

        self.ups.apply(init_weights)
        
        # Zero-init conv_post for stable training start (generator begins emitting silence)
        nn.init.zeros_(self.conv_post.weight)
        nn.init.zeros_(self.conv_post.bias)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
            nn.init.normal_(self.cond.weight, mean=0.0, std=0.01)
            if self.cond.bias is not None:
                nn.init.constant_(self.cond.bias, 0.0)

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

        for i in range(self.num_upsamples):
            x = self.pre_snake[i](x)
            x = self.ups[i](x)
            x = self.anti_alias_convs[i](x)
            
            har_out = self.har_convs[i](har_source)
            min_len = min(x.shape[-1], har_out.shape[-1])
            x = x[..., :min_len] + har_out[..., :min_len]
            
            start_idx = i * self.num_kernels
            xs = sum(self.resblocks[start_idx + j](x) for j in range(self.num_kernels))
            
            x = xs / self.num_kernels

        x = self.post_snake(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_parametrizations(self.conv_pre)
        
        for l in self.ups:
            remove_parametrizations(l)
        
        for l in self.resblocks:
            l.remove_weight_norm()
        
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
        backbone_depths: list = [3, 3, 4, 3],
        backbone_dims: list = [96, 192, 256, 320],
        backbone_kernel_size: int = 7,
    ):
        super().__init__()

        self.backbone = ConvNeXtEncoder(
            input_channels=initial_channel,
            depths=backbone_depths,
            dims=backbone_dims,
            kernel_size=backbone_kernel_size,
        )

        backbone_out_channels = backbone_dims[-1]

        self.head = HiFiGANNSFHead(
            input_channels=backbone_out_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            gin_channels=gin_channels,
            sr=sr,
        )

        self.upp = math.prod(upsample_rates)
        
        self.m_source = SourceModulePCPH(
            sample_rate=sr,
            hop_length=self.upp,
            random_init_phase=True,
            power_factor=0.1,
            add_noise_std=0.003,
            use_pchip=True,
        )


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
        if x.shape[-1] != f0.shape[-1]:
            raise ValueError(
                f"Temporal dimension mismatch: x has {x.shape[-1]} frames, "
                f"f0 has {f0.shape[-1]} frames. They must match."
            )
        
        har_source = self.m_source(f0, self.upp)
        x = self.backbone(x)
        x = self.head(x, har_source, g=g)
        return x

    def remove_weight_norm(self):
        self.head.remove_weight_norm()
