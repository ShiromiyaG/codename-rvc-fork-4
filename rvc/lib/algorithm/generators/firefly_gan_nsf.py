"""
FireflyGAN-NSF Generator for RVC.

A ConvNeXt-backbone + HiFiGAN-head vocoder with Neural Source Filter (NSF)
for pitch-conditioned waveform synthesis. Adapted from the fish-vocoder
FireflyGAN architecture with NSF source-filter injection at each upsampling
stage, following the same interface as other RVC generators.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock


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
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


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
# NSF Source Module (same as used in HiFiGAN-NSF)
# ---------------------------------------------------------------------------

class SineGenerator(nn.Module):
    """Generates sine waves at f0 and its harmonics for source-filter synthesis."""

    def __init__(self, sampling_rate: int, num_harmonics: int = 0,
                 sine_amplitude: float = 0.1, noise_stddev: float = 0.003,
                 voiced_threshold: float = 0.0):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.sine_amplitude = sine_amplitude
        self.noise_stddev = noise_stddev
        self.voiced_threshold = voiced_threshold
        self.waveform_dim = self.num_harmonics + 1

    def _compute_voiced_unvoiced(self, f0: torch.Tensor):
        return (f0 > self.voiced_threshold).float()

    def _generate_sine_wave(self, f0: torch.Tensor, upsampling_factor: int):
        batch_size, length, _ = f0.shape

        upsampling_grid = torch.arange(
            1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device
        )
        phase_increments = (f0 / self.sampling_rate) * upsampling_grid
        phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        phase_increments += F.pad(cumulative_phase, (0, 0, 1, 0), mode="constant")
        phase_increments = phase_increments.reshape(batch_size, -1, 1)

        harmonic_scale = torch.arange(
            1, self.waveform_dim + 1, dtype=f0.dtype, device=f0.device
        ).reshape(1, 1, -1)
        phase_increments = phase_increments * harmonic_scale

        random_phase = torch.rand(1, 1, self.waveform_dim, device=f0.device)
        random_phase[..., 0] = 0
        phase_increments = phase_increments + random_phase

        sine_waves = torch.sin(2 * np.pi * phase_increments)
        return sine_waves

    def forward(self, f0: torch.Tensor, upsampling_factor: int):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude

            voiced_mask = self._compute_voiced_unvoiced(f0)
            voiced_mask = F.interpolate(
                voiced_mask.transpose(2, 1),
                scale_factor=float(upsampling_factor),
                mode="nearest",
            ).transpose(2, 1)

            noise_amplitude = voiced_mask * self.noise_stddev + (1 - voiced_mask) * (
                self.sine_amplitude / 3
            )
            noise = noise_amplitude * torch.randn_like(sine_waves)
            sine_waveforms = sine_waves * voiced_mask + noise

        return sine_waveforms, voiced_mask, noise


class SourceModuleHnNSF(nn.Module):
    """Harmonic-plus-Noise Source Module for NSF-based synthesis."""

    def __init__(self, sample_rate: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, add_noise_std: float = 0.003,
                 voiced_threshold: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGenerator(
            sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, upsample_factor: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


# ---------------------------------------------------------------------------
# HiFiGAN-NSF Head (processes ConvNeXt backbone output with NSF injection)
# ---------------------------------------------------------------------------

def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class HiFiGANNSFHead(nn.Module):
    """
    HiFiGAN generator head with NSF source injection.
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
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        # Pre-conv to map backbone output channels to upsample_initial_channel
        self.conv_pre = weight_norm(nn.Conv1d(
            input_channels, upsample_initial_channel,
            pre_conv_kernel_size, 1,
            padding=get_padding(pre_conv_kernel_size),
        ))

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

            # Depthwise conv anti-aliasing filter (groups=channels => near-zero parameter cost)
            self.anti_alias_convs.append(
                weight_norm(nn.Conv1d(
                    channels[i], channels[i],
                    kernel_size=3, stride=1, padding=1,
                    groups=channels[i],
                ))
            )

            # NSF source injection convs
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            pad = 0 if stride == 1 else (kernel - stride) // 2
            self.noise_convs.append(
                nn.Conv1d(1, channels[i], kernel_size=kernel, stride=stride, padding=pad)
            )

        # Residual blocks
        self.resblocks = nn.ModuleList([
            ResBlock(channels[i], k, d)
            for i in range(len(self.ups))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        # Post-conv
        self.conv_post = weight_norm(nn.Conv1d(
            channels[-1], 1,
            post_conv_kernel_size, 1,
            padding=get_padding(post_conv_kernel_size),
        ))

        self.ups.apply(init_weights)
        self.anti_alias_convs.apply(init_weights)
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
            x = F.leaky_relu(x, self.lrelu_slope)

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

        x = F.leaky_relu(x, self.lrelu_slope)
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
# FireflyGAN-NSF: Full Generator (ConvNeXt backbone + HiFiGAN-NSF head)
# ---------------------------------------------------------------------------

class FireflyGANNSFGenerator(nn.Module):
    """
    FireflyGAN with Neural Source Filter (NSF).

    Architecture:
      1. ConvNeXt backbone encodes VITS latent features into a deep representation
      2. HiFiGAN head with NSF harmonic injection synthesizes the audio waveform
      3. f0 (pitch) drives a sine-wave source that is injected at each upsampling stage

    Interface: forward(x, f0, g=None) — same as HiFiGANNSFGenerator for RVC compatibility.
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

        # Store for NSF source generation
        self.upp = math.prod(upsample_rates)
        self.m_source = self.head.m_source  # share source module reference

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
        # Generate harmonic source from f0
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)  # [B, 1, T_audio]

        # Encode through ConvNeXt backbone
        x = self.backbone(x)

        # Synthesize through HiFiGAN-NSF head
        x = self.head(x, har_source, g=g)

        return x

    def remove_weight_norm(self):
        self.head.remove_weight_norm()
