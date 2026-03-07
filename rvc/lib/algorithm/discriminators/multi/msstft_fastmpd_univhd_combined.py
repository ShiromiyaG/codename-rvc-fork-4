"""
MS-STFT + FastMPD + UnivHD Combined Discriminator for ChouwaGAN / RVC

Combines:
- Multi-Scale STFT Discriminator (MS-STFT) — frequency-domain, phase-aware
- Fast Multi-Period Discriminator (FastMPD) — periodic waveform patterns
- Universal Harmonic Discriminator (UnivHD) — harmonic tracking

Architecture:
- MS-STFT: 32-channel base → max 128 channels (3 scales)
- FastMPD: 32-channel base → max 128 channels (5 periods, 4 layers)
- UnivHD: learnable harmonic filterbank + HCB + MDC (~0.33M params)
- Total params: ~2.7M (vs ~8M for original MPD+MSD)
- Total sub-discriminators: 9 (3 MS-STFT + 5 FastMPD + 1 UnivHD)

FastMPD vs original HiFi-GAN MPD:
- 4 strided layers instead of 5 (receptive field: 161 periods, sufficient for voice)
- max 128 channels instead of 1024 (~215K/period vs ~3.5M/period)
- Total MPD: ~1.08M (vs ~17M original)
- Keeps all 5 periods [2,3,5,7,11] (the periods are cheap, channels are expensive)

MRD: REMOVED — its role (magnitude multi-resolution) is fully covered by MS-STFT.

UnivHD Reference:
  "A Universal Harmonic Discriminator for High-quality GAN-based Vocoder"
  Nan Xu, Zhaolong Huang, Xiao Zeng — arXiv:2512.03486 (ASRU 2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import List, Tuple, Optional


def get_norm(use_spectral_norm: bool):
    """Return appropriate normalization function."""
    return spectral_norm if use_spectral_norm else weight_norm


# ══════════════════════════════════════════════════════════════════════════════
# Fast Multi-Period Discriminator (FastMPD)
# ══════════════════════════════════════════════════════════════════════════════

class DiscriminatorP(nn.Module):
    """
    Lightweight single-period sub-discriminator.

    Reshapes waveform to 2D periodic view: (B, 1, T) → (B, 1, T//p, p)
    Then applies 2D convolutions to compare adjacent periods.

    vs HiFi-GAN original:
    - 4 strided layers instead of 5
    - max 128 channels instead of 1024
    - ~215K params per period (vs ~3.5M original)
    - Same receptive field coverage for voice (RF=161 periods ≈ 37ms at p=11, 48kHz)
    """

    def __init__(
        self,
        period: int,
        channels: int = 32,
        max_channels: int = 128,
        n_layers: int = 4,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = get_norm(use_spectral_norm)

        # Strided convolutions (downsampling along period axis)
        self.convs = nn.ModuleList()
        in_ch = 1
        for i in range(n_layers):
            out_ch = min(channels * (2 ** i), max_channels)
            self.convs.append(
                norm_f(
                    nn.Conv2d(
                        in_ch, out_ch,
                        (kernel_size, 1),
                        stride=(stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
            )
            in_ch = out_ch

        # Non-strided conv (refine features at final resolution)
        self.conv_final = norm_f(
            nn.Conv2d(
                in_ch, in_ch,
                (kernel_size, 1),
                stride=(1, 1),
                padding=(kernel_size // 2, 0),
            )
        )

        # Post conv → 1 channel (real/fake score)
        self.conv_post = norm_f(
            nn.Conv2d(in_ch, 1, (3, 1), padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # Reshape to periodic 2D view: (B, 1, T) → (B, 1, T//p, p)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # Strided convolutions
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)

        # Final non-strided conv
        x = F.leaky_relu(self.conv_final(x), 0.1)
        fmap.append(x)

        # Score prediction
        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap


# ══════════════════════════════════════════════════════════════════════════════
# Universal Harmonic Discriminator (UnivHD)
# ══════════════════════════════════════════════════════════════════════════════

class HarmonicFilterbank(nn.Module):
    """
    Learnable triangular band-pass harmonic filterbank.

    Transforms STFT magnitude into a harmonic tensor [H+1, F, T] with dynamic
    frequency resolution. Low-frequency bands get higher resolution (narrower
    bandwidth) for better pitch modeling, while high-frequency bands get wider
    bandwidth for tracking fast-changing harmonics.

    Includes half-harmonic (h=0.5) for fine-grained sub-fundamental modeling.
    Bandwidth follows ERB scale with a learnable gamma parameter per harmonic.
    """

    def __init__(
        self,
        n_harmonics: int = 10,
        bins_per_octave: int = 24,
        f_min: float = 32.7,
        sample_rate: int = 48000,
        n_fft: int = 2048,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics

        # Harmonics: [0.5, 1, 2, ..., H] — includes half-harmonic
        harmonics = [0.5] + list(range(1, n_harmonics + 1))
        self.register_buffer("harmonics", torch.tensor(harmonics, dtype=torch.float32))
        self.n_total_harmonics = len(harmonics)

        # Max frequency for first harmonic (Nyquist criterion: fs / 2H)
        f_max = sample_rate / (2.0 * n_harmonics)

        # Number of log-spaced frequency bins per harmonic range
        n_bins = int(bins_per_octave * math.log2(f_max / f_min))
        self.n_bins = n_bins

        # Center frequencies for first harmonic: f_c[k] = f_min * 2^(k/B)
        k = torch.arange(n_bins, dtype=torch.float32)
        f_c = f_min * (2.0 ** (k / bins_per_octave))
        self.register_buffer("f_c", f_c)  # (n_bins,)

        # STFT frequency bin centers
        f_stft = torch.arange(n_fft // 2 + 1, dtype=torch.float32) * (
            sample_rate / n_fft
        )
        self.register_buffer("f_stft", f_stft)  # (F_stft,)

        # Learnable bandwidth scaling per harmonic (gamma >= 1, init at 1)
        self.gamma = nn.Parameter(torch.ones(self.n_total_harmonics))

        # Cache for inference
        self._cached_filt: Optional[torch.Tensor] = None

    def _build_filter(self) -> torch.Tensor:
        """Build triangular filterbank tensor."""
        gamma = torch.clamp(self.gamma, min=1.0)  # (H+1,)

        # Center frequencies: (H+1, n_bins)
        centers = self.harmonics.unsqueeze(1) * self.f_c.unsqueeze(0)

        # ERB bandwidth: fb_w = gamma * 24.7 * (4.37 * center / 1000 + 1)
        bw = gamma.unsqueeze(1) * 24.7 * (4.37 * centers / 1000.0 + 1.0)

        # Triangular filter: max(0, 1 - |f_stft - center| / bw)
        diff = torch.abs(
            self.f_stft.unsqueeze(0).unsqueeze(2) - centers.unsqueeze(1)
        )  # (H+1, F_stft, n_bins)
        filt = torch.clamp(1.0 - diff / (bw.unsqueeze(1) + 1e-8), min=0.0)

        return filt

    def forward(self, stft_mag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stft_mag: (B, F_stft, T) magnitude spectrogram
        Returns:
            harmonic_tensor: (B, n_total_harmonics, n_bins, T)
        """
        if self.training:
            filt = self._build_filter()
            self._cached_filt = None  # invalidate cache during training
        else:
            if self._cached_filt is None:
                self._cached_filt = self._build_filter()
            filt = self._cached_filt

        return torch.einsum("hfk,bft->bhkt", filt, stft_mag)


class HybridConvBlock(nn.Module):
    """
    Hybrid Convolution Block (HCB) from UnivHD.

    Combines depthwise separable convolution (intra-harmonic patterns) with
    normal convolution (inter-harmonic relationships). Outputs are summed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (7, 7),
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = get_norm(use_spectral_norm)
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        # DSConv: depthwise (groups=in_channels → intra-harmonic)
        self.dsconv = norm_f(
            nn.Conv2d(
                in_channels, in_channels, kernel_size,
                padding=pad, groups=in_channels,
            )
        )
        # PConv: pointwise 1×1 (channel mixing after DSConv)
        self.pconv = norm_f(nn.Conv2d(in_channels, out_channels, (1, 1)))
        # Normal Conv: inter-harmonic modeling
        self.conv = norm_f(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ds_out = F.leaky_relu(self.dsconv(x), 0.1)
        ds_out = F.leaky_relu(self.pconv(ds_out), 0.1)
        conv_out = F.leaky_relu(self.conv(x), 0.1)
        return ds_out + conv_out


class MultiScaleDilatedConv(nn.Module):
    """
    Multi-Scale Dilated Convolution (MDC) block from UnivHD.

    3 sequential dilated convolutions (d=1,2,4) with residual connection,
    followed by a stride-2 normal convolution for frequency downsampling.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int] = (5, 5),
        dilations: Optional[List[int]] = None,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]

        norm_f = get_norm(use_spectral_norm)

        self.dilated_convs = nn.ModuleList()
        for d in dilations:
            pad = (d * (kernel_size[0] - 1) // 2, d * (kernel_size[1] - 1) // 2)
            self.dilated_convs.append(
                norm_f(
                    nn.Conv2d(
                        channels, channels, kernel_size,
                        dilation=d, padding=pad,
                    )
                )
            )

        # Stride-2 along frequency for downsampling
        pad_ds = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv_ds = norm_f(
            nn.Conv2d(
                channels, channels, kernel_size,
                stride=(2, 1), padding=pad_ds,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for dc in self.dilated_convs:
            x = F.leaky_relu(dc(x), 0.1)
        x = x + residual  # skip connection over dilated stack
        x = F.leaky_relu(self.conv_ds(x), 0.1)
        return x


class UniversalHarmonicDiscriminator(nn.Module):
    """
    Universal Harmonic Discriminator (UnivHD).

    Pipeline: Waveform → STFT → HarmonicFilterbank → HCB → MDC×3 → FinalConv → Score
    Parameters: ~0.33M
    """

    def __init__(
        self,
        sr: int = 48000,
        n_harmonics: int = 10,
        bins_per_octave: int = 24,
        channels: int = 32,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 32.7,
        n_mdc: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer("window", torch.hann_window(n_fft))

        # Harmonic filterbank
        self.filterbank = HarmonicFilterbank(
            n_harmonics=n_harmonics,
            bins_per_octave=bins_per_octave,
            f_min=f_min,
            sample_rate=sr,
            n_fft=n_fft,
        )

        n_total_harmonics = self.filterbank.n_total_harmonics

        # HCB: harmonics → channels
        self.hcb = HybridConvBlock(
            n_total_harmonics, channels,
            kernel_size=(7, 7),
            use_spectral_norm=use_spectral_norm,
        )

        # MDC blocks
        self.mdc_blocks = nn.ModuleList(
            [
                MultiScaleDilatedConv(
                    channels,
                    kernel_size=(5, 5),
                    use_spectral_norm=use_spectral_norm,
                )
                for _ in range(n_mdc)
            ]
        )

        # Compute freq dimension after MDC stride-2 downsampling
        freq_dim = self.filterbank.n_bins
        mdc_k = 5
        for _ in range(n_mdc):
            pad = mdc_k // 2
            freq_dim = (freq_dim + 2 * pad - mdc_k) // 2 + 1

        # Final conv: kernel spans entire remaining freq dimension
        norm_f = get_norm(use_spectral_norm)
        self.conv_post = norm_f(
            nn.Conv2d(channels, 1, (freq_dim, 3), padding=(0, 1))
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        stft_mag = self._stft_magnitude(x)
        h_tensor = self.filterbank(stft_mag)
        h_tensor = h_tensor.to(memory_format=torch.channels_last)

        fmaps = []

        x = self.hcb(h_tensor)
        fmaps.append(x)

        for mdc in self.mdc_blocks:
            x = mdc(x)
            fmaps.append(x)

        x = self.conv_post(x)
        fmaps.append(x)

        return torch.flatten(x, 1, -1), fmaps

    def _stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude spectrogram."""
        x_squeezed = x.squeeze(1)
        x_float = (
            x_squeezed.float()
            if x_squeezed.dtype != torch.float32
            else x_squeezed
        )

        pad = (self.n_fft - self.hop_length) // 2
        x_float = F.pad(x_float, (pad, pad), mode="constant")
        stft = torch.stft(
            x_float,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=x_float.dtype, device=x_float.device),
            center=False,
            return_complex=True,
        )

        return stft.abs()


# ══════════════════════════════════════════════════════════════════════════════
# MS-STFT Sub-Discriminator
# ══════════════════════════════════════════════════════════════════════════════

class DiscriminatorSTFT(nn.Module):
    """
    Single-scale STFT discriminator (EnCodec / DAC style).
    Operates on complex STFT (real + imaginary) for phase-aware discrimination.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        channels: int = 32,
        n_layers: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.register_buffer("window", torch.hann_window(win_length))
        self.convs = nn.ModuleList()
        norm_f = get_norm(use_spectral_norm)

        # First conv: 2 channels (real + imag) → channels
        self.convs.append(
            norm_f(nn.Conv2d(2, channels, (3, 9), padding=(1, 4)))
        )

        # Intermediate convs with stride-2 along time
        in_ch = channels
        for i in range(n_layers - 1):
            out_ch = min(channels * (2 ** (i + 1)), 128)
            self.convs.append(
                norm_f(
                    nn.Conv2d(
                        in_ch, out_ch, (3, 9),
                        stride=(1, 2), padding=(1, 4),
                    )
                )
            )
            in_ch = out_ch

        # Final 3×3 conv
        self.convs.append(
            norm_f(nn.Conv2d(in_ch, in_ch, (3, 3), padding=(1, 1)))
        )

        # Post conv → 1 channel (prediction)
        self.conv_post = norm_f(
            nn.Conv2d(in_ch, 1, (3, 3), padding=(1, 1))
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self._stft(x)
        x = x.to(memory_format=torch.channels_last)

        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute complex STFT as 2-channel real tensor (real + imag)."""
        x_squeezed = x.squeeze(1)
        x_float = (
            x_squeezed.float()
            if x_squeezed.dtype != torch.float32
            else x_squeezed
        )

        pad = (self.n_fft - self.hop_length) // 2
        x_float = F.pad(x_float, (pad, pad), mode="constant")
        stft = torch.stft(
            x_float,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x_float.dtype, device=x_float.device),
            center=False,
            return_complex=True,
        )

        return torch.stack([stft.real, stft.imag], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Combined Discriminator
# ══════════════════════════════════════════════════════════════════════════════

class ChouwaGANDiscriminator(nn.Module):
    """
    Combined MS-STFT + FastMPD + UnivHD discriminator.

    Default: 3 STFT scales + 5 MPD periods + 1 UnivHD = 9 sub-discriminators.

    Each component provides unique, non-overlapping discriminative signal:
    - MS-STFT: "the frequency content and phase alignment should match"
    - FastMPD: "the periodic waveform structure should match sample-by-sample"
    - UnivHD:  "the harmonic relationships and envelope should match"

    Total: ~2.7M params (vs ~8M for original HiFi-GAN MPD+MSD)

    Config parameters (passed via **cfg):
    - stft_configs: List of dicts with n_fft, hop_length, win_length
    - stft_channels: Base channel count for MS-STFT (default: 32)
    - stft_n_layers: Layers per MS-STFT discriminator (default: 3)
    - mpd_periods: List of periods for FastMPD (default: [2,3,5,7,11])
    - mpd_channels: Base channel count for FastMPD (default: 32)
    - mpd_max_channels: Max channels for FastMPD (default: 128)
    - mpd_n_layers: Strided layers per MPD sub-disc (default: 4)
    - univhd_*: UnivHD configuration (see UniversalHarmonicDiscriminator)
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        sample_rate: int = 48000,
        **cfg,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # ── MS-STFT ──────────────────────────────────────────────────────
        stft_configs = cfg.get(
            "stft_configs",
            [
                {"n_fft": 2048, "hop_length": 512, "win_length": 2048},
                {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
                {"n_fft": 512, "hop_length": 128, "win_length": 512},
            ],
        )
        stft_channels = cfg.get("stft_channels", 32)
        stft_n_layers = cfg.get("stft_n_layers", 3)

        # ── FastMPD ──────────────────────────────────────────────────────
        mpd_periods = cfg.get("mpd_periods", [2, 3, 5, 7, 11])
        mpd_channels = cfg.get("mpd_channels", 32)
        mpd_max_channels = cfg.get("mpd_max_channels", 128)
        mpd_n_layers = cfg.get("mpd_n_layers", 4)

        # ── Build unified discriminator list ─────────────────────────────
        self.discriminators = nn.ModuleList(
            # MS-STFT sub-discriminators
            [
                DiscriminatorSTFT(
                    n_fft=sc["n_fft"],
                    hop_length=sc["hop_length"],
                    win_length=sc["win_length"],
                    channels=stft_channels,
                    n_layers=stft_n_layers,
                    use_spectral_norm=use_spectral_norm,
                )
                for sc in stft_configs
            ]
            # FastMPD sub-discriminators
            + [
                DiscriminatorP(
                    period=p,
                    channels=mpd_channels,
                    max_channels=mpd_max_channels,
                    n_layers=mpd_n_layers,
                    use_spectral_norm=use_spectral_norm,
                )
                for p in mpd_periods
            ]
        )

        # ── UnivHD ───────────────────────────────────────────────────────
        self.univhd = UniversalHarmonicDiscriminator(
            sr=sample_rate,
            n_harmonics=cfg.get("univhd_n_harmonics", 10),
            bins_per_octave=cfg.get("univhd_bins_per_octave", 24),
            channels=cfg.get("univhd_channels", 32),
            n_fft=cfg.get("univhd_n_fft", 2048),
            hop_length=cfg.get("univhd_hop_length", 512),
            use_spectral_norm=use_spectral_norm,
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        compute_fmaps: bool = True,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """
        Forward pass through all sub-discriminators.

        Args:
            y: Real waveform (B, 1, T)
            y_hat: Generated waveform (B, 1, T)
            compute_fmaps: If False, skip collecting feature maps (D step)

        Returns:
            y_d_rs: List of discriminator scores for real samples
            y_d_gs: List of discriminator scores for generated samples
            fmap_rs: List of feature maps for real (empty if compute_fmaps=False)
            fmap_gs: List of feature maps for generated (empty if compute_fmaps=False)
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        use_ckpt = self.use_checkpointing and self.training

        # ── MS-STFT + FastMPD ────────────────────────────────────────────
        for d in self.discriminators:
            if use_ckpt:
                y_d_r, fmap_r = grad_checkpoint(
                    d, y, use_reentrant=False
                )
                y_d_g, fmap_g = grad_checkpoint(
                    d, y_hat, use_reentrant=False
                )
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            if compute_fmaps:
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

        # ── UnivHD ───────────────────────────────────────────────────────
        if use_ckpt:
            y_d_r_h, fmap_r_h = grad_checkpoint(
                self.univhd, y, use_reentrant=False
            )
            y_d_g_h, fmap_g_h = grad_checkpoint(
                self.univhd, y_hat, use_reentrant=False
            )
        else:
            y_d_r_h, fmap_r_h = self.univhd(y)
            y_d_g_h, fmap_g_h = self.univhd(y_hat)

        y_d_rs.append(y_d_r_h)
        y_d_gs.append(y_d_g_h)
        if compute_fmaps:
            fmap_rs.append(fmap_r_h)
            fmap_gs.append(fmap_g_h)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
