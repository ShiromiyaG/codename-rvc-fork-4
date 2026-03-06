"""
MS-STFT + MRD Combined Discriminator for ChouwaGAN.

Combines:
- N× MS-STFT (Multi-Scale STFT Discriminator) — frequency-domain, phase-aware
- N× MRD     (Multi-Resolution Discriminator)  — spectrogram envelope

Both are lightweight and complementary:
- MS-STFT catches phase coherence / harmonic structure
- MRD catches spectral envelope at multiple resolutions

Total sub-discriminators: 5 (3 MS-STFT + 2 MRD) — balanced for 8GB VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple

# Pick BF16 when available (Ampere+), else FP16
_AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

def get_norm(use_spectral_norm: bool):
    return spectral_norm if use_spectral_norm else weight_norm


# ─── High-Frequency Sub-Band Discriminator ────────────────────────────────────

class HighFrequencySubBandDiscriminator(nn.Module):
    """
    Isolates the 10 kHz to Nyquist band and discriminates ONLY that region.
    Forces the generator to pay attention to high-frequency content and phase.
    Operates on complex STFT (2 channels: real + imag).
    """
    def __init__(self, sr=48000, channels=32, use_spectral_norm=False):
        super().__init__()
        self.sr = sr
        self.n_fft = 1024
        self.hop = 256
        
        # Band of interest: 10 kHz to Nyquist
        self.freq_start = int(10000 / (sr / self.n_fft))
        
        norm_f = get_norm(use_spectral_norm)

        self.register_buffer("window", torch.hann_window(self.n_fft))

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(2, channels, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
            norm_f(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1))),
        ])

    @torch._dynamo.disable(recursive=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmaps = []
        x = x.squeeze(1).float()
        
        # Pad to center
        pad = (self.n_fft - self.hop) // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        
        stft = torch.stft(
            x, self.n_fft, self.hop, self.n_fft,
            window=self.window.to(dtype=x.dtype, device=x.device), 
            center=False, return_complex=True
        )
        
        # stft: (B, F, T) complex → crop to HF only, then split real/imag
        stft = torch.view_as_real(stft[:, self.freq_start:, :])  # (B, F_hf, T, 2)
        stft = stft.permute(0, 3, 1, 2)  # (B, 2, F_hf, T)
        
        # Convolutions in BF16 for speed+VRAM savings
        with torch.autocast(device_type="cuda", dtype=_AMP_DTYPE):
            for conv in self.convs:
                stft = conv(stft)
                stft = F.leaky_relu(stft, 0.1)
                fmaps.append(stft)
            
        return stft, fmaps


# ─── MS-STFT Sub-Discriminator ────────────────────────────────────────────────

class DiscriminatorSTFT(nn.Module):
    """Single-scale STFT discriminator (EnCodec / DAC style).

    Operates on STFT magnitude spectrogram with 2D convolutions.
    Lightweight: uses small channel counts (32-128) with weight norm.
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

        # Input: magnitude spectrogram (B, 1, F, T)
        # F = n_fft // 2 + 1
        self.convs = nn.ModuleList()

        norm_f = get_norm(use_spectral_norm)

        # First conv: 1 → channels
        self.convs.append(
            norm_f(nn.Conv2d(1, channels, (3, 9), padding=(1, 4)))
        )

        # Intermediate convs with stride-2 along time
        in_ch = channels
        for i in range(n_layers - 1):
            out_ch = min(channels * (2 ** (i + 1)), 128)
            self.convs.append(
                norm_f(nn.Conv2d(
                    in_ch, out_ch,
                    (3, 9), stride=(1, 2), padding=(1, 4)
                ))
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: waveform (B, 1, T)
        Returns:
            prediction: (B, T') flattened
            fmap: list of intermediate feature maps
        """
        fmap = []

        # Compute magnitude spectrogram (float32)
        mag = self._stft(x)       # (B, F, T')
        x = mag.unsqueeze(1)      # (B, 1, F, T')

        # Convolutions in BF16 for speed+VRAM savings
        with torch.autocast(device_type="cuda", dtype=_AMP_DTYPE):
            for conv in self.convs:
                x = conv(x)
                x = F.leaky_relu(x, 0.1)
                fmap.append(x)

            x = self.conv_post(x)
            fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    @torch._dynamo.disable(recursive=False)
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute magnitude spectrogram."""
        x = x.squeeze(1).float()  # (B, T) — STFT requires float32
        # Pad to center
        pad = (self.n_fft - self.hop_length) // 2
        x = F.pad(x, (pad, pad), mode="reflect")

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x.dtype, device=x.device),
            center=False,
            return_complex=True,
        )
        # Magnitude
        mag = stft.abs()  # (B, F, T')
        return mag


# ─── MRD Sub-Discriminator ────────────────────────────────────────────────────

class DiscriminatorR(nn.Module):
    """Multi-Resolution Discriminator sub-band (UnivNet/BigVGAN style).

    Operates on magnitude spectrogram at a specific (n_fft, hop, win) resolution.
    Uses small 2D convolutions (32 channels) — very lightweight.
    """

    def __init__(self, resolution: List[int], channels: int = 32, use_spectral_norm: bool = False):
        super().__init__()
        self.resolution = resolution
        assert len(self.resolution) == 3, f"MRD needs [n_fft, hop, win], got {self.resolution}"
        
        n_fft, hop, win = self.resolution
        self.register_buffer("window", torch.hann_window(win))
        
        norm_f = get_norm(use_spectral_norm)

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(
            nn.Conv2d(channels, 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self._spectrogram(x).unsqueeze(1)  # (B, 1, F, T')

        # Convolutions in BF16 for speed+VRAM savings
        with torch.autocast(device_type="cuda", dtype=_AMP_DTYPE):
            for conv in self.convs:
                x = conv(x)
                x = F.leaky_relu(x, 0.1)
                fmap.append(x)

            x = self.conv_post(x)
            fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    @torch._dynamo.disable(recursive=False)
    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        x = x.squeeze(1).float()  # STFT requires float32
        pad = (n_fft - hop_length) // 2
        x = F.pad(x, (pad, pad), mode="reflect")

        stft = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=self.window.to(dtype=x.dtype, device=x.device), center=False, return_complex=True,
        )
        return stft.abs()


# ─── Combined Discriminator ──────────────────────────────────────────────────

class MSSTFT_MRD_Combined(nn.Module):
    """Combined MS-STFT + MRD discriminator for ChouwaGAN.

    Default config: 3 STFT scales + 2 MRD resolutions = 5 sub-discriminators.

    Architecture balance:
    - MS-STFT uses 32-channel base → max 128 channels (lightweight)
    - MRD uses 32-channel throughout (very lightweight)
    - Total params: ~2M (vs ~8M for MPD+MSD, ~4M for FastMPD+CQT)

    Both domains are frequency-based but complementary:
    - MS-STFT: fine-grained magnitude at multiple window sizes
    - MRD: envelope shape at multiple resolutions
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        sample_rate: int = 40000,
        **cfg,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # ── MS-STFT config ─────────────────────────────────────────
        stft_configs = cfg.get("stft_configs", [
            {"n_fft": 2048, "hop_length": 512, "win_length": 2048},
            {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
            {"n_fft": 512,  "hop_length": 128, "win_length": 512},
            # HF coverage handled by HighFrequencySubBandDiscriminator
        ])
        stft_channels = cfg.get("stft_channels", 32)
        stft_n_layers = cfg.get("stft_n_layers", 3)

        # ── MRD config ──────────────────────────────────────────────
        # Each resolution is [n_fft, hop_length, win_length]
        mrd_resolutions = cfg.get("mrd_resolutions", [
            [1024, 120, 600],
            [2048, 240, 1200],
        ])
        mrd_channels = cfg.get("mrd_channels", 32)

        # ── Build unified discriminator list ────────────────────────
        self.discriminators = nn.ModuleList(
            # N× MS-STFT
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
            # N× MRD
            + [
                DiscriminatorR(resolution=res, channels=mrd_channels, use_spectral_norm=use_spectral_norm)
                for res in mrd_resolutions
            ]
        )
        
        # ── High Frequency Sub-Band Specialist ────────────────────────
        hf_disc_channels = cfg.get("hf_disc_channels", 32)
        self.hf_disc = HighFrequencySubBandDiscriminator(
            sr=sample_rate, 
            channels=hf_disc_channels,
            use_spectral_norm=use_spectral_norm
        )

    @torch._dynamo.disable
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
        """Forward pass through all sub-discriminators.

        Args:
            y:     real waveform  (B, 1, T)
            y_hat: generated waveform  (B, 1, T)
            compute_fmaps: If False, skip collecting feature maps.

        Returns:
            y_d_rs, y_d_gs, fmap_rs, fmap_gs
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            if self.training and self.use_checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            if compute_fmaps:
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

        # High Frequency discriminator
        if self.training and self.use_checkpointing:
            y_d_r_hf, fmap_r_hf = checkpoint(self.hf_disc, y, use_reentrant=False)
            y_d_g_hf, fmap_g_hf = checkpoint(self.hf_disc, y_hat, use_reentrant=False)
        else:
            y_d_r_hf, fmap_r_hf = self.hf_disc(y)
            y_d_g_hf, fmap_g_hf = self.hf_disc(y_hat)
            
        y_d_rs.append(y_d_r_hf)
        y_d_gs.append(y_d_g_hf)
        if compute_fmaps:
            fmap_rs.append(fmap_r_hf)
            fmap_gs.append(fmap_g_hf)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
