"""
MS-STFT + MRD Combined Discriminator for ChouwaGAN

Combines:
- Multi-Scale STFT Discriminator (MS-STFT) — frequency-domain, phase-aware
- Multi-Resolution Discriminator (MRD) — spectrogram envelope analysis
- High-Frequency Sub-Band Discriminator — specialized for HF content

Architecture:
- MS-STFT: 32-channel base → max 128 channels (lightweight)
- MRD: 32-channel throughout (very lightweight)
- HF Sub-Band: 32-channel (specialized for 10kHz+ region)
- Total params: ~2M (vs ~8M for MPD+MSD)

Total sub-discriminators: 6 (3 MS-STFT + 2 MRD + 1 HF).

Bug Fixes Applied:
1. HF Discriminator: Separated conv_post from convs to prevent LeakyReLU on prediction layer
2. HF Discriminator: Added torch.flatten to output for shape consistency with other discriminators
3. Fixed sample_rate default from 40000 to 48000 (consistent with generator)
4. Fixed use_checkpointing to use the argument instead of hardcoding False
5. Removed inplace=True from all LeakyReLU calls for gradient checkpointing compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from typing import List, Tuple


def get_norm(use_spectral_norm: bool):
    """Return appropriate normalization function."""
    return spectral_norm if use_spectral_norm else weight_norm


# ══════════════════════════════════════════════════════════════════════════════
# High-Frequency Sub-Band Discriminator
# ══════════════════════════════════════════════════════════════════════════════

class HighFrequencySubBandDiscriminator(nn.Module):
    """
    Isolates the 10 kHz to Nyquist band and discriminates ONLY that region.
    Forces the generator to pay attention to high-frequency content and phase.
    Operates on complex STFT (2 channels: real + imaginary).
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

        # Intermediate convolutions (without prediction layer)
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(2, channels, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        
        # Prediction layer (separate, no activation applied)
        self.conv_post = norm_f(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Compute STFT
        x_squeezed = x.squeeze(1)
        x_float = x_squeezed.float() if x_squeezed.dtype != torch.float32 else x_squeezed
        
        pad = (self.n_fft - self.hop) // 2
        x_float = F.pad(x_float, (pad, pad), mode="reflect")
        stft = torch.stft(
            x_float, self.n_fft, self.hop, self.n_fft,
            window=self.window.to(dtype=x_float.dtype, device=x_float.device), 
            center=False, return_complex=True
        )
        
        # Crop to HF only, then split real/imag
        stft = torch.view_as_real(stft[:, self.freq_start:, :])  # (B, F_hf, T, 2)
        stft = stft.permute(0, 3, 1, 2).contiguous()  # (B, 2, F_hf, T)
        
        # Use channels_last for optimized Conv2D on Ampere+ GPUs
        stft = stft.to(memory_format=torch.channels_last)
        
        # Intermediate convolutions with activation
        fmaps = []
        for conv in self.convs:
            stft = conv(stft)
            stft = F.leaky_relu(stft, 0.1)  # removed inplace for checkpointing compatibility
            fmaps.append(stft)
        
        # Prediction layer (no activation)
        stft = self.conv_post(stft)
        fmaps.append(stft)
        
        # Flatten to match other discriminators' output shape
        stft = torch.flatten(stft, 1, -1)
        
        return stft, fmaps




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
        # Compute complex STFT (real + imag)
        x = self._stft(x)  # (B, 2, F, T')
        
        # Use channels_last for optimized Conv2D on Ampere+ GPUs
        x = x.to(memory_format=torch.channels_last)

        # Convolutions
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)  # removed inplace for checkpointing compatibility
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute complex STFT as 2-channel real tensor (real + imag)."""
        x_squeezed = x.squeeze(1)
        x_float = x_squeezed.float() if x_squeezed.dtype != torch.float32 else x_squeezed
        
        pad = (self.n_fft - self.hop_length) // 2
        x_float = F.pad(x_float, (pad, pad), mode="reflect")
        stft = torch.stft(
            x_float,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x_float.dtype, device=x_float.device),
            center=False,
            return_complex=True,
        )
        
        # Return real+imag as 2 channels (B, 2, F, T')
        return torch.stack([stft.real, stft.imag], dim=1)




# ══════════════════════════════════════════════════════════════════════════════
# MRD Sub-Discriminator
# ══════════════════════════════════════════════════════════════════════════════

class DiscriminatorR(nn.Module):
    """
    Multi-Resolution Discriminator sub-band (UnivNet/BigVGAN style).
    Operates on magnitude spectrogram at a specific (n_fft, hop, win) resolution.
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
        x = self._spectrogram(x).unsqueeze(1)  # (B, 1, F, T')
        
        # Use channels_last for optimized Conv2D on Ampere+ GPUs
        x = x.to(memory_format=torch.channels_last)

        # Convolutions
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)  # removed inplace for checkpointing compatibility
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute magnitude spectrogram."""
        n_fft, hop_length, win_length = self.resolution
        
        x_squeezed = x.squeeze(1)
        x_float = x_squeezed.float() if x_squeezed.dtype != torch.float32 else x_squeezed
        
        pad = (n_fft - hop_length) // 2
        x_float = F.pad(x_float, (pad, pad), mode="reflect")
        stft = torch.stft(
            x_float, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=self.window.to(dtype=x_float.dtype, device=x_float.device), 
            center=False, return_complex=True,
        )
        
        return stft.abs()




# ══════════════════════════════════════════════════════════════════════════════
# Combined Discriminator
# ══════════════════════════════════════════════════════════════════════════════

class MSSTFT_MRD_Combined(nn.Module):
    """
    Combined MS-STFT + MRD discriminator for ChouwaGAN.
    
    Default config: 3 STFT scales + 2 MRD resolutions + 1 HF sub-band = 6 sub-discriminators.
    
    Architecture:
    - MS-STFT uses 32-channel base → max 128 channels (lightweight)
    - MRD uses 32-channel throughout (very lightweight)
    - HF Sub-Band uses 32-channel (specialized)
    - Total params: ~2M (vs ~8M for MPD+MSD)
    
    Config parameters (passed via **cfg from config.mrd):
    - stft_configs: List of dicts with n_fft, hop_length, win_length for MS-STFT
    - stft_channels: Base channel count for MS-STFT (default: 32)
    - stft_n_layers: Number of layers per MS-STFT discriminator (default: 3)
    - mrd_resolutions: List of [n_fft, hop, win] for MRD discriminators
    - mrd_channels: Channel count for MRD (default: 32)
    - hf_disc_channels: Channel count for HF discriminator (default: 32)
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        sample_rate: int = 48000,  # Fixed: consistent with generator default
        **cfg,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing  # Fixed: use the argument instead of hardcoding

        # MS-STFT configuration
        stft_configs = cfg.get("stft_configs", [
            {"n_fft": 2048, "hop_length": 512, "win_length": 2048},
            {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
            {"n_fft": 512,  "hop_length": 128, "win_length": 512},
        ])
        stft_channels = cfg.get("stft_channels", 32)
        stft_n_layers = cfg.get("stft_n_layers", 3)

        # MRD configuration
        mrd_resolutions = cfg.get("mrd_resolutions", [
            [1024, 120, 600],
            [2048, 240, 1200],
        ])
        mrd_channels = cfg.get("mrd_channels", 32)

        # Build unified discriminator list
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
                DiscriminatorR(
                    resolution=res, 
                    channels=mrd_channels, 
                    use_spectral_norm=use_spectral_norm
                )
                for res in mrd_resolutions
            ]
        )
        
        # High Frequency Sub-Band Specialist
        hf_disc_channels = cfg.get("hf_disc_channels", 32)
        self.hf_disc = HighFrequencySubBandDiscriminator(
            sr=sample_rate, 
            channels=hf_disc_channels,
            use_spectral_norm=use_spectral_norm
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
            compute_fmaps: If False, skip collecting feature maps
        
        Returns:
            y_d_rs: List of discriminator scores for real samples
            y_d_gs: List of discriminator scores for generated samples
            fmap_rs: List of feature maps for real samples
            fmap_gs: List of feature maps for generated samples
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        # Process all standard discriminators (MS-STFT + MRD)
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            if compute_fmaps:
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

        # Process High Frequency discriminator
        y_d_r_hf, fmap_r_hf = self.hf_disc(y)
        y_d_g_hf, fmap_g_hf = self.hf_disc(y_hat)
            
        y_d_rs.append(y_d_r_hf)
        y_d_gs.append(y_d_g_hf)
        if compute_fmaps:
            fmap_rs.append(fmap_r_hf)
            fmap_gs.append(fmap_g_hf)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
