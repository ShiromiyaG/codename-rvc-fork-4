import torch
import torch.nn.functional as F

import torch.nn as nn

from torch.nn import Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from torchaudio.transforms import Spectrogram, Resample

import typing
from typing import Optional, List, Union, Dict, Tuple

from torch.utils.checkpoint import checkpoint
from rvc.train.utils import AttrDict

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE


class MPD_MSD_MRD_Combined(torch.nn.Module):
    """
    Class combining:
    Multi-Period, Multi-Scale and Multi-Resolution Discriminators.
    Optionally includes a High-Band Discriminator for 8-16 kHz.
    """

    def __init__(self, use_spectral_norm: bool = False, use_checkpointing: bool = False, **multi_resolution_cfg):
        super().__init__()
        self.mrd_cfg = multi_resolution_cfg
        self.use_checkpointing = use_checkpointing

        periods = self.mrd_cfg.get("periods", [2, 3, 5, 7, 11]) # [2, 3, 5, 7, 11, 17, 23, 37]  -  MPD carry style
        mrd_d_mult = float(self.mrd_cfg.get("mrd_d_mult", 1.0))
        use_highband = bool(self.mrd_cfg.get("use_highband", False))

        self.resolutions = self.mrd_cfg["resolutions"]

        assert len(self.resolutions) >= 1, \
            f"MRD requires at least one resolution triplet. Got {self.resolutions}"


        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
            + [DiscriminatorR(self.mrd_cfg, resolution, d_mult=mrd_d_mult) for resolution in self.resolutions]
        )

        # Optional: High-Band Discriminator for sharper 8-16 kHz learning
        if use_highband:
            self.discriminators.append(DiscriminatorHB())

    def forward(self, y, y_hat, compute_fmaps: bool = True):
        """Forward pass through all sub-discriminators.

        Args:
            y: real waveform
            y_hat: generated waveform
            compute_fmaps: If False, returns empty lists for fmap_rs/fmap_gs.
                Use False during the D training step (fmaps are not needed
                for the discriminator loss, only for generator feature-
                matching loss).  Skipping fmap references lets PyTorch
                free intermediate activations sooner, reducing peak VRAM.
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

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorR(nn.Module):
    def __init__(self, cfg: AttrDict, resolution: List[List[int]], d_mult: float = 1.0):
        super().__init__()
        self.cfg = cfg

        self.resolution = resolution
        assert len(self.resolution) == 3, f"MRD layer requires list with len=3, got {self.resolution}"

        self.lrelu_slope = 0.1
        self.d_mult = d_mult

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        window = torch.hann_window(win_length, device=x.device)
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class DiscriminatorHB(nn.Module):
    """High-Band Discriminator: STFT → crop upper-half bins → conv2d.

    Focuses exclusively on the 8–16 kHz band (at 32 kHz SR) where the
    generator tends to hallucinate phantom harmonics.  Uses n_fft=1024
    for 31.25 Hz/bin resolution, then keeps only bins 256-512.
    Lightweight: ~190 K params (vs ~2 M for full 5-resolution MRD).
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        ch = 32
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1,  ch, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self._highband_spectrogram(x)
        x = x.unsqueeze(1)  # (B, 1, F_high, T)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def _highband_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=x.device)
        pad = (self.n_fft - self.hop_length) // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        x = x.squeeze(1)
        stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            center=False, return_complex=True,
        )
        mag = stft.abs()  # (B, n_fft//2+1, T)
        # Keep only upper half of frequency bins (8-16 kHz at 32 kHz SR)
        n_bins = mag.shape[1]
        return mag[:, n_bins // 2 :, :]
