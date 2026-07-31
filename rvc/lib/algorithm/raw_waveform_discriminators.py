"""Waveform discriminators for the raw NSF GAN."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


def _conv2d(in_channels: int, out_channels: int, kernel, stride=(1, 1), groups=1):
    return spectral_norm(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=(kernel[0] // 2, kernel[1] // 2),
            groups=groups,
        )
    )


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, speaker_dim: int = 0):
        super().__init__()
        self.period = int(period)
        channels = [32, 128, 512, 1024, 1024]
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.append(
                _conv2d(
                    in_channels,
                    out_channels,
                    (5, 1),
                    stride=(3, 1),
                )
            )
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)
        self.post = _conv2d(in_channels, 1, (3, 1))
        self.speaker_projection = (
            nn.Linear(speaker_dim, in_channels, bias=False)
            if speaker_dim > 0
            else None
        )

    def forward(self, waveform: torch.Tensor, speaker: torch.Tensor | None = None):
        batch, _, length = waveform.shape
        remainder = length % self.period
        if remainder:
            waveform = F.pad(waveform, (0, self.period - remainder), mode="reflect")
        value = waveform.view(batch, 1, -1, self.period)
        features = []
        for layer in self.layers:
            value = layer(value)
            if isinstance(layer, nn.LeakyReLU):
                features.append(value)
        value = self.post(value)
        if self.speaker_projection is not None and speaker is not None:
            pooled = features[-1].mean(dim=(2, 3))
            value = value + (
                pooled * self.speaker_projection(speaker)
            ).sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        return value.flatten(1), features


class ResolutionDiscriminator(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int, speaker_dim: int = 0):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win = int(win)
        channels = [32, 128, 256, 512, 512]
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(
                _conv2d(
                    in_channels,
                    out_channels,
                    (5, 3),
                    stride=(2, 2),
                )
            )
            blocks.append(nn.LeakyReLU(0.2))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.post = _conv2d(in_channels, 1, (3, 3))
        self.speaker_projection = (
            nn.Linear(speaker_dim, in_channels, bias=False)
            if speaker_dim > 0
            else None
        )

    def forward(self, waveform: torch.Tensor, speaker: torch.Tensor | None = None):
        window = torch.hann_window(
            self.win, device=waveform.device, dtype=waveform.dtype
        )
        spectrum = torch.stft(
            waveform[:, 0].float(),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=window.float(),
            center=True,
            return_complex=True,
        ).abs().unsqueeze(1).to(waveform.dtype)
        value = spectrum
        features = []
        for block in self.blocks:
            value = block(value)
            if isinstance(block, nn.LeakyReLU):
                features.append(value)
        value = self.post(value)
        if self.speaker_projection is not None and speaker is not None:
            pooled = features[-1].mean(dim=(2, 3))
            value = value + (
                pooled * self.speaker_projection(speaker)
            ).sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        return value.flatten(1), features


class RawWaveformDiscriminator(nn.Module):
    def __init__(
        self,
        periods: Iterable[int] = (2, 3, 5, 7, 11),
        resolutions: Iterable[tuple[int, int, int]] = (
            (1024, 256, 1024),
            (2048, 512, 2048),
            (4096, 1024, 4096),
        ),
        speaker_dim: int = 192,
    ):
        super().__init__()
        self.periods = nn.ModuleList(
            [PeriodDiscriminator(period, speaker_dim) for period in periods]
        )
        self.resolutions = nn.ModuleList(
            [ResolutionDiscriminator(*resolution, speaker_dim) for resolution in resolutions]
        )

    def forward(self, waveform: torch.Tensor, speaker: torch.Tensor | None = None):
        logits = []
        features = []
        for discriminator in (*self.periods, *self.resolutions):
            logit, inner = discriminator(waveform, speaker)
            logits.append(logit)
            features.append(inner)
        return logits, features


def discriminator_hinge_loss(real_logits, fake_logits):
    return 0.5 * sum(
        F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
        for real, fake in zip(real_logits, fake_logits)
    ) / max(1, len(real_logits))


def generator_adversarial_loss(fake_logits):
    return -sum(logit.mean() for logit in fake_logits) / max(1, len(fake_logits))


def feature_matching_loss(real_features, fake_features):
    total = None
    count = 0
    for real_group, fake_group in zip(real_features, fake_features):
        for real, fake in zip(real_group, fake_group):
            value = F.l1_loss(fake, real.detach())
            total = value if total is None else total + value
            count += 1
    return total if total is not None else torch.tensor(0.0)

