"""Inference-only pc-NSF-HiFiGAN used by the mel-VITS pipeline.

The implementation follows SingingVocoders' exported generator format.  The
vocoder is intentionally kept outside the acoustic model checkpoint: it is a
fixed renderer shared by every voice model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm


LRELU_SLOPE = 0.1


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _init_weights(module: nn.Module) -> None:
    if "Conv" in module.__class__.__name__:
        nn.init.normal_(module.weight, 0.0, 0.01)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=_padding(kernel_size, d),
                    )
                )
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=_padding(kernel_size),
                    )
                )
                for _ in dilations
            ]
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = F.leaky_relu(x, LRELU_SLOPE)
            residual = conv1(residual)
            residual = F.leaky_relu(residual, LRELU_SLOPE)
            x = x + conv2(residual)
        return x

    def remove_weight_norm(self) -> None:
        for layer in (*self.convs1, *self.convs2):
            remove_weight_norm(layer)


class _SineGenerator(nn.Module):
    def __init__(self, sample_rate: int, harmonics: int = 8):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonics = harmonics

    @torch.no_grad()
    def forward(self, f0: torch.Tensor, upsample_factor: int) -> torch.Tensor:
        f0 = f0.unsqueeze(-1)
        steps = torch.arange(
            1, upsample_factor + 1, device=f0.device, dtype=f0.dtype
        )
        phase = f0 / self.sample_rate * steps
        phase_end = torch.remainder(phase[..., -1:].float() + 0.5, 1.0) - 0.5
        phase_acc = phase_end.cumsum(dim=1).remainder(1.0).to(f0)
        phase = phase + F.pad(phase_acc[:, :-1], (0, 0, 1, 0))
        phase = phase.reshape(f0.shape[0], -1, 1)
        harmonics = torch.arange(
            1, self.harmonics + 2, device=f0.device, dtype=f0.dtype
        ).view(1, 1, -1)
        phase = phase * harmonics
        voiced = F.interpolate(
            (f0 > 0).float().transpose(1, 2),
            scale_factor=upsample_factor,
            mode="nearest",
        ).transpose(1, 2)
        sine = torch.sin(2 * np.pi * phase) * 0.1
        noise_scale = voiced * 0.003 + (1.0 - voiced) * (0.1 / 3.0)
        return sine * voiced + torch.randn_like(sine) * noise_scale


class _SourceModule(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.l_sin_gen = _SineGenerator(sample_rate)
        self.l_linear = nn.Linear(9, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor, upsample_factor: int) -> torch.Tensor:
        return self.l_tanh(self.l_linear(self.l_sin_gen(f0, upsample_factor)))


class PCNSFHiFiGAN(nn.Module):
    """Mel + frame-level F0 to waveform generator."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.sample_rate = int(config["sampling_rate"])
        self.hop_size = int(config["hop_size"])
        self.num_mels = int(config["num_mels"])
        self.upsample_rates = list(config["upsample_rates"])
        self.num_kernels = len(config["resblock_kernel_sizes"])
        if str(config.get("resblock", "1")) != "1":
            raise ValueError("Only SingingVocoders ResBlock1 exports are supported")
        self.mini_nsf = bool(config.get("mini_nsf", False))
        self.noise_sigma = float(config.get("noise_sigma", 0.0))
        if int(np.prod(self.upsample_rates)) != self.hop_size:
            raise ValueError("pc-NSF upsample_rates product must equal hop_size")

        initial = int(config["upsample_initial_channel"])
        self.conv_pre = weight_norm(nn.Conv1d(self.num_mels, initial, 7, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        if self.mini_nsf:
            self.source_sr = self.sample_rate / int(np.prod(self.upsample_rates[2:]))
            self.upp = int(np.prod(self.upsample_rates[:2]))
        else:
            self.source_sr = self.sample_rate
            self.upp = self.hop_size
            self.m_source = _SourceModule(self.sample_rate)
            self.noise_convs = nn.ModuleList()

        channels = initial
        for index, (rate, kernel) in enumerate(
            zip(self.upsample_rates, config["upsample_kernel_sizes"])
        ):
            out_channels = channels // 2
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        channels,
                        out_channels,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )
            if not self.mini_nsf:
                remaining = int(np.prod(self.upsample_rates[index + 1 :]))
                if remaining > 1:
                    self.noise_convs.append(
                        nn.Conv1d(
                            1,
                            out_channels,
                            remaining * 2,
                            stride=remaining,
                            padding=remaining // 2,
                        )
                    )
                else:
                    self.noise_convs.append(nn.Conv1d(1, out_channels, 1))
            elif index == 1:
                self.source_conv = nn.Conv1d(1, out_channels, 1)
            for block_kernel, dilations in zip(
                config["resblock_kernel_sizes"],
                config["resblock_dilation_sizes"],
            ):
                self.resblocks.append(
                    _ResBlock(out_channels, int(block_kernel), list(dilations))
                )
            channels = out_channels

        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, padding=3))
        self.apply(_init_weights)

    def _fast_sine(self, f0: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(1, self.upp + 1, device=f0.device, dtype=f0.dtype)
        base = f0.unsqueeze(-1) / self.source_sr
        delta = F.pad(base[:, 1:] - base[:, :-1], (0, 0, 0, 1))
        phase = base * steps + 0.5 * delta * steps * (steps - 1) / self.upp
        phase_end = torch.remainder(phase[..., -1:].float() + 0.5, 1.0) - 0.5
        accumulated = phase_end.cumsum(dim=1).remainder(1.0).to(f0)
        phase = phase + F.pad(accumulated[:, :-1], (0, 0, 1, 0))
        return torch.sin(2 * np.pi * phase.reshape(f0.shape[0], 1, -1))

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3 or mel.shape[1] != self.num_mels:
            raise ValueError(
                f"Expected mel [B, {self.num_mels}, T], got {tuple(mel.shape)}"
            )
        if f0.ndim == 3:
            f0 = f0.squeeze(1)
        if f0.shape[-1] != mel.shape[-1]:
            f0 = F.interpolate(
                f0.unsqueeze(1), size=mel.shape[-1], mode="linear", align_corners=False
            ).squeeze(1)

        if self.mini_nsf:
            source = self._fast_sine(f0)
        else:
            source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(mel)
        if self.noise_sigma > 0:
            x = x + self.noise_sigma * torch.randn_like(x)
        for index, upsample in enumerate(self.ups):
            x = upsample(F.leaky_relu(x, LRELU_SLOPE))
            source_at_scale = None
            if not self.mini_nsf:
                source_at_scale = self.noise_convs[index](source)
            elif index == 1:
                source_at_scale = self.source_conv(source)
            if source_at_scale is not None:
                length = min(x.shape[-1], source_at_scale.shape[-1])
                x = x[..., :length] + source_at_scale[..., :length]
            merged = 0.0
            offset = index * self.num_kernels
            for block in self.resblocks[offset : offset + self.num_kernels]:
                merged = merged + block(x)
            x = merged / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x, LRELU_SLOPE)))

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()

    @classmethod
    def from_export(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "PCNSFHiFiGAN":
        checkpoint_path = Path(checkpoint_path)
        config_path = (
            Path(config_path)
            if config_path is not None
            else checkpoint_path.with_name("config.json")
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"pc-NSF-HiFiGAN checkpoint not found: {checkpoint_path}"
            )
        if not config_path.is_file():
            raise FileNotFoundError(f"pc-NSF config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        model = cls(config)
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        state = payload.get("generator", payload.get("state_dict", payload))
        state = {
            key.removeprefix("generator."): value
            for key, value in state.items()
            if not key.startswith("discriminator.")
        }
        model.load_state_dict(state, strict=True)
        model.eval()
        model.requires_grad_(False)
        return model
