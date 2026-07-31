"""Raw waveform voice conversion built around the repository's pc-NSF decoder.

The model intentionally has no acoustic model or external content encoder in
its inference path.  A small strided waveform encoder supplies a constrained
content representation, a separate prosody stem predicts F0/voicing/energy,
and the existing pc-NSF-HiFiGAN decoder renders the target speaker.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.pc_nsf_hifigan import PCNSFHiFiGAN


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


class _Residual1d(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Conv1d(
            channels,
            channels,
            5,
            padding=2 * dilation,
            dilation=dilation,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(F.silu(self.norm(value)))
        return value + 0.2 * hidden


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        kernel = max(4, stride * 2)
        self.prefilter = nn.Conv1d(
            in_channels,
            in_channels,
            5,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        with torch.no_grad():
            self.prefilter.weight.fill_(0.0)
            self.prefilter.weight[:, 0, :] = torch.tensor(
                [1.0, 4.0, 6.0, 4.0, 1.0], dtype=self.prefilter.weight.dtype
            ) / 16.0
        self.prefilter.weight.requires_grad_(False)
        self.down = nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=(kernel - stride) // 2,
        )
        self.residual = nn.Sequential(
            _Residual1d(out_channels, 1),
            _Residual1d(out_channels, 3),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.residual(F.silu(self.down(self.prefilter(value))))


class _RawStem(nn.Module):
    def __init__(self, channels: int, final_stride: int):
        super().__init__()
        strides = [2, 2, 4, 4, 8]
        if final_stride == 1024:
            strides.append(2)
        widths = [32, 64, 128, 192, 256, 256]
        blocks = []
        input_channels = 1
        for index, stride in enumerate(strides):
            output_channels = widths[index]
            blocks.append(_DownBlock(input_channels, output_channels, stride))
            input_channels = output_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = input_channels
        self.post = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, 3, padding=1),
            nn.GroupNorm(1, input_channels),
            nn.SiLU(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.post(self.blocks(waveform))


class RawNSFWaveformGAN(nn.Module):
    """Waveform-to-waveform any-to-target conversion generator."""

    def __init__(
        self,
        decoder_config: dict[str, Any],
        spk_embed_dim: int = 109,
        speaker_dim: int = 192,
        content_channels: int = 96,
        sample_rate: int = 44100,
        hop_size: int = 512,
        speaker_adversarial_weight: float = 0.1,
        gradient_checkpointing: bool = True,
        **_: Any,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hop_size = int(hop_size)
        self.content_channels = int(content_channels)
        self.speaker_dim = int(speaker_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.emb_g = nn.Embedding(int(spk_embed_dim), self.speaker_dim)
        self.content_stem = _RawStem(self.content_channels, 1024)
        self.prosody_stem = _RawStem(1, 512)
        self.content_proj = nn.Conv1d(256, self.content_channels, 1)
        self.prosody_head = nn.Conv1d(256, 3, 1)
        self.speaker_proj = nn.Linear(self.speaker_dim, 32)
        self.prosody_proj = nn.Conv1d(3, 32, 1)
        self.conditioning = nn.Sequential(
            nn.Conv1d(self.content_channels + 32 + 32, 192, 3, padding=1),
            nn.GroupNorm(1, 192),
            nn.SiLU(),
            nn.Conv1d(192, int(decoder_config["num_mels"]), 1),
        )
        decoder_config = dict(decoder_config)
        # The raw architecture uses the complete NSF path.  Existing
        # mini-NSF checkpoints remain useful for all shared decoder weights;
        # the full-source modules are initialized and learned by the raw GAN.
        decoder_config["mini_nsf"] = False
        decoder_config["speaker_dim"] = self.speaker_dim
        self.decoder = PCNSFHiFiGAN(decoder_config)
        self.content_speaker_classifier = nn.Sequential(
            nn.Conv1d(self.content_channels, self.content_channels, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.content_channels, int(spk_embed_dim)),
        )
        self.speaker_adversarial_weight = float(speaker_adversarial_weight)

    def load_decoder_checkpoint(self, state: dict[str, torch.Tensor]) -> int:
        """Load compatible pc-NSF tensors while retaining new conditioning layers."""
        decoder_state = self.decoder.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in decoder_state and decoder_state[key].shape == value.shape
        }
        decoder_state.update(compatible)
        self.decoder.load_state_dict(decoder_state, strict=True)
        return len(compatible)

    def _pad_waveform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        length = waveform.size(-1)
        padded = (1024 - length % 1024) % 1024
        if padded:
            waveform = F.pad(waveform, (0, padded), mode="reflect")
        return waveform, length

    def _encode(
        self, waveform: torch.Tensor, sid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        content_raw = self.content_stem(waveform)
        content = self.content_proj(content_raw)
        prosody_raw = self.prosody_stem(waveform)
        prosody = self.prosody_head(prosody_raw)
        log_f0 = prosody[:, :1].clamp(-3.0, 7.4)
        uv_logits = prosody[:, 1:2]
        energy = prosody[:, 2:3]
        f0 = torch.exp(log_f0[:, 0]).clamp(20.0, 1600.0)
        f0 = f0 * torch.sigmoid(uv_logits[:, 0])
        content = F.interpolate(
            content,
            size=prosody.size(-1),
            mode="linear",
            align_corners=False,
        )
        speaker = self.emb_g(sid)
        speaker_features = self.speaker_proj(speaker).unsqueeze(-1)
        speaker_features = speaker_features.expand(-1, -1, prosody.size(-1))
        condition_input = torch.cat(
            (content, self.prosody_proj(prosody), speaker_features), dim=1
        )
        condition = self.conditioning(condition_input)
        content_logits = self.content_speaker_classifier(
            _GradientReversal.apply(content, self.speaker_adversarial_weight)
        )
        return {
            "content": content,
            "condition": condition,
            "speaker": speaker,
            "log_f0": log_f0[:, 0],
            "uv_logits": uv_logits[:, 0],
            "energy": energy[:, 0],
            "f0": f0,
            "content_speaker_logits": content_logits,
        }

    def forward(
        self,
        waveform: torch.Tensor,
        sid: torch.Tensor,
        f0_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        waveform, original_length = self._pad_waveform(waveform)
        encoded = self._encode(waveform, sid)
        f0 = encoded["f0"] if f0_override is None else f0_override
        if f0.shape[-1] != encoded["condition"].shape[-1]:
            f0 = F.interpolate(
                f0.unsqueeze(1),
                size=encoded["condition"].size(-1),
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        decoder_inputs = (encoded["condition"], f0.detach(), encoded["speaker"])
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            rendered = checkpoint(self.decoder, *decoder_inputs, use_reentrant=False)
        else:
            rendered = self.decoder(*decoder_inputs)
        rendered = rendered[..., : waveform.size(-1)]
        rendered = rendered[..., :original_length]
        encoded["waveform"] = rendered
        encoded["source_length"] = torch.tensor(
            original_length, device=waveform.device
        )
        return encoded

    @torch.jit.export
    def infer(
        self,
        waveform: torch.Tensor,
        sid: torch.Tensor,
        f0_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(waveform, sid, f0_override)["waveform"]
