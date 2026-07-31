"""Mel-VITS acoustic model for voice conversion.

Unlike the legacy fork, this module never synthesizes waveforms.  It predicts
the log-mel representation expected by the external pc-NSF-HiFiGAN.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import rand_slice_segments, slice_segments
from rvc.lib.algorithm.normalizing_flow import ResidualCouplingBlock
from rvc.lib.algorithm.posterior_encoder import PosteriorEncoder
from rvc.lib.algorithm.text_encoder import TextEncoder


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float):
        ctx.scale = scale
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


class SpeakerClassifier(nn.Module):
    def __init__(self, channels: int, speakers: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
            nn.SiLU(),
        )
        self.output = nn.Linear(channels, speakers)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(value) * mask
        pooled = hidden.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        return self.output(pooled)


class _MelResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Conv1d(
            channels, channels * 2, kernel_size, padding=padding, dilation=dilation
        )
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(F.silu(self.norm(x)))
        value, gate = hidden.chunk(2, dim=1)
        return (x + self.proj(value * torch.sigmoid(gate))) * mask


class MelDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        mel_channels: int,
        gin_channels: int,
        layers: int = 8,
        kernel_size: int = 5,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        self.pre = nn.Conv1d(latent_channels, hidden_channels, 1)
        self.condition = nn.Conv1d(gin_channels, hidden_channels, 1)
        self.f0 = nn.Sequential(
            nn.Conv1d(1, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
        )
        self.blocks = nn.ModuleList(
            [
                _MelResidualBlock(
                    hidden_channels, kernel_size, dilation=2 ** (index % 4)
                )
                for index in range(layers)
            ]
        )
        self.post = nn.Sequential(
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, mel_channels, 1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        f0: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if f0.ndim == 2:
            f0 = f0.unsqueeze(1)
        if f0.shape[-1] != latent.shape[-1]:
            f0 = F.interpolate(f0, size=latent.shape[-1], mode="linear", align_corners=False)
        log_f0 = torch.log1p(f0.clamp_min(0.0)) / 7.0
        hidden = (self.pre(latent) + self.condition(condition) + self.f0(log_f0)) * mask
        for block in self.blocks:
            if self.checkpointing and self.training:
                hidden = checkpoint(block, hidden, mask, use_reentrant=False)
            else:
                hidden = block(hidden, mask)
        return self.post(hidden) * mask


class Synthesizer(nn.Module):
    """Conditional VAE/flow acoustic model whose output is log-mel."""

    def __init__(
        self,
        spec_channels: int = 128,
        segment_size: int = 32,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.0,
        spk_embed_dim: int = 109,
        gin_channels: int = 256,
        sr: int = 44100,
        use_f0: bool = True,
        text_enc_hidden_dim: int = 768,
        mel_channels: int = 128,
        mel_decoder_layers: int = 8,
        mel_decoder_kernel_size: int = 5,
        checkpointing: bool = False,
        use_2_sample_kl: bool = False,
        training_auxiliaries: bool = True,
        use_sdpa: bool = True,
        **_: object,
    ):
        super().__init__()
        if not use_f0:
            raise ValueError("Mel-VITS requires F0 conditioning")
        self.segment_size = segment_size
        self.sr = sr
        self.mel_channels = mel_channels
        self.use_2_sample_kl = use_2_sample_kl
        self.enc_p = TextEncoder(
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            embedding_dim=text_enc_hidden_dim,
            f0=True,
            checkpointing=checkpointing,
            use_sdpa=use_sdpa,
        )
        self.enc_q = PosteriorEncoder(
            in_channels=mel_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            gin_channels=gin_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            checkpointing=checkpointing,
        )
        self.flow = ResidualCouplingBlock(
            channels=inter_channels,
            hidden_channels=hidden_channels,
            n_flows=4,
            n_layers=3,
            kernel_size=5,
            dilation_rate=1,
            gin_channels=gin_channels,
            checkpointing=checkpointing,
        )
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        self.dec = MelDecoder(
            latent_channels=inter_channels,
            hidden_channels=hidden_channels,
            mel_channels=mel_channels,
            gin_channels=gin_channels,
            layers=mel_decoder_layers,
            kernel_size=mel_decoder_kernel_size,
            checkpointing=checkpointing,
        )
        self.content_speaker_classifier = (
            SpeakerClassifier(inter_channels, spk_embed_dim)
            if training_auxiliaries
            else None
        )
        self.mel_speaker_classifier = (
            SpeakerClassifier(mel_channels, spk_embed_dim)
            if training_auxiliaries
            else None
        )
        self.pc_vocoder = None

    def set_vocoder(self, vocoder: nn.Module) -> None:
        """Attach the shared frozen waveform renderer for deployment."""
        vocoder.eval()
        vocoder.requires_grad_(False)
        self.pc_vocoder = vocoder

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        ds: torch.Tensor,
    ):
        condition = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(
            phone=phone, pitch=pitch, pitchf=pitchf, lengths=phone_lengths
        )
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=condition)
        z_p = self.flow(z, spec_mask, g=condition)
        z_p2 = None
        if self.use_2_sample_kl:
            z2 = (
                m_q.float()
                + torch.randn_like(m_q, dtype=torch.float32)
                * torch.exp(logs_q.float())
            ).to(m_q.dtype) * spec_mask
            z_p2 = self.flow(z2, spec_mask, g=condition)
        segment_size = min(self.segment_size, z.size(-1))
        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, segment_size)
        f0_slice = slice_segments(pitchf, ids_slice, segment_size, 2)
        mask_slice = slice_segments(spec_mask, ids_slice, segment_size, 3)
        mel = self.dec(z_slice, f0_slice, condition, mask_slice)
        speaker_logits = self.speaker_logits(
            m_p, x_mask, mel, mask_slice, adversarial_scale=1.0
        )
        return (
            mel,
            ids_slice,
            x_mask,
            spec_mask,
            (z, z_p, z_p2, m_p, logs_p, m_q, logs_q),
            speaker_logits,
        )

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        seed: int = 0,
        noise_scale: float = 0.35,
    ):
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        condition = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(
            phone=phone, pitch=pitch, pitchf=nsff0, lengths=phone_lengths
        )
        z_p = (
            m_p.float()
            + torch.exp(logs_p.float())
            * torch.randn_like(m_p, dtype=torch.float32)
            * noise_scale
        ).to(m_p.dtype) * x_mask
        z = self.flow(z_p, x_mask, g=condition, reverse=True)
        mel = self.dec(z, nsff0, condition, x_mask)
        output = (
            self.pc_vocoder(mel, nsff0)
            if self.pc_vocoder is not None
            else mel
        )
        return output, x_mask, (z, z_p, m_p, logs_p)

    def conversion_cycle(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        target_sid: torch.Tensor,
    ):
        """Generate a non-parallel SID swap and map it back to content space."""
        target_condition = self.emb_g(target_sid).unsqueeze(-1)
        m_p, _, mask = self.enc_p(
            phone=phone, pitch=pitch, pitchf=pitchf, lengths=phone_lengths
        )
        converted_z = self.flow(m_p * mask, mask, g=target_condition, reverse=True)
        converted_mel = self.dec(converted_z, pitchf, target_condition, mask)
        encoded_z, _, _, converted_mask = self.enc_q(
            converted_mel, phone_lengths, g=target_condition
        )
        recovered_content = self.flow(
            encoded_z, converted_mask, g=target_condition
        )
        return converted_mel, recovered_content, m_p.detach(), converted_mask

    def speaker_logits(
        self,
        content: torch.Tensor,
        content_mask: torch.Tensor,
        mel: torch.Tensor,
        mel_mask: torch.Tensor,
        adversarial_scale: float = 1.0,
    ):
        """Predict source identity adversarially and decoded target identity normally."""
        if (
            self.content_speaker_classifier is None
            or self.mel_speaker_classifier is None
        ):
            raise RuntimeError("Speaker auxiliaries are disabled in this export")
        reversed_content = _GradientReversal.apply(content, adversarial_scale)
        return (
            self.content_speaker_classifier(reversed_content, content_mask),
            self.mel_speaker_classifier(mel, mel_mask),
        )
