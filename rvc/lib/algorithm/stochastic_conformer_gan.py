"""Stochastic Residual Conformer-GAN mel converter.

The Conformer generator predicts a stable coarse mel and a small stochastic
residual.  Mel discriminators are training-only; inference keeps only the
generator, its prior and the external pc-NSF-HiFiGAN vocoder.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def sequence_mask(lengths: torch.Tensor, maximum: int) -> torch.Tensor:
    return (
        torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]
    ).unsqueeze(1).to(torch.float32)


class ConformerBlock(nn.Module):
    """Compact native-PyTorch Conformer block with masked SDPA."""

    def __init__(self, channels: int, heads: int, ffn_expansion: int, kernel: int, dropout: float, window_size: int = 256):
        super().__init__()
        if channels % heads:
            raise ValueError("Conformer channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.window_size = int(window_size)
        self.norm_ff1 = nn.LayerNorm(channels)
        self.ff1 = nn.Sequential(
            nn.Linear(channels, channels * ffn_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ffn_expansion, channels),
            nn.Dropout(dropout),
        )
        self.norm_attn = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.attn_out = nn.Linear(channels, channels)
        self.attn_dropout = dropout
        self.norm_conv = nn.LayerNorm(channels)
        self.conv_in = nn.Conv1d(channels, channels * 2, 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=kernel // 2,
            groups=channels,
        )
        self.conv_out = nn.Conv1d(channels, channels, 1)
        self.norm_ff2 = nn.LayerNorm(channels)
        self.ff2 = nn.Sequential(
            nn.Linear(channels, channels * ffn_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ffn_expansion, channels),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(channels)

    def _attention(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, c = value.shape
        q, k, v = self.qkv(value).chunk(3, -1)
        width = c // self.heads
        q = q.view(b, t, self.heads, width).transpose(1, 2)
        k = k.view(b, t, self.heads, width).transpose(1, 2)
        v = v.view(b, t, self.heads, width).transpose(1, 2)
        key_mask = mask[:, :, None, :].bool()
        if self.window_size > 0 and self.window_size < t:
            position = torch.arange(t, device=value.device)
            local = (position[:, None] - position[None, :]).abs() <= self.window_size // 2
            key_mask = key_mask & local[None, None]
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        return attended.transpose(1, 2).reshape(b, t, c)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        value = self.norm_ff1(x.transpose(1, 2))
        x = x + 0.5 * self.ff1(value).transpose(1, 2)

        value = self.norm_attn(x.transpose(1, 2))
        # Speaker AdaLN is deliberately confined to the generator backbone;
        # it keeps identity conditioning out of the stochastic posterior.
        speaker_scale, speaker_bias = speaker.chunk(2, -1)
        value = value * (1.0 + 0.1 * speaker_scale[:, None]) + 0.1 * speaker_bias[:, None]
        x = x + self.attn_out(self._attention(value, mask)).transpose(1, 2)

        value = self.norm_conv(x.transpose(1, 2)).transpose(1, 2)
        value, gate = self.conv_in(value).chunk(2, 1)
        value = value * torch.sigmoid(gate)
        value = F.silu(self.depthwise(value)) * mask
        x = x + self.conv_out(value) * mask

        value = self.norm_ff2(x.transpose(1, 2))
        x = x + 0.5 * self.ff2(value).transpose(1, 2)
        return self.final_norm(x.transpose(1, 2)).transpose(1, 2) * mask


class ResidualMelDecoder(nn.Module):
    def __init__(self, channels: int, latent_channels: int, mel_channels: int, blocks: int = 4, global_latent_channels: int = 8):
        super().__init__()
        self.input = nn.Conv1d(channels + latent_channels, channels, 1)
        self.global_to_film = nn.Linear(global_latent_channels, channels * 2)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    nn.Conv1d(channels, channels * 2, 5, padding=2),
                    nn.SiLU(),
                    nn.Conv1d(channels * 2, channels, 1),
                )
            )
        self.output = nn.Conv1d(channels, mel_channels, 3, padding=1)

    def forward(self, condition: torch.Tensor, local: torch.Tensor, mask: torch.Tensor, global_z: torch.Tensor) -> torch.Tensor:
        value = self.input(torch.cat((condition, local), 1)) * mask
        for block in self.blocks:
            value = (value + block(value)) * mask
        scale, bias = self.global_to_film(global_z).chunk(2, 1)
        value = value * (1.0 + 0.1 * torch.tanh(scale)[:, :, None])
        value = value + 0.1 * torch.tanh(bias)[:, :, None]
        return self.output(value) * mask


class MelPatchDiscriminator(nn.Module):
    """2-D random-area mel discriminator returning logits and features."""

    def __init__(self, channels: int = 48):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(1, channels, 5, 2, 2),
                nn.Conv2d(channels, channels * 2, 5, 2, 2),
                nn.Conv2d(channels * 2, channels * 4, 5, 2, 2),
                nn.Conv2d(channels * 4, channels * 4, 3, 1, 1),
            ]
        )
        self.out = nn.Conv2d(channels * 4, 1, 3, 1, 1)

    def forward(self, mel: torch.Tensor):
        value = mel.unsqueeze(1)
        features = []
        for block in self.blocks:
            value = F.leaky_relu(block(value), 0.2)
            features.append(value)
        return self.out(value), features


class VoicingMelDiscriminator(nn.Module):
    """Mel discriminator explicitly seeing F0 and voiced/unvoiced state."""

    def __init__(self, channels: int = 48):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(3, channels, 5, 2, 2),
                nn.Conv2d(channels, channels * 2, 5, 2, 2),
                nn.Conv2d(channels * 2, channels * 4, 5, 2, 2),
                nn.Conv2d(channels * 4, channels * 4, 3, 1, 1),
            ]
        )
        self.out = nn.Conv2d(channels * 4, 1, 3, 1, 1)

    def forward(self, mel: torch.Tensor, pitchf: torch.Tensor):
        f0 = (torch.log1p(pitchf.clamp_min(0)) / 7.0).unsqueeze(1)
        uv = (pitchf > 0).to(mel.dtype).unsqueeze(1)
        f0 = f0.unsqueeze(2).expand(-1, -1, mel.size(1), -1)
        uv = uv.unsqueeze(2).expand(-1, -1, mel.size(1), -1)
        value = torch.cat((mel.unsqueeze(1), f0.to(mel.dtype), uv), 1)
        features = []
        for block in self.blocks:
            value = F.leaky_relu(block(value), 0.2)
            features.append(value)
        return self.out(value), features


class StochasticResidualConformerGAN(nn.Module):
    def __init__(
        self,
        spec_channels=128,
        mel_channels=128,
        hidden_channels=192,
        text_enc_hidden_dim=768,
        spk_embed_dim=109,
        gin_channels=128,
        conformer_blocks=6,
        attention_heads=4,
        ffn_expansion=2,
        convolution_kernel=15,
        attention_window_size=256,
        global_latent_channels=8,
        local_latent_channels=12,
        local_downsample=4,
        residual_decoder_blocks=4,
        residual_cap=1.5,
        checkpointing=False,
        **_,
    ):
        super().__init__()
        self.checkpointing = bool(checkpointing)
        self.mel_channels = int(mel_channels)
        self.hidden_channels = int(hidden_channels)
        self.global_latent_channels = int(global_latent_channels)
        self.local_latent_channels = int(local_latent_channels)
        self.local_downsample = int(local_downsample)
        self.residual_cap = float(residual_cap)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        self.phone = nn.Linear(text_enc_hidden_dim, hidden_channels)
        self.pitch = nn.Embedding(256, hidden_channels)
        self.f0 = nn.Conv1d(2, hidden_channels, 1)
        self.speaker = nn.Conv1d(gin_channels, hidden_channels, 1)
        self.onset = nn.Conv1d(1, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    hidden_channels,
                    attention_heads,
                    ffn_expansion,
                    convolution_kernel,
                    0.05,
                    attention_window_size,
                )
                for _ in range(conformer_blocks)
            ]
        )
        self.speaker_adaln = nn.Linear(gin_channels, hidden_channels * 2)
        self.base_down = nn.Conv1d(hidden_channels, hidden_channels, 4, 2, 1)
        self.base_out = nn.Conv1d(hidden_channels, mel_channels, 1)
        self.prior_global = nn.Sequential(
            nn.Linear(hidden_channels, 96),
            nn.SiLU(),
            nn.Linear(96, global_latent_channels * 2),
        )
        self.prior_local = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, local_latent_channels * 2, 1),
        )
        self.posterior_global = nn.Sequential(
            nn.Linear(hidden_channels + mel_channels, 96),
            nn.SiLU(),
            nn.Linear(96, global_latent_channels * 2),
        )
        self.posterior_local = nn.Sequential(
            nn.Conv1d(hidden_channels + mel_channels, 128, 5, 2, 2),
            nn.SiLU(),
            nn.Conv1d(128, 128, 5, 2, 2),
            nn.SiLU(),
            nn.Conv1d(128, local_latent_channels * 2, 1),
        )
        self.residual_decoder = ResidualMelDecoder(
            hidden_channels,
            local_latent_channels,
            mel_channels,
            residual_decoder_blocks,
            global_latent_channels,
        )
        self.register_buffer("mel_mean", torch.zeros(mel_channels))
        self.register_buffer("mel_std", torch.ones(mel_channels))
        self.register_buffer("hybrid_quality_patch", torch.tensor(0, dtype=torch.int32))
        self.pc_vocoder = None
        self.random_area_discriminator = MelPatchDiscriminator()
        self.voicing_discriminator = VoicingMelDiscriminator()

    def set_statistics(self, stats):
        self.mel_mean.copy_(stats["mel_mean"])
        self.mel_std.copy_(stats["mel_std"])

    def set_vocoder(self, vocoder):
        vocoder.eval().requires_grad_(False)
        self.pc_vocoder = vocoder

    @staticmethod
    def _gaussian(value):
        mean, logs = value.chunk(2, -1)
        return mean, logs.clamp(-7.0, 3.0)

    def _conditions(self, phone, pitch, pitchf, sid, lengths, source_onset=None):
        time = phone.size(1)
        mask = sequence_mask(lengths, time).to(phone.dtype)
        f0 = torch.stack(
            (torch.log1p(pitchf.clamp_min(0)) / 7.0, (pitchf > 0).float()), 1
        )
        speaker_embedding = self.emb_g(sid)
        value = (
            self.phone(phone).transpose(1, 2)
            + self.pitch(pitch.clamp(0, 255)).transpose(1, 2)
            + self.f0(f0)
            + self.speaker(speaker_embedding.unsqueeze(-1))
        ) * mask
        if source_onset is None:
            source_onset = torch.diff(
                phone.detach().float().mean(-1),
                dim=1,
                prepend=phone.detach().float()[:, :1].mean(-1),
            ).abs().unsqueeze(1)
        elif source_onset.ndim == 2:
            source_onset = source_onset.unsqueeze(1)
        source_onset = F.interpolate(
            source_onset.float(), size=time, mode="linear", align_corners=False
        ).to(value.dtype)
        value = value + self.onset(source_onset)
        speaker_style = self.speaker_adaln(speaker_embedding)
        for block in self.blocks:
            value = block(value, mask, speaker_style)
        return value * mask, speaker_embedding, mask

    def _base(self, condition, mask):
        down = self.base_down(condition)
        down_mask = F.interpolate(mask, size=down.size(-1), mode="nearest")
        down = F.silu(down) * down_mask
        return self.base_out(
            F.interpolate(down, size=condition.size(-1), mode="linear", align_corners=False)
        ) * mask

    def forward(
        self,
        phone,
        phone_lengths,
        pitch,
        pitchf,
        spec,
        spec_lengths,
        ds,
        local_prior_mix: float = 0.0,
        global_prior: bool = False,
        source_onset: Optional[torch.Tensor] = None,
    ):
        if source_onset is None:
            energy = spec.float().mean(1)
            source_onset = F.relu(
                torch.diff(energy, dim=-1, prepend=energy[:, :1])
            )
            source_onset = source_onset / source_onset.amax(
                -1, keepdim=True
            ).clamp_min(1e-5)
        condition, speaker, mask = self._conditions(
            phone, pitch, pitchf, ds, phone_lengths, source_onset
        )
        base = self._base(condition, mask)
        pooled = (condition * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
        residual = spec - base.detach()
        residual_pool = (residual * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
        mu_p, logs_p = self._gaussian(self.prior_global(pooled))
        mu_q, logs_q = self._gaussian(
            self.posterior_global(torch.cat((pooled, residual_pool), -1))
        )
        if global_prior:
            global_z = mu_p
        else:
            global_z = mu_q + torch.randn_like(mu_q) * torch.exp(logs_q)

        prior_local_raw = self.prior_local(condition * mask)
        posterior_local_raw = self.posterior_local(torch.cat((condition, residual), 1))
        # The local heads are [B, 2C, T/4]; keep the channel-first layout.
        mu_lp, logs_lp = prior_local_raw.chunk(2, 1)
        mu_lq, logs_lq = posterior_local_raw.chunk(2, 1)
        logs_lp = logs_lp.clamp(-7.0, 3.0)
        logs_lq = logs_lq.clamp(-7.0, 3.0)
        local_q = mu_lq + torch.randn_like(mu_lq) * torch.exp(logs_lq)
        local_p = mu_lp
        probability = float(torch.as_tensor(local_prior_mix).detach().cpu())
        probability = max(0.0, min(1.0, probability))
        if probability > 0.0:
            choose_prior = (
                torch.rand(phone.size(0), 1, 1, device=phone.device) < probability
            )
            local_z = torch.where(choose_prior, local_p, local_q)
            local_prior_fraction = choose_prior.float().mean()
        else:
            local_z = local_q
            local_prior_fraction = local_q.new_zeros(())
        local_z = F.interpolate(local_z, size=condition.size(-1), mode="nearest")
        delta = self.residual_decoder(condition, local_z, mask, global_z)
        delta = self.residual_cap * torch.tanh(delta)
        mel = (base + delta) * mask
        return {
            "mel": mel,
            "base": base,
            "delta": delta,
            "mask": mask,
            "global_distribution": (mu_q, logs_q, mu_p, logs_p),
            "local_distribution": (mu_lq, logs_lq, mu_lp, logs_lp),
            "local_prior_fraction": local_prior_fraction,
        }

    @torch.jit.export
    def infer(
        self,
        phone,
        phone_lengths,
        pitch,
        nsff0,
        sid,
        seed: int = 0,
        # The validation path evaluates the prior means (without sampling).
        # Keep interactive inference on the same, stable path by default.
        # Sampling remains available through the explicit arguments used by
        # callers that want controlled stochastic variation.
        global_noise_scale: float = 0.0,
        local_noise_scale: float = 0.0,
        stochastic_strength: float = 1.0,
        source_onset: Optional[torch.Tensor] = None,
    ):
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        condition, _, mask = self._conditions(
            phone, pitch, nsff0, sid, phone_lengths, source_onset
        )
        base = self._base(condition, mask)
        pooled = (condition * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
        mu_p, logs_p = self._gaussian(self.prior_global(pooled))
        global_z = mu_p + torch.randn_like(mu_p) * torch.exp(logs_p) * global_noise_scale
        local_raw = self.prior_local(condition * mask)
        mu_lp, logs_lp = local_raw.chunk(2, 1)
        logs_lp = logs_lp.clamp(-7.0, 3.0)
        local_z = mu_lp + torch.randn_like(mu_lp) * torch.exp(logs_lp) * local_noise_scale
        local_z = F.interpolate(local_z, size=condition.size(-1), mode="nearest")
        delta = self.residual_decoder(condition, local_z, mask, global_z)
        # The generator operates in the normalized mel domain used by the
        # dataset statistics.  pc-NSF-HiFiGAN, however, was trained on the
        # original (de-normalized) mel values.  Passing the normalized tensor
        # directly to the vocoder drives its post-tanh output into saturation
        # and produces heavily distorted/clipped audio.
        normalized_mel = (
            base + stochastic_strength * self.residual_cap * torch.tanh(delta)
        ) * mask
        vocoder_mel = (
            normalized_mel * self.mel_std[None, :, None]
            + self.mel_mean[None, :, None]
        )
        output = (
            self.pc_vocoder(vocoder_mel, nsff0)
            if self.pc_vocoder is not None
            else vocoder_mel
        )
        return output, mask, (normalized_mel, base, delta, global_z, local_z)


def hinge_discriminator_loss(real_logits, fake_logits):
    return 0.5 * (
        F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    )


def generator_adversarial_loss(fake_logits):
    return -fake_logits.mean()
