"""Stationary compositional residual Hybrid-FSQ acoustic model."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def orthonormal_dct(rows: int, columns: int) -> torch.Tensor:
    n = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(columns, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos(math.pi / rows * (n + 0.5) * k)
    basis[:, 0] *= math.sqrt(1.0 / rows)
    if columns > 1:
        basis[:, 1:] *= math.sqrt(2.0 / rows)
    return basis


def temporal_lowpass(value: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    radius = kernel_size // 2
    if value.size(-1) <= radius:
        return value
    return F.avg_pool1d(
        F.pad(value, (radius, radius), mode="reflect"), kernel_size, stride=1
    )


def sequence_mask(lengths: torch.Tensor, maximum: int) -> torch.Tensor:
    return (
        torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]
    ).unsqueeze(1).to(torch.float32)


class ResidualTCN(nn.Module):
    def __init__(self, channels: int, layers: int, dropout: float, checkpointing: bool):
        super().__init__()
        self.checkpointing = checkpointing
        self.blocks = nn.ModuleList()
        for index in range(layers):
            dilation = 2 ** (index % 4)
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.GroupNorm(1, channels),
                        "conv": nn.Conv1d(
                            channels,
                            channels * 2,
                            5,
                            padding=2 * dilation,
                            dilation=dilation,
                        ),
                        "out": nn.Conv1d(channels, channels, 1),
                    }
                )
            )
        self.dropout = dropout

    def _block(self, x, mask, block):
        value, gate = block["conv"](F.silu(block["norm"](x))).chunk(2, 1)
        update = block["out"](value * torch.sigmoid(gate))
        update = F.dropout(update, self.dropout, self.training)
        return (x + update) * mask

    def forward(self, x, mask):
        for block in self.blocks:
            if self.checkpointing and self.training:
                x = checkpoint(self._block, x, mask, block, use_reentrant=False)
            else:
                x = self._block(x, mask, block)
        return x


class SDPA1D(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, mask):
        sequence = self.norm(x.transpose(1, 2))
        batch, time, channels = sequence.shape
        q, k, v = self.qkv(sequence).chunk(3, -1)
        width = channels // self.heads
        q = q.view(batch, time, self.heads, width).transpose(1, 2)
        k = k.view(batch, time, self.heads, width).transpose(1, 2)
        v = v.view(batch, time, self.heads, width).transpose(1, 2)
        valid = mask[:, :, None, :].bool()
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=valid)
        attended = attended.transpose(1, 2).reshape(batch, time, channels)
        return (x + self.out(attended).transpose(1, 2)) * mask


class FSQ(nn.Module):
    def __init__(self, levels: tuple[int, ...]):
        super().__init__()
        self.levels = tuple(levels)
        vectors = torch.cartesian_prod(
            *[torch.linspace(-1.0, 1.0, level) for level in levels]
        )
        self.register_buffer("vectors", vectors, persistent=True)

    def forward(self, value):
        value = torch.tanh(value)
        indices = []
        quantized = []
        joint = torch.zeros_like(value[:, 0], dtype=torch.long)
        for dimension, levels in enumerate(self.levels):
            scaled = (value[:, dimension] + 1.0) * (levels - 1) / 2.0
            index = scaled.round().clamp(0, levels - 1).long()
            q = index.float() * 2.0 / (levels - 1) - 1.0
            quantized.append(value[:, dimension] + (q - value[:, dimension]).detach())
            indices.append(index)
            multiplier = math.prod(self.levels[dimension + 1 :])
            joint += index * multiplier
        return torch.stack(quantized, 1), joint


class PreQuantProjection(nn.Module):
    """Weight-normalized projection plus bounded per-FSQ-dimension affine."""

    def __init__(self, input_channels: int, dimensions: int):
        super().__init__()
        self.projection = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(input_channels, dimensions, 1)
        )
        self.scale_parameter = nn.Parameter(torch.zeros(1, dimensions, 1))
        self.bias_parameter = nn.Parameter(torch.zeros(1, dimensions, 1))

    def forward(self, value):
        scale = 0.5 + 1.5 * torch.sigmoid(self.scale_parameter)
        bias = 0.5 * torch.tanh(self.bias_parameter)
        return self.projection(value) * scale + bias


class ResidualPathDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, checkpointing):
        super().__init__()
        self.pre = nn.Conv1d(input_channels, 128, 1)
        self.tcn = ResidualTCN(128, 3, 0.05, checkpointing)
        self.out = nn.Conv1d(128, output_channels, 3, padding=1)

    def forward(self, value, mask):
        return self.out(self.tcn(self.pre(value) * mask, mask)) * mask


class HybridFSQSynthesizer(nn.Module):
    """One-pass mel converter; training-only posteriors are exportable separately."""

    def __init__(
        self,
        spec_channels=128,
        mel_channels=128,
        hidden_channels=192,
        text_enc_hidden_dim=768,
        spk_embed_dim=109,
        gin_channels=128,
        tcn_blocks=6,
        attention_heads=4,
        checkpointing=False,
        local_prior_components=2,
        global_latent_channels=8,
        **_,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        self.mel_channels = mel_channels
        self.hidden_channels = hidden_channels
        self.components = local_prior_components
        self.global_latent_channels = global_latent_channels
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        self.pitch = nn.Embedding(256, hidden_channels)
        self.phone = nn.Linear(text_enc_hidden_dim, hidden_channels)
        self.f0 = nn.Conv1d(2, hidden_channels, 1)
        self.speaker = nn.Conv1d(gin_channels, hidden_channels, 1)
        self.backbone = ResidualTCN(hidden_channels, tcn_blocks, 0.1, checkpointing)
        self.attention = SDPA1D(hidden_channels, attention_heads)

        self.base_down = nn.Conv1d(hidden_channels, 128, 4, stride=2, padding=1)
        self.base_tcn = ResidualTCN(128, 3, 0.05, checkpointing)
        self.base_out = nn.Conv1d(128, mel_channels, 1)

        latent_input = text_enc_hidden_dim + 3 + gin_channels
        self.latent_stem = nn.Sequential(
            nn.Conv1d(latent_input, 128, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.SiLU(),
        )
        self.global_prior = nn.Linear(128, global_latent_channels * 2)
        self.global_posterior = nn.Sequential(
            nn.Linear(128 + 8, 64),
            nn.SiLU(),
            nn.Linear(64, global_latent_channels * 2),
        )
        self.global_decoder = nn.Sequential(
            nn.Linear(global_latent_channels, 32), nn.SiLU(), nn.Linear(32, 8)
        )

        self.slow_posterior = nn.Sequential(
            nn.Conv1d(mel_channels + 128, 96, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(96, 96, 5, stride=2, padding=2),
            nn.SiLU(),
        )
        self.slow_prequant = PreQuantProjection(96, 3)
        self.fast_posterior = nn.Sequential(
            nn.Conv1d(mel_channels + 128, 96, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(96, 96, 3, padding=1),
            nn.SiLU(),
        )
        self.fast_prequant = PreQuantProjection(96, 3)
        self.slow_fsq = FSQ((8, 8, 8))
        self.fast_fsq = FSQ((5, 5, 5))
        self.pi_head = nn.Linear(128, local_prior_components)
        self.slow_prior = nn.Conv1d(
            128, local_prior_components * 512, 1
        )
        self.fast_prior = nn.Conv1d(
            128, local_prior_components * 125, 1
        )
        self.slow_decoder = ResidualPathDecoder(
            hidden_channels + 3, mel_channels, checkpointing
        )
        self.fast_decoder = ResidualPathDecoder(
            hidden_channels + 3, mel_channels, checkpointing
        )
        basis = orthonormal_dct(mel_channels, 8)
        self.register_buffer("dct_basis", basis)
        self.register_buffer("mel_mean", torch.zeros(mel_channels))
        self.register_buffer("mel_std", torch.ones(mel_channels))
        self.register_buffer("global_cap", torch.ones(mel_channels))
        self.register_buffer("slow_cap", torch.ones(mel_channels))
        self.register_buffer("fast_cap", torch.ones(mel_channels))
        self.pc_vocoder = None

    def set_statistics(self, stats):
        self.mel_mean.copy_(stats["mel_mean"])
        self.mel_std.copy_(stats["mel_std"])
        self.dct_basis.copy_(stats["dct_basis"])
        self.global_cap.copy_(stats["caps"]["global"])
        self.slow_cap.copy_(stats["caps"]["slow"])
        self.fast_cap.copy_(stats["caps"]["fast"])

    def set_vocoder(self, vocoder):
        vocoder.eval().requires_grad_(False)
        self.pc_vocoder = vocoder

    def _conditions(self, phone, pitch, pitchf, sid, lengths, source_onset=None):
        time = phone.size(1)
        mask = sequence_mask(lengths, time).to(phone.dtype)
        f0 = torch.stack(
            (torch.log1p(pitchf.clamp_min(0)) / 7.0, (pitchf > 0).float()), 1
        )
        speaker = self.emb_g(sid).unsqueeze(-1)
        acoustic = (
            self.phone(phone).transpose(1, 2)
            + self.pitch(pitch.clamp(0, 255)).transpose(1, 2)
            + self.f0(f0)
            + self.speaker(speaker)
        ) * mask
        acoustic = self.backbone(acoustic, mask)
        if self.checkpointing and self.training:
            acoustic = checkpoint(
                self.attention, acoustic, mask, use_reentrant=False
            )
        else:
            acoustic = self.attention(acoustic, mask)
        if source_onset is None:
            source_onset = torch.diff(
                phone.detach(), dim=1, prepend=phone.detach()[:, :1]
            ).square().mean(-1, keepdim=True).transpose(1, 2)
        elif source_onset.ndim == 2:
            source_onset = source_onset.unsqueeze(1)
        source_onset = F.interpolate(
            source_onset.float(), size=time, mode="linear", align_corners=False
        ).to(phone.dtype)
        raw = torch.cat(
            (
                phone.detach().transpose(1, 2),
                f0.detach(),
                source_onset.detach(),
                speaker.detach().expand(-1, -1, time),
            ),
            1,
        )
        latent = self.latent_stem(raw) * mask
        return acoustic, latent, speaker, mask

    def _base(self, condition, mask):
        hidden = self.base_down(condition)
        down_mask = F.interpolate(mask, size=hidden.size(-1), mode="nearest")
        hidden = self.base_tcn(hidden, down_mask)
        hidden = F.interpolate(hidden, size=condition.size(-1), mode="linear", align_corners=False)
        return self.base_out(hidden) * mask

    def _prior_logits(self, latent, length):
        slow = F.interpolate(latent, size=math.ceil(length / 4), mode="linear", align_corners=False)
        fast = F.interpolate(latent, size=math.ceil(length / 2), mode="linear", align_corners=False)
        b = latent.size(0)
        slow_logits = self.slow_prior(slow).view(b, self.components, 512, -1)
        fast_logits = self.fast_prior(fast).view(b, self.components, 125, -1)
        pooled = latent.mean(-1)
        return self.pi_head(pooled), slow_logits, fast_logits

    @staticmethod
    def _gaussian(parameters):
        mean, log_scale = parameters.chunk(2, -1)
        return mean, log_scale.clamp(-7.0, 3.0)

    def forward(
        self,
        phone,
        phone_lengths,
        pitch,
        pitchf,
        spec,
        spec_lengths,
        ds,
        slow_target,
        fast_target,
        global_coeff,
    ):
        mel_energy = spec.float().mean(1)
        mel_onset = F.relu(
            torch.diff(mel_energy, dim=-1, prepend=mel_energy[:, :1])
        )
        mel_onset = mel_onset / mel_onset.amax(-1, keepdim=True).clamp_min(1e-5)
        condition, latent, _, mask = self._conditions(
            phone, pitch, pitchf, ds, phone_lengths, mel_onset
        )
        base = self._base(condition, mask)
        pooled = (latent * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        mu_p, logs_p = self._gaussian(self.global_prior(pooled))
        mu_q, logs_q = self._gaussian(
            self.global_posterior(torch.cat((pooled, global_coeff), -1))
        )
        z = mu_q + torch.randn_like(mu_q) * torch.exp(logs_q)
        global_coeff_hat = self.global_decoder(z)
        global_delta = (global_coeff_hat @ self.dct_basis.T).unsqueeze(-1)
        global_delta = (
            self.global_cap[None, :, None] * torch.tanh(global_delta)
        ).expand(-1, -1, spec.size(-1)) * mask

        slow_condition = F.interpolate(latent, size=slow_target.size(-1), mode="linear", align_corners=False)
        fast_condition = F.interpolate(latent, size=fast_target.size(-1), mode="linear", align_corners=False)
        slow_pre = self.slow_prequant(
            self.slow_posterior(torch.cat((slow_target, slow_condition), 1))
        )
        fast_pre = self.fast_prequant(
            self.fast_posterior(torch.cat((fast_target, fast_condition), 1))
        )
        slow_q, slow_ids = self.slow_fsq(slow_pre)
        fast_q, fast_ids = self.fast_fsq(fast_pre)
        slow_up = F.interpolate(slow_q, size=spec.size(-1), mode="nearest")
        fast_up = F.interpolate(fast_q, size=spec.size(-1), mode="nearest")
        slow_raw = self.slow_decoder(torch.cat((condition, slow_up), 1), mask)
        fast_raw = self.fast_decoder(torch.cat((condition, fast_up), 1), mask)
        slow_delta = temporal_lowpass(slow_raw, 5)
        slow_delta = slow_delta - (
            (slow_delta * mask).sum(-1, keepdim=True)
            / mask.sum(-1, keepdim=True).clamp_min(1)
        )
        slow_delta = self.slow_cap[None, :, None] * torch.tanh(slow_delta)
        fast_delta = fast_raw - temporal_lowpass(fast_raw, 5)
        fast_delta = self.fast_cap[None, :, None] * torch.tanh(fast_delta)
        pi, slow_logits, fast_logits = self._prior_logits(latent, spec.size(-1))
        final = (base + global_delta + slow_delta + fast_delta) * mask
        return {
            "mel": final,
            "base": base,
            "global": global_delta,
            "slow": slow_delta,
            "fast": fast_delta,
            "mask": mask,
            "global_distribution": (mu_q, logs_q, mu_p, logs_p),
            "prior": (pi, slow_logits, fast_logits),
            "codes": (slow_ids, fast_ids, slow_q, fast_q),
        }

    @staticmethod
    def mixture_prior_loss(pi, slow_logits, fast_logits, slow_ids, fast_ids):
        slow_logp = F.log_softmax(slow_logits.float(), 2)
        fast_logp = F.log_softmax(fast_logits.float(), 2)
        slow_target = slow_ids[:, None, None].expand(-1, pi.size(1), 1, -1)
        fast_target = fast_ids[:, None, None].expand(-1, pi.size(1), 1, -1)
        slow_score = slow_logp.gather(2, slow_target).squeeze(2).sum(-1)
        fast_score = fast_logp.gather(2, fast_target).squeeze(2).sum(-1)
        score = F.log_softmax(pi.float(), -1) + slow_score + fast_score
        token_count = max(1, slow_ids.size(-1) + fast_ids.size(-1))
        nll = -torch.logsumexp(score, -1).mean() / token_count
        responsibilities = torch.softmax(score, -1).detach()

        # Geometry is returned separately by ``prior_geometry`` to keep this
        # exact mixture NLL free of reductions before logsumexp.
        return nll, responsibilities

    def prior_geometry(self, prior, codes, responsibilities):
        _, slow_logits, fast_logits = prior
        _, _, slow_q, fast_q = codes
        slow_expected = torch.einsum(
            "bkct,cd->bkdt", torch.softmax(slow_logits.float(), 2), self.slow_fsq.vectors
        )
        fast_expected = torch.einsum(
            "bkct,cd->bkdt", torch.softmax(fast_logits.float(), 2), self.fast_fsq.vectors
        )
        slow_target = slow_q.detach()[:, None]
        fast_target = fast_q.detach()[:, None]
        slow_error = F.smooth_l1_loss(
            slow_expected, slow_target.expand_as(slow_expected), reduction="none"
        ).mean((2, 3))
        fast_error = F.smooth_l1_loss(
            fast_expected, fast_target.expand_as(fast_expected), reduction="none"
        ).mean((2, 3))
        return ((slow_error + fast_error) * responsibilities).sum(-1).mean()

    @torch.jit.export
    def infer(
        self,
        phone,
        phone_lengths,
        pitch,
        nsff0,
        sid,
        seed: int = 0,
        noise_scale: float = 0.5,
        temperature: float = 0.7,
        source_onset: Optional[torch.Tensor] = None,
    ):
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        condition, latent, _, mask = self._conditions(
            phone, pitch, nsff0, sid, phone_lengths, source_onset
        )
        base = self._base(condition, mask)
        pooled = (latent * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        mu, logs = self._gaussian(self.global_prior(pooled))
        z = mu + torch.randn_like(mu) * torch.exp(logs) * noise_scale
        global_delta = (self.global_decoder(z) @ self.dct_basis.T).unsqueeze(-1)
        global_delta = self.global_cap[None, :, None] * torch.tanh(global_delta)
        pi, slow_logits, fast_logits = self._prior_logits(latent, phone.size(1))
        component = (
            pi.argmax(-1)
            if temperature <= 0
            else torch.distributions.Categorical(logits=pi / temperature).sample()
        )
        batch = torch.arange(phone.size(0), device=phone.device)
        slow_selected = slow_logits[batch, component]
        fast_selected = fast_logits[batch, component]

        def sample(logits, top_k):
            if temperature <= 0:
                return logits.argmax(1)
            if 0 < top_k < logits.size(1):
                values, indices = torch.topk(logits, top_k, dim=1)
                selected = torch.distributions.Categorical(
                    logits=values.permute(0, 2, 1) / temperature
                ).sample()
                return indices.permute(0, 2, 1).gather(
                    -1, selected.unsqueeze(-1)
                ).squeeze(-1)
            return torch.distributions.Categorical(
                logits=logits.permute(0, 2, 1) / temperature
            ).sample()

        slow_ids = sample(slow_selected, 4)
        fast_ids = sample(fast_selected, 3)
        slow_q = self.slow_fsq.vectors[slow_ids].permute(0, 2, 1)
        fast_q = self.fast_fsq.vectors[fast_ids].permute(0, 2, 1)
        slow_q = F.interpolate(slow_q, size=phone.size(1), mode="nearest")
        fast_q = F.interpolate(fast_q, size=phone.size(1), mode="nearest")
        slow_raw = self.slow_decoder(torch.cat((condition, slow_q), 1), mask)
        fast_raw = self.fast_decoder(torch.cat((condition, fast_q), 1), mask)
        slow = temporal_lowpass(slow_raw, 5)
        slow = slow - (slow * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
        slow = self.slow_cap[None, :, None] * torch.tanh(slow)
        fast = fast_raw - temporal_lowpass(fast_raw, 5)
        fast = self.fast_cap[None, :, None] * torch.tanh(fast)
        normalized = (base + global_delta + slow + fast) * mask
        mel = normalized * self.mel_std[None, :, None] + self.mel_mean[None, :, None]
        output = self.pc_vocoder(mel, nsff0) if self.pc_vocoder is not None else mel
        return output, mask, (z, component, slow_ids, fast_ids)
