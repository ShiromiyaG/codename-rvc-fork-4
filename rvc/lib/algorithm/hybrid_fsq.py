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
        self.levels = tuple(int(level) for level in levels)
        if len(set(self.levels)) != 1:
            raise ValueError("Each Hybrid-FSQ branch requires equal scalar levels")
        self.dimensions = len(self.levels)
        self.level_count = self.levels[0]
        values = torch.linspace(-1.0, 1.0, self.level_count)
        self.register_buffer(
            "level_values",
            values[None].expand(self.dimensions, -1).clone(),
            persistent=True,
        )

    def forward(self, value):
        bounded = torch.tanh(value)
        distance = (
            bounded.unsqueeze(2) - self.level_values[None, :, :, None]
        ).abs()
        indices = distance.argmin(2)
        quantized = self.values_from_ids(indices).to(bounded)
        return bounded + (quantized - bounded).detach(), indices, bounded

    def values_from_ids(self, indices):
        levels = self.level_values[None, :, :, None].expand(
            indices.size(0), -1, -1, indices.size(-1)
        )
        return levels.gather(2, indices.unsqueeze(2)).squeeze(2)

    def soft_targets(self, bounded, temperature):
        distance = (
            bounded.detach().unsqueeze(2)
            - self.level_values[None, :, :, None]
        ).square()
        return torch.softmax(-distance / max(1e-4, temperature), 2)

    def expected(self, logits):
        probability = torch.softmax(logits.float(), 2)
        return (
            probability
            * self.level_values[None, :, :, None].to(probability)
        ).sum(2).to(logits.dtype)

    def sample(self, logits, temperature, top_k, noise_scale):
        expected = self.expected(logits)
        if temperature <= 0 or noise_scale <= 0:
            return expected, logits.argmax(2)
        scaled = logits.float() / max(1e-4, temperature)
        if 0 < top_k < scaled.size(2):
            values, level_ids = torch.topk(scaled, top_k, dim=2)
            local = torch.distributions.Categorical(
                logits=values.permute(0, 1, 3, 2)
            ).sample()
            indices = level_ids.permute(0, 1, 3, 2).gather(
                -1, local.unsqueeze(-1)
            ).squeeze(-1)
        else:
            indices = torch.distributions.Categorical(
                logits=scaled.permute(0, 1, 3, 2)
            ).sample()
        sampled = self.values_from_ids(indices).to(expected)
        return expected + float(noise_scale) * (sampled - expected), indices


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
    def __init__(
        self,
        input_channels,
        output_channels,
        checkpointing,
        hidden_channels=160,
        layers=4,
    ):
        super().__init__()
        self.pre = nn.Conv1d(input_channels, hidden_channels, 1)
        self.tcn = ResidualTCN(
            hidden_channels, layers, 0.05, checkpointing
        )
        self.out = nn.Conv1d(hidden_channels, output_channels, 3, padding=1)

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
        global_latent_channels=8,
        slow_fsq_levels=(8, 8, 8, 8),
        fast_fsq_levels=(5, 5, 5, 5),
        prior_soft_target_temperature=0.08,
        hybrid_quality_patch=1,
        **_,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        self.mel_channels = mel_channels
        self.hidden_channels = hidden_channels
        self.global_latent_channels = global_latent_channels
        self.prior_soft_target_temperature = prior_soft_target_temperature
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

        # Local texture is the quality-limiting path at inference.  Keep it
        # fully independent from the global Gaussian, but give both the
        # posterior and decoder enough bandwidth to retain consonants and
        # high-band detail after FSQ quantization.
        local_channels = 128
        self.slow_posterior = nn.Sequential(
            nn.Conv1d(mel_channels + 128, local_channels, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(local_channels, local_channels, 5, stride=2, padding=2),
            nn.SiLU(),
        )
        self.slow_prequant = PreQuantProjection(
            local_channels, len(slow_fsq_levels)
        )
        self.fast_posterior = nn.Sequential(
            nn.Conv1d(mel_channels + 128, local_channels, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(local_channels, local_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.fast_prequant = PreQuantProjection(
            local_channels, len(fast_fsq_levels)
        )
        self.slow_fsq = FSQ(tuple(slow_fsq_levels))
        self.fast_fsq = FSQ(tuple(fast_fsq_levels))
        self.slow_prior_context = ResidualTCN(128, 4, 0.05, checkpointing)
        self.fast_prior_context = ResidualTCN(128, 4, 0.05, checkpointing)
        self.slow_prior = nn.Conv1d(
            128, self.slow_fsq.dimensions * self.slow_fsq.level_count, 1
        )
        self.fast_prior = nn.Conv1d(
            128, self.fast_fsq.dimensions * self.fast_fsq.level_count, 1
        )
        self.slow_decoder = ResidualPathDecoder(
            hidden_channels + self.slow_fsq.dimensions,
            mel_channels,
            checkpointing,
            hidden_channels=192,
            layers=5,
        )
        self.fast_decoder = ResidualPathDecoder(
            hidden_channels + self.fast_fsq.dimensions,
            mel_channels,
            checkpointing,
            hidden_channels=192,
            layers=5,
        )
        basis = orthonormal_dct(mel_channels, 8)
        self.register_buffer("dct_basis", basis)
        self.register_buffer("mel_mean", torch.zeros(mel_channels))
        self.register_buffer("mel_std", torch.ones(mel_channels))
        self.register_buffer("global_cap", torch.ones(mel_channels))
        self.register_buffer("slow_cap", torch.ones(mel_channels))
        self.register_buffer("fast_cap", torch.ones(mel_channels))
        self.register_buffer(
            "hybrid_quality_patch",
            torch.tensor(int(hybrid_quality_patch), dtype=torch.int32),
        )
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

    def _prior_logits(self, latent, length, mask):
        slow = F.interpolate(latent, size=math.ceil(length / 4), mode="linear", align_corners=False)
        fast = F.interpolate(latent, size=math.ceil(length / 2), mode="linear", align_corners=False)
        slow_mask = F.interpolate(mask, size=slow.size(-1), mode="nearest")
        fast_mask = F.interpolate(mask, size=fast.size(-1), mode="nearest")
        slow = self.slow_prior_context(slow * slow_mask, slow_mask)
        fast = self.fast_prior_context(fast * fast_mask, fast_mask)
        b = latent.size(0)
        slow_logits = self.slow_prior(slow).view(
            b, self.slow_fsq.dimensions, self.slow_fsq.level_count, -1
        )
        fast_logits = self.fast_prior(fast).view(
            b, self.fast_fsq.dimensions, self.fast_fsq.level_count, -1
        )
        return slow_logits, fast_logits

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
        local_prior_mix: float = 0.0,
        global_prior: bool = False,
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
        if global_prior:
            z = mu_p
        else:
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
        slow_q, slow_ids, slow_bounded = self.slow_fsq(slow_pre)
        fast_q, fast_ids, fast_bounded = self.fast_fsq(fast_pre)
        slow_soft = self.slow_fsq.soft_targets(
            slow_bounded, self.prior_soft_target_temperature
        )
        fast_soft = self.fast_fsq.soft_targets(
            fast_bounded, self.prior_soft_target_temperature
        )
        slow_logits, fast_logits = self._prior_logits(
            latent, spec.size(-1), mask
        )
        slow_prior = self.slow_fsq.expected(slow_logits)
        fast_prior = self.fast_fsq.expected(fast_logits)
        local_prior_mix = torch.as_tensor(
            local_prior_mix, device=spec.device, dtype=torch.float32
        ).clamp(0, 1)
        use_prior = (
            torch.rand(spec.size(0), 1, 1, device=spec.device)
            < local_prior_mix
        )
        slow_used = torch.where(use_prior, slow_prior, slow_q)
        fast_used = torch.where(use_prior, fast_prior, fast_q)
        slow_up = F.interpolate(slow_used, size=spec.size(-1), mode="nearest")
        fast_up = F.interpolate(fast_used, size=spec.size(-1), mode="nearest")
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
        final = (base + global_delta + slow_delta + fast_delta) * mask
        return {
            "mel": final,
            "base": base,
            "global": global_delta,
            "slow": slow_delta,
            "fast": fast_delta,
            "mask": mask,
            "global_distribution": (mu_q, logs_q, mu_p, logs_p),
            "prior": (slow_logits, fast_logits),
            "codes": (
                slow_ids,
                fast_ids,
                slow_q,
                fast_q,
                slow_soft,
                fast_soft,
            ),
            "local_prior_fraction": use_prior.float().mean(),
        }

    @staticmethod
    def factorized_prior_loss(prior, codes, mask):
        slow_logits, fast_logits = prior
        slow_soft, fast_soft = codes[4:6]

        def cross_entropy(logits, target):
            loss = -(target.float() * F.log_softmax(logits.float(), 2)).sum(2)
            local_mask = F.interpolate(
                mask.float(), size=loss.size(-1), mode="nearest"
            )
            return (loss * local_mask).sum() / (
                local_mask.sum() * loss.size(1)
            ).clamp_min(1)

        return 0.5 * (
            cross_entropy(slow_logits, slow_soft)
            + cross_entropy(fast_logits, fast_soft)
        )

    def prior_geometry(self, prior, codes, mask):
        slow_logits, fast_logits = prior
        slow_q, fast_q = codes[2:4]

        def geometry(prediction, target):
            error = F.smooth_l1_loss(
                prediction.float(),
                target.detach().float(),
                reduction="none",
            )
            local_mask = F.interpolate(
                mask.float(), size=error.size(-1), mode="nearest"
            )
            return (error * local_mask).sum() / (
                local_mask.sum() * error.size(1)
            ).clamp_min(1)

        return 0.5 * (
            geometry(self.slow_fsq.expected(slow_logits), slow_q)
            + geometry(self.fast_fsq.expected(fast_logits), fast_q)
        )

    @torch.jit.export
    def infer(
        self,
        phone,
        phone_lengths,
        pitch,
        nsff0,
        sid,
        seed: int = 0,
        noise_scale: float = 0.35,
        temperature: float = 0.65,
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
        slow_logits, fast_logits = self._prior_logits(
            latent, phone.size(1), mask
        )
        slow_q, slow_ids = self.slow_fsq.sample(
            slow_logits,
            temperature,
            min(4, self.slow_fsq.level_count),
            noise_scale,
        )
        fast_q, fast_ids = self.fast_fsq.sample(
            fast_logits,
            temperature,
            min(3, self.fast_fsq.level_count),
            noise_scale,
        )
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
        return output, mask, (z, slow_ids, fast_ids)
