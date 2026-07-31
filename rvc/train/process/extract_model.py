"""Export compact acoustic checkpoints for all registered architectures."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from collections import OrderedDict

import torch


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def extract_model(
    ckpt,
    sr,
    name,
    model_path,
    epoch,
    step,
    hps,
    vocoder="pc-NSF-HiFiGAN",
    architecture="Mel-VITS",
    pitch_guidance=True,
    version="mel-vits-1",
):
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    metadata = {}
    info_path = os.path.join(os.getcwd(), "logs", name, "model_info.json")
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    model_config = _plain(hps.model)
    model_config.update(
        {
            "spec_channels": hps.data.n_mel_channels,
            "mel_channels": hps.data.n_mel_channels,
            "segment_size": hps.train.segment_size // hps.data.hop_length,
            "sr": hps.data.sample_rate,
            "use_f0": True,
        }
    )
    if architecture == "Mel-VITS":
        model_config["training_auxiliaries"] = False
    vocoder_config = _plain(hps.vocoder) if hasattr(hps, "vocoder") else {}
    training_only_prefixes = (
        ("enc_q.", "content_speaker_classifier.", "mel_speaker_classifier.")
        if architecture == "Mel-VITS"
        else (
            "global_posterior.",
            "slow_posterior.",
            "fast_posterior.",
            "slow_prequant.",
            "fast_prequant.",
        ) if architecture == "Hybrid-FSQ" else (
            "posterior_global.",
            "posterior_local.",
            "random_area_discriminator.",
            "voicing_discriminator.",
        ) if architecture == "Stochastic-Residual-Conformer-GAN" else (
            "content_speaker_classifier.",
        )
    )
    weights = OrderedDict(
        (key, value.detach().cpu().half())
        for key, value in ckpt.items()
        if not key.startswith(training_only_prefixes)
    )
    hash_input = f"{name}-{epoch}-{step}-{version}-{model_config}"
    payload = OrderedDict(
        weight=weights,
        model_config=model_config,
        vocoder_config=vocoder_config,
        config=(
            [
                hps.data.n_mel_channels,
                model_config["segment_size"],
                hps.model.inter_channels,
                hps.model.hidden_channels,
                hps.model.filter_channels,
                hps.model.n_heads,
                hps.model.n_layers,
                hps.model.kernel_size,
                hps.model.p_dropout,
                hps.model.spk_embed_dim,
                hps.model.gin_channels,
                hps.data.sample_rate,
            ]
            if architecture == "Mel-VITS"
            else [
                hps.data.n_mel_channels,
                model_config["segment_size"],
                hps.model.hidden_channels,
                hps.model.spk_embed_dim,
                hps.model.gin_channels,
                hps.data.sample_rate,
            ] if architecture != "Raw-NSF-Waveform-GAN" else [
                hps.data.sample_rate,
                hps.data.hop_length,
                hps.model.spk_embed_dim,
                hps.model.speaker_dim,
                hps.model.content_channels,
            ]
        ),
        epoch=epoch,
        step=step,
        sr=sr,
        f0=pitch_guidance,
        version=version,
        architecture=architecture,
        vocoder=vocoder,
        creation_date=datetime.datetime.now().isoformat(),
        model_hash=hashlib.sha256(hash_input.encode()).hexdigest(),
        model_name=name,
        embedder_model=metadata.get("embedder_model"),
        speakers_id=metadata.get("speakers_id", hps.model.spk_embed_dim),
    )
    torch.save(payload, model_path)
    print(f"Saved {architecture} model '{model_path}' (epoch {epoch}, step {step})")
