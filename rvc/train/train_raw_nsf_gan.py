"""Trainer for the raw NSF waveform GAN architecture."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rvc.lib.algorithm.raw_nsf_gan import RawNSFWaveformGAN
from rvc.lib.algorithm.raw_waveform_discriminators import (
    RawWaveformDiscriminator,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from rvc.train.losses import MultiResolutionSTFTLoss
from rvc.train.mel_processing import mel_spectrogram_torch
from rvc.train.process.extract_model import extract_model
from rvc.train.utils import load_filepaths_and_text


ROOT = Path.cwd()


def _arg(index, default, cast=str):
    if len(sys.argv) <= index:
        return default
    value = sys.argv[index]
    if value.strip().lower() in {"", "none", "null"}:
        return default
    if cast is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    return cast(value)


NAME = _arg(1, "model")
SAVE_EVERY = max(1, _arg(2, 10, int))
EPOCHS = max(1, _arg(3, 500, int))
PRETRAIN_G = _arg(4, "")
GPU_IDS = _arg(6, "0")
BATCH = max(1, _arg(7, 2, int))
SAMPLE_RATE = _arg(8, 44100, int)
SAVE_LATEST = _arg(9, True, bool)
SAVE_WEIGHTS = _arg(10, True, bool)
CLEANUP = _arg(13, False, bool)
OPTIMIZER = _arg(16, "AdamW")
OPTIMIZER_D = _arg(17, "AdamW")
FP16 = _arg(43, True, bool)
ACCUMULATION = max(1, _arg(44, 1, int))
VALIDATION_RATIO = min(0.25, max(0.0, _arg(49, 0.05, float)))
EMA_DECAY = min(0.99999, max(0.0, _arg(50, 0.999, float)))
CUSTOM_LR = _arg(36, False, bool)
LEARNING_RATE = _arg(37, 1e-4, float)
LEARNING_RATE_D = _arg(38, 1e-4, float)
TORCH_COMPILE = _arg(42, False, bool)

EXPERIMENT = ROOT / "logs" / NAME
CONFIG = EXPERIMENT / "config.json"


class RawWaveDataset(Dataset):
    def __init__(self, entries, sample_rate, segment_size, augment=True):
        self.entries = list(entries)
        self.sample_rate = int(sample_rate)
        self.segment_size = int(segment_size)
        self.hop = 512
        self.augment = bool(augment)

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _read_f0(path, frames):
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
        if values.size == 0:
            return np.zeros(frames, dtype=np.float32)
        positions = np.linspace(0.0, max(0, values.size - 1), frames)
        return np.interp(positions, np.arange(values.size), values).astype(np.float32)

    def __getitem__(self, index):
        audio_path, _, _, f0_path, sid = self.entries[index]
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if int(sample_rate) != self.sample_rate:
            raise ValueError(f"Raw-NSF expects {self.sample_rate} Hz audio, got {sample_rate}")
        waveform = np.asarray(waveform, dtype=np.float32)
        full_frames = max(1, math.ceil(waveform.size / self.hop))
        f0_full = self._read_f0(f0_path, full_frames)
        if waveform.size < self.segment_size:
            pad = self.segment_size - waveform.size
            waveform = np.pad(waveform, (0, pad), mode="reflect")
            start = 0
        else:
            start = random.randint(0, waveform.size - self.segment_size) if self.augment else 0
            waveform = waveform[start : start + self.segment_size]
        first = min(f0_full.size - 1, start // self.hop)
        frames = math.ceil(self.segment_size / self.hop)
        f0 = f0_full[first : first + frames]
        if f0.size < frames:
            f0 = np.pad(f0, (0, frames - f0.size))
        return (
            torch.from_numpy(waveform.copy()).unsqueeze(0),
            torch.from_numpy(f0.astype(np.float32)),
            torch.tensor(int(sid), dtype=torch.long),
        )


class EMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            target = self.shadow[key]
            value = value.detach().to(target)
            if target.is_floating_point():
                target.lerp_(value, 1.0 - self.decay)
            else:
                target.copy_(value)

    def apply(self, model):
        current = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return current

    @staticmethod
    def restore(model, current):
        model.load_state_dict(current, strict=True)


class _AttrDict(dict):
    """Small config adapter matching the project's HParams attribute access."""

    __getattr__ = dict.__getitem__


def _make_optimizer(parameters, name, lr):
    name = str(name).lower()
    if name == "adabelief":
        try:
            from rvc.train.custom_optimizers.adabelief import AdaBelief

            return AdaBelief(parameters, lr=lr, betas=(0.8, 0.99))
        except ImportError:
            pass
    return torch.optim.AdamW(parameters, lr=lr, betas=(0.8, 0.99))


def _load_decoder_pretrain(model, path):
    requested = path
    if not path or not Path(path).is_file():
        path = os.environ.get(
            "RVC_PC_NSF_CHECKPOINT",
            "rvc/models/vocoders/pc_nsf_hifigan_44.1k_hop512_128bin.pth",
        )
    if not Path(path).is_file():
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get(
        "generator",
        payload.get("state_dict", payload.get("model", payload)),
    )
    state = {
        key.removeprefix("generator.").removeprefix("module."): value
        for key, value in state.items()
    }
    if any(key.startswith("decoder.") for key in state):
        state = {
            key.removeprefix("decoder."): value
            for key, value in state.items()
            if key.startswith("decoder.")
        }
    loaded = model.load_decoder_checkpoint(state)
    if loaded == 0 and requested and Path(path).resolve() != Path(
        os.environ.get(
            "RVC_PC_NSF_CHECKPOINT",
            "rvc/models/vocoders/pc_nsf_hifigan_44.1k_hop512_128bin.pth",
        )
    ).resolve():
        fallback = os.environ.get(
            "RVC_PC_NSF_CHECKPOINT",
            "rvc/models/vocoders/pc_nsf_hifigan_44.1k_hop512_128bin.pth",
        )
        if Path(fallback).is_file():
            return _load_decoder_pretrain(model, fallback)
    return loaded


def _mel(value, config):
    return mel_spectrogram_torch(
        value[:, 0],
        int(config["filter_length"]),
        int(config["n_mel_channels"]),
        int(config["sample_rate"]),
        int(config["hop_length"]),
        int(config["win_length"]),
        float(config["mel_fmin"]),
        float(config["mel_fmax"]),
        center=False,
    )


def _split(entries, ratio):
    entries = list(entries)
    random.Random(1234).shuffle(entries)
    count = int(len(entries) * ratio)
    return entries[count:], entries[:count]


def main():
    if SAMPLE_RATE != 44100:
        raise ValueError("Raw-NSF-Waveform-GAN supports only 44100 Hz")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if not CONFIG.is_file():
        recommended = ROOT / "rvc" / "configs" / "raw_nsf_gan" / "44100.json"
        CONFIG.write_text(recommended.read_text(encoding="utf-8"), encoding="utf-8")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    entries = load_filepaths_and_text(str(EXPERIMENT / "filelist.txt"))
    train_entries, validation_entries = _split(entries, VALIDATION_RATIO)
    segment_size = int(config["train"]["segment_size"])
    train_loader = DataLoader(
        RawWaveDataset(train_entries, SAMPLE_RATE, segment_size, augment=True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=max(0, int(config["train"].get("loader_workers", 2))),
        pin_memory=True,
        drop_last=True,
    )
    decoder_config_path = ROOT / config["model"]["decoder_config"]
    decoder_config = json.loads(decoder_config_path.read_text(encoding="utf-8"))
    device = torch.device(
        f"cuda:{GPU_IDS.split('-')[0]}"
        if torch.cuda.is_available() and GPU_IDS != "-"
        else "cpu"
    )
    if "-" in GPU_IDS and len([item for item in GPU_IDS.split("-") if item.isdigit()]) > 1:
        print("[Raw-NSF-Waveform-GAN] Multi-GPU launch is not enabled yet; using the first selected GPU.")
    model = RawNSFWaveformGAN(
        decoder_config=decoder_config,
        **{key: value for key, value in config["model"].items() if key != "decoder_config"},
    ).to(device)
    loaded = _load_decoder_pretrain(model, PRETRAIN_G)
    discriminator = RawWaveformDiscriminator(
        speaker_dim=int(config["model"]["speaker_dim"]),
        **config.get("discriminator", {}),
    ).to(device)
    params_g = list(model.parameters())
    params_d = list(discriminator.parameters())
    lr_g = LEARNING_RATE if CUSTOM_LR else float(config["train"].get("learning_rate_g", 1e-4))
    lr_d = LEARNING_RATE_D if CUSTOM_LR else float(config["train"].get("learning_rate_d", lr_g))
    optimizer_g = _make_optimizer(params_g, OPTIMIZER, lr_g)
    optimizer_d = _make_optimizer(params_d, OPTIMIZER_D, lr_d)
    scaler = GradScaler("cuda", enabled=FP16 and device.type == "cuda")
    ema = EMA(model, EMA_DECAY)
    stft_loss = MultiResolutionSTFTLoss().to(device)
    start_epoch = 1
    global_step = 0
    latest = EXPERIMENT / "G_latest.pth"
    if latest.is_file() and not CLEANUP:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        discriminator.load_state_dict(payload.get("discriminator", {}), strict=False)
        if payload.get("optimizer"):
            optimizer_g.load_state_dict(payload["optimizer"])
        if payload.get("optimizer_d"):
            optimizer_d.load_state_dict(payload["optimizer_d"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        if payload.get("ema"):
            ema.shadow = payload["ema"]
    if CLEANUP:
        for path in EXPERIMENT.glob("G_*.pth"):
            path.unlink()
    print(f"[Raw-NSF-Waveform-GAN] decoder_tensors={loaded} train={len(train_entries)} device={device}")
    mel_config = config["data"]
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        discriminator.train()
        progress = tqdm(train_loader, desc=f"Raw-NSF epoch {epoch}/{EPOCHS}")
        for batch_index, (waveform, target_f0, sid) in enumerate(progress):
            waveform = waveform.to(device, non_blocking=True)
            target_f0 = target_f0.to(device, non_blocking=True)
            sid = sid.to(device, non_blocking=True)
            do_conversion = sid.numel() > 1 and random.random() < float(
                config["train"].get("conversion_probability", 0.25)
            )
            target_sid = sid.roll(1) if do_conversion else sid
            target_real = waveform.roll(1, 0) if do_conversion else waveform
            teacher_f0 = (
                target_f0
                if global_step < int(config["train"].get("teacher_f0_steps", 10000))
                else None
            )
            with autocast(device_type=device.type, enabled=FP16 and device.type == "cuda", dtype=torch.float16):
                output = model(waveform, sid, teacher_f0)
                fake = output["waveform"]
                speaker = output["speaker"]
                real_logits, _ = discriminator(waveform, speaker.detach())
                fake_logits, _ = discriminator(fake.detach(), speaker.detach())
                d_loss = discriminator_hinge_loss(real_logits, fake_logits)
                if do_conversion:
                    converted_d = model(waveform, target_sid, teacher_f0)
                    target_speaker = converted_d["speaker"]
                    target_real_logits, _ = discriminator(
                        target_real, target_speaker.detach()
                    )
                    converted_logits, _ = discriminator(
                        converted_d["waveform"].detach(), target_speaker.detach()
                    )
                    d_loss = 0.5 * (
                        d_loss
                        + discriminator_hinge_loss(
                            target_real_logits, converted_logits
                        )
                    )
            optimizer_d.zero_grad(set_to_none=True)
            scaler.scale(d_loss / ACCUMULATION).backward()
            scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(params_d, float(config["train"].get("gradient_clip", 5.0)))
            scaler.step(optimizer_d)

            for parameter in params_d:
                parameter.requires_grad_(False)
            with autocast(device_type=device.type, enabled=FP16 and device.type == "cuda", dtype=torch.float16):
                fake_logits, fake_features = discriminator(fake, speaker)
                real_logits_g, real_features = discriminator(waveform, speaker)
                loss_stft = stft_loss(fake, waveform)
                loss_mel = F.l1_loss(_mel(fake, mel_config), _mel(waveform, mel_config))
                frames = min(target_f0.size(-1), output["log_f0"].size(-1))
                target = target_f0[..., :frames]
                pred_log_f0 = output["log_f0"][..., :frames]
                voiced = target > 0
                loss_pitch = F.smooth_l1_loss(
                    pred_log_f0[voiced], target.clamp_min(1.0).log()[voiced]
                ) if voiced.any() else pred_log_f0.new_zeros(())
                loss_uv = F.binary_cross_entropy_with_logits(
                    output["uv_logits"][..., :frames], voiced.float()
                )
                energy_target = F.adaptive_avg_pool1d(
                    waveform.abs(), output["energy"].size(-1)
                )[:, 0, :frames].log1p()
                loss_energy = F.smooth_l1_loss(
                    output["energy"][..., :frames], energy_target
                )
                loss_fm = feature_matching_loss(real_features, fake_features)
                loss_adv = generator_adversarial_loss(fake_logits)
                loss_spk = F.cross_entropy(output["content_speaker_logits"], sid)
                loss_content = fake.new_zeros(())
                loss_conversion_adv = fake.new_zeros(())
                if do_conversion:
                    converted = converted_d["waveform"]
                    converted_logits_g, _ = discriminator(
                        converted, converted_d["speaker"]
                    )
                    loss_conversion_adv = generator_adversarial_loss(
                        converted_logits_g
                    )
                    converted_padded, _ = model._pad_waveform(converted)
                    converted_content = model._encode(
                        converted_padded, target_sid
                    )["content"]
                    loss_content = F.smooth_l1_loss(
                        converted_content,
                        output["content"].detach(),
                    )
                adv_weight = float(config["train"].get("adversarial_weight_end", 1.0)) * min(
                    1.0,
                    global_step / max(1, int(config["train"].get("adversarial_ramp_steps", 30000))),
                )
                g_loss = (
                    float(config["train"].get("stft_weight", 5.0)) * loss_stft
                    + float(config["train"].get("mel_weight", 10.0)) * loss_mel
                    + float(config["train"].get("pitch_weight", 2.0)) * loss_pitch
                    + float(config["train"].get("uv_weight", 1.0)) * loss_uv
                    + float(config["train"].get("energy_weight", 0.5)) * loss_energy
                    + float(config["train"].get("feature_matching_weight", 2.0)) * loss_fm
                    + adv_weight * loss_adv
                    + adv_weight * loss_conversion_adv
                    + float(config["train"].get("content_weight", 1.0)) * loss_content
                    + float(config["train"].get("content_speaker_weight", 0.1)) * loss_spk
                )
            optimizer_g.zero_grad(set_to_none=True)
            scaler.scale(g_loss / ACCUMULATION).backward()
            for parameter in params_d:
                parameter.requires_grad_(True)
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(params_g, float(config["train"].get("gradient_clip", 5.0)))
            scaler.step(optimizer_g)
            scaler.update()
            ema.update(model)
            global_step += 1
            progress.set_postfix(g=f"{float(g_loss):.3f}", d=f"{float(d_loss):.3f}", stft=f"{float(loss_stft):.3f}")
        current = ema.apply(model)
        checkpoint = EXPERIMENT / ("G_latest.pth" if SAVE_LATEST else f"G_{global_step}.pth")
        torch.save(
            {
                "model": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "ema": ema.shadow,
            },
            checkpoint,
        )
        if SAVE_WEIGHTS and (epoch % SAVE_EVERY == 0 or epoch == EPOCHS):
            export_config = _AttrDict(
                model=_AttrDict(config["model"]),
                data=_AttrDict(config["data"]),
                train=_AttrDict(config["train"]),
                vocoder=_AttrDict(config.get("vocoder", {})),
            )
            extract_model(
                model.state_dict(),
                "44.1k",
                NAME,
                str(ROOT / "assets" / "weights" / f"{NAME}_{epoch}e_{global_step}s.pth"),
                epoch,
                global_step,
                export_config,
                "pc-NSF-HiFiGAN",
                "Raw-NSF-Waveform-GAN",
                version="raw-nsf-waveform-gan-1",
            )
        ema.restore(model, current)
