"""Single-phase trainer for the Stochastic Residual Conformer-GAN."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from contextlib import contextmanager, nullcontext
import platform
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from rvc.lib.algorithm.stochastic_conformer_gan import (
    StochasticResidualConformerGAN,
    generator_adversarial_loss,
    hinge_discriminator_loss,
)
from rvc.lib.algorithm.pc_nsf_hifigan import PCNSFHiFiGAN
from rvc.train.hybrid_data import (
    HybridFSQCollate,
    HybridFSQDataset,
    PackedLocalityBatchSampler,
    attach_evaluation_parents,
    build_hybrid_packed_cache,
    build_hybrid_statistics,
    load_hybrid_packed_cache,
    split_entries_by_source,
)
from rvc.train.process.extract_model import extract_model
from rvc.train.losses import MultiResolutionSTFTLoss
from rvc.train.train_hybrid import _warm_mel_cache
from rvc.train.utils import (
    latest_checkpoint_path,
    load_config_from_json,
    load_filepaths_and_text,
    save_checkpoint,
)


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
SAVE_EVERY = _arg(2, 10, int)
EPOCHS = _arg(3, 500, int)
PRETRAIN_G = _arg(4, "")
PRETRAIN_D = _arg(5, "")
GPU_IDS = _arg(6, "0")
BATCH = _arg(7, 4, int)
SAVE_LATEST = _arg(9, True, bool)
SAVE_WEIGHTS = _arg(10, True, bool)
CLEANUP = _arg(13, False, bool)
OPTIMIZER = _arg(16, "AdamW")
OPTIMIZER_D = _arg(17, "AdamW")
CHECKPOINTING = _arg(18, True, bool)
TF32 = _arg(19, True, bool)
BENCHMARK = _arg(20, True, bool)
DETERMINISTIC = _arg(21, False, bool)
SCHEDULER = _arg(23, "exp decay step")
GAMMA = _arg(25, 0.999875, float)
LOG_INTERVAL = _arg(29, 50, int)
CUSTOM_LR = _arg(36, False, bool)
LEARNING_RATE = _arg(37, 1e-4, float)
SAVE_BEST = _arg(40, True, bool)
TORCH_COMPILE = _arg(42, False, bool)
FP16 = _arg(43, True, bool)
ACCUMULATION = max(1, _arg(44, 1, int))
VALIDATION_RATIO = min(0.25, max(0.0, _arg(49, 0.05, float)))
EMA_DECAY = min(0.99999, max(0.0, _arg(50, 0.999, float)))
EMA_IN_RAM = _arg(56, True, bool)
EMA_INTERVAL = max(1, _arg(57, 10, int))
VOCODER_VALIDATION = _arg(61, False, bool)
VOCODER_VALIDATION_BATCHES = max(1, _arg(62, 1, int))

EXPERIMENT = ROOT / "logs" / NAME
CONFIG = EXPERIMENT / "config.json"


class EMA:
    def __init__(self, model, decay, cpu=True):
        self.decay = decay
        device = "cpu" if cpu else None
        self.shadow = {
            key: value.detach().to(device=device).clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model, elapsed=1):
        decay = self.decay ** max(1, elapsed)
        for key, value in model.state_dict().items():
            average = self.shadow[key]
            value = value.detach().to(average)
            if average.is_floating_point():
                average.lerp_(value, 1.0 - decay)
            else:
                average.copy_(value)

    @contextmanager
    @torch.no_grad()
    def apply(self, model):
        current = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(current, strict=True)


def _mask_l1(prediction, target, mask):
    error = (prediction.float() - target.float()).abs() * mask.float()
    return error.sum() / (mask.sum() * prediction.size(1)).clamp_min(1.0)


def _multiscale(prediction, target, mask):
    total = prediction.new_zeros((), dtype=torch.float32)
    count = 0
    for scale in (1, 2, 4, 8):
        if prediction.size(-1) < scale:
            continue
        if scale == 1:
            p, t, m = prediction, target, mask
        else:
            p = F.avg_pool1d(prediction, scale, scale)
            t = F.avg_pool1d(target, scale, scale)
            m = F.avg_pool1d(mask, scale, scale).clamp(0, 1)
        total = total + _mask_l1(p, t, m)
        count += 1
    return total / max(1, count)


def _kl_gaussian(distribution, mask=None, free_bits=0.0):
    mu_q, logs_q, mu_p, logs_p = (item.float() for item in distribution)
    value = logs_p - logs_q + 0.5 * (
        torch.exp(2 * (logs_q - logs_p))
        + (mu_q - mu_p).square() * torch.exp(-2 * logs_p)
        - 1.0
    )
    if mask is None:
        dimensions = value.mean(0)
    else:
        value = value * mask.float()
        dimensions = value.sum((0, 2)) / mask.float().sum((0, 2)).clamp_min(1.0)
    if free_bits > 0:
        dimensions = dimensions.clamp_min(float(free_bits))
    return dimensions.sum(), dimensions


def _local_mix(step, config):
    start = float(getattr(config.train, "local_prior_mix_start", 0.2))
    end = float(getattr(config.train, "local_prior_mix_end", 0.75))
    ramp = max(1, int(getattr(config.train, "local_prior_mix_ramp_steps", 12000)))
    return start + (end - start) * min(1.0, step / ramp)


def _random_patch(value, frequency=96, time=128):
    f = min(value.size(1), frequency)
    t = min(value.size(-1), time)
    f0 = 0 if value.size(1) == f else random.randint(0, value.size(1) - f)
    t0 = 0 if value.size(-1) == t else random.randint(0, value.size(-1) - t)
    return value[:, f0 : f0 + f, t0 : t0 + t], f0, t0


def _disc_inputs(mel, pitchf, f0=0, t0=0, frequency=96, time=128):
    mel_patch = mel[:, f0 : f0 + min(frequency, mel.size(1)), t0 : t0 + min(time, mel.size(-1))]
    pitch_patch = pitchf[:, t0 : t0 + mel_patch.size(-1)]
    return mel_patch, pitch_patch


def _disc_step(discriminator, real_mel, fake_mel, real_f0=None, fake_f0=None):
    if real_f0 is None:
        real_logits, _ = discriminator(real_mel)
        fake_logits, _ = discriminator(fake_mel.detach())
    else:
        real_logits, _ = discriminator(real_mel, real_f0)
        fake_logits, _ = discriminator(fake_mel.detach(), fake_f0)
    return hinge_discriminator_loss(real_logits, fake_logits)


def _generator_gan(discriminator, real_mel, fake_mel, real_f0=None, fake_f0=None):
    if real_f0 is None:
        real_logits, real_features = discriminator(real_mel)
        fake_logits, fake_features = discriminator(fake_mel)
    else:
        real_logits, real_features = discriminator(real_mel, real_f0)
        fake_logits, fake_features = discriminator(fake_mel, fake_f0)
    feature_matching = fake_mel.new_zeros((), dtype=torch.float32)
    for real_feature, fake_feature in zip(real_features, fake_features):
        feature_matching = feature_matching + F.l1_loss(
            fake_feature, real_feature.detach()
        )
    return generator_adversarial_loss(fake_logits), feature_matching


def _save_validation_preview(
    writer,
    epoch,
    sample_index,
    global_step,
    predicted_mel,
    target_mel,
    predicted_wave=None,
    target_wave=None,
):
    """Save one colored original/generated/error panel and optional WAV pair."""
    preview_dir = (
        EXPERIMENT / "validation_samples_stochastic_conformer_gan"
        / f"epoch_{epoch:04d}"
    )
    mel_dir = preview_dir / "mel"
    audio_dir = preview_dir / "audio"
    mel_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{sample_index:02d}"
    os.environ.setdefault("MPLCONFIGDIR", str(EXPERIMENT / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_cpu = torch.nan_to_num(target_mel.detach().float().cpu())
    predicted_cpu = torch.nan_to_num(predicted_mel.detach().float().cpu())
    error_cpu = (predicted_cpu - target_cpu).abs()
    shared = torch.cat((target_cpu.flatten(), predicted_cpu.flatten()))
    shared_low = float(torch.quantile(shared, 0.01))
    shared_high = float(torch.quantile(shared, 0.99))
    error_high = max(1e-5, float(torch.quantile(error_cpu, 0.99)))
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for axis, image, title in (
        (axes[0], target_cpu, "Original mel"),
        (axes[1], predicted_cpu, "Generated mel"),
        (axes[2], error_cpu, "Absolute difference"),
    ):
        is_error = axis is axes[2]
        rendered = axis.imshow(
            image.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="inferno" if is_error else "turbo",
            vmin=0.0 if is_error else shared_low,
            vmax=error_high if is_error else shared_high,
        )
        axis.set_title(title)
        axis.set_ylabel("Mel bin")
        figure.colorbar(rendered, ax=axis, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("Frame")
    figure.suptitle(
        f"Stochastic Conformer-GAN validation — epoch {epoch}, {stem}",
        fontsize=14,
    )
    figure.canvas.draw()
    composite = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    writer.add_image(
        f"validation_previews/mel/{stem}",
        torch.from_numpy(composite).permute(2, 0, 1),
        global_step,
        dataformats="CHW",
    )
    figure.savefig(mel_dir / f"{stem}.png", dpi=130)
    plt.close(figure)
    if predicted_wave is None or target_wave is None:
        return
    audio_dir.mkdir(parents=True, exist_ok=True)
    predicted_wave = torch.nan_to_num(predicted_wave.detach().float().cpu().reshape(-1)).clamp(-1, 1)
    target_wave = torch.nan_to_num(target_wave.detach().float().cpu().reshape(-1)).clamp(-1, 1)
    sf.write(audio_dir / f"{stem}_generated.wav", predicted_wave.numpy(), 44100, subtype="PCM_16")
    sf.write(audio_dir / f"{stem}_original.wav", target_wave.numpy(), 44100, subtype="PCM_16")
    writer.add_audio(
        f"validation_previews/audio/{stem}/generated",
        predicted_wave.unsqueeze(0),
        global_step,
        sample_rate=44100,
    )
    writer.add_audio(
        f"validation_previews/audio/{stem}/original",
        target_wave.unsqueeze(0),
        global_step,
        sample_rate=44100,
    )


def _make_optimizer(parameters, lr, name):
    if name == "RAdam":
        return torch.optim.RAdam(parameters, lr=lr, betas=(0.8, 0.99), eps=1e-9)
    if name == "Sched-Free AdamW":
        from schedulefree import AdamWScheduleFree

        return AdamWScheduleFree(parameters, lr=lr, betas=(0.8, 0.99))
    if name == "Sched-Free RAdam":
        from schedulefree import RAdamScheduleFree

        return RAdamScheduleFree(parameters, lr=lr, betas=(0.8, 0.99))
    return torch.optim.AdamW(
        parameters, lr=lr, betas=(0.8, 0.99), weight_decay=1e-4
    )


def _load_pretrained_flexible(module, path, label, allowed_prefixes=None):
    """Load matching tensors from either a full or exported checkpoint.

    Pretrained RVC files generally come from a different generator revision,
    so requiring an exact state-dict match would discard useful tensors such
    as speaker embeddings.  Shape-compatible tensors are loaded and the rest
    remain freshly initialized.  This is intentionally used only when there
    is no resumable experiment checkpoint.
    """
    if not path:
        return 0
    checkpoint_path = Path(str(path))
    if not checkpoint_path.is_file():
        print(f"[Stochastic-Conformer-GAN] {label} pretrained file not found: {path}")
        return 0
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = payload
    if isinstance(payload, dict):
        keys = (
            ("discriminator", "model", "state_dict", "weight", "generator")
            if label.lower().startswith("d")
            else ("model", "weight", "state_dict", "generator", "discriminator")
        )
        for key in keys:
            if isinstance(payload.get(key), dict):
                source = payload[key]
                break
    normalized_source = {}
    for key, value in source.items():
        if not torch.is_tensor(value):
            continue
        key = str(key).removeprefix("module.").removeprefix("generator.")
        if label.lower().startswith("d"):
            key = key.removeprefix("discriminator.")
            if str(key).startswith("random_area_discriminator."):
                key = "random_area." + str(key)[len("random_area_discriminator.") :]
            elif str(key).startswith("voicing_discriminator."):
                key = "voicing." + str(key)[len("voicing_discriminator.") :]
        normalized_source[key] = value
    source = normalized_source
    target = module.state_dict()
    loaded = {}
    skipped = 0
    for key, value in source.items():
        if allowed_prefixes is not None and not any(
            key.startswith(prefix) for prefix in allowed_prefixes
        ):
            continue
        # Dataset mel statistics must always come from this experiment.
        if key in {"mel_mean", "mel_std"}:
            continue
        target_value = target.get(key)
        if target_value is not None and value.shape == target_value.shape:
            loaded[key] = value.to(dtype=target_value.dtype)
            continue
        # A one-speaker pretrain can initialize every target speaker row.
        if (
            key.endswith("emb_g.weight")
            and target_value is not None
            and value.ndim == target_value.ndim == 2
            and value.shape[1] == target_value.shape[1]
        ):
            loaded[key] = value.float().mean(0, keepdim=True).repeat(
                target_value.shape[0], 1
            ).to(dtype=target_value.dtype)
            continue
        skipped += 1
    if loaded:
        module.load_state_dict(loaded, strict=False)
    print(
        f"[Stochastic-Conformer-GAN] Loaded {len(loaded)} {label} pretrained "
        f"tensors from {checkpoint_path}; skipped {skipped}."
    )
    return len(loaded)


def _prepare_data(config, entries, train_entries, validation_entries, rank):
    packed_enabled = bool(getattr(config.data, "packed_cache", True))
    packed_cache = load_hybrid_packed_cache(entries, EXPERIMENT) if packed_enabled else None
    if rank == 0 and packed_cache is None:
        _warm_mel_cache(config, entries)
        stats = build_hybrid_statistics(train_entries, EXPERIMENT, config.data.n_mel_channels)
        if validation_entries:
            stats = attach_evaluation_parents(stats, validation_entries)
            torch.save(stats, EXPERIMENT / "hybrid_stats.pt")
        if packed_enabled:
            build_hybrid_packed_cache(
                entries,
                EXPERIMENT,
                remove_individual_mels=bool(
                    getattr(config.data, "remove_individual_mels_after_packing", True)
                ),
            )
    if packed_enabled:
        packed_cache = load_hybrid_packed_cache(entries, EXPERIMENT)
    stats_path = EXPERIMENT / "hybrid_stats.pt"
    if rank == 0 and not stats_path.is_file():
        _warm_mel_cache(config, entries)
        stats = build_hybrid_statistics(train_entries, EXPERIMENT, config.data.n_mel_channels)
        if validation_entries:
            stats = attach_evaluation_parents(stats, validation_entries)
            torch.save(stats, stats_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    if validation_entries and packed_cache is None:
        stats = attach_evaluation_parents(stats, validation_entries)
    return stats, packed_cache


def _build_loaders(config, entries, train_entries, validation_entries, stats, packed_cache, rank, world_size):
    segment_frames = int(getattr(config.train, "segment_frames", 256))
    dataset = HybridFSQDataset(
        config.data,
        entries=train_entries,
        augment=True,
        stats=stats,
        segment_frames=segment_frames,
        load_waveform=False,
        packed_cache=packed_cache,
    )
    if packed_cache is not None:
        sampler = PackedLocalityBatchSampler(
            dataset,
            BATCH,
            num_replicas=world_size,
            rank=rank,
            locality_batches=int(getattr(config.data, "packed_locality_batches", 64)),
            seed=int(getattr(config.train, "seed", 1234)),
        )
    else:
        from rvc.train.data_utils import DistributedBucketSampler

        sampler = DistributedBucketSampler(
            dataset, BATCH, [32, 64, 128, 192, 256, 384, 512, 768, 1200],
            num_replicas=world_size, rank=rank, shuffle=True,
        )
    workers = min(
        max(1, int(getattr(config.data, "packed_loader_workers" if packed_cache else "loader_workers", 2))),
        max(1, (os.cpu_count() or 2) // world_size),
    )
    prefetch = max(1, int(getattr(config.data, "packed_prefetch_factor" if packed_cache else "prefetch_factor", 2)))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=HybridFSQCollate(),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch,
    )
    validation_loader = None
    if validation_entries:
        val_dataset = HybridFSQDataset(
            config.data,
            entries=validation_entries,
            augment=False,
            stats=stats,
            segment_frames=segment_frames,
            load_waveform=VOCODER_VALIDATION,
            waveform_items=VOCODER_VALIDATION_BATCHES * BATCH,
            packed_cache=packed_cache,
        )
        validation_loader = DataLoader(
            val_dataset,
            batch_size=BATCH,
            collate_fn=HybridFSQCollate(),
            num_workers=max(1, workers // 2),
            pin_memory=True,
            persistent_workers=workers > 1,
            prefetch_factor=prefetch,
        )
    return loader, validation_loader


def _ensure_config():
    """Install the Conformer-GAN config without discarding dataset settings."""
    recommended_path = ROOT / "rvc" / "configs" / "stochastic_conformer_gan" / "44100.json"
    with recommended_path.open("r", encoding="utf-8") as handle:
        recommended = json.load(handle)
    if not CONFIG.is_file():
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG.open("w", encoding="utf-8") as handle:
            json.dump(recommended, handle, indent=4)
        print("[Stochastic-Conformer-GAN] Created the architecture config.")
        return
    with CONFIG.open("r", encoding="utf-8") as handle:
        current = json.load(handle)
    model = current.get("model", {})
    data = current.get("data", {})
    needs_update = (
        current.get("architecture") != recommended["architecture"]
        or "conformer_blocks" not in model
        or "local_latent_channels" not in model
    )
    if not needs_update:
        # Existing experiments created with the first Conformer config used
        # ramps sized for a very large pretrain.  That leaves a short
        # finetune (often only a few thousand steps) with an almost unused
        # prior.  Migrate only untouched legacy defaults; custom values are
        # preserved.
        train = current.setdefault("train", {})
        changed = False
        if train.get("local_prior_mix_ramp_steps") == 12000:
            train["local_prior_mix_ramp_steps"] = 3000
            changed = True
        if train.get("kl_warmup_steps") == 30000:
            train["kl_warmup_steps"] = 3000
            changed = True
        if changed:
            with CONFIG.open("w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=4)
            print(
                "[Stochastic-Conformer-GAN] Updated legacy prior/KL ramps "
                "for short finetunes."
            )
        return
    migrated = recommended
    if "spk_embed_dim" in model:
        migrated["model"]["spk_embed_dim"] = model["spk_embed_dim"]
    for key in ("f0_min", "f0_max"):
        if key in data:
            migrated["data"][key] = data[key]
    for key in (
        "packed_cache",
        "remove_individual_mels_after_packing",
        "packed_loader_workers",
        "packed_prefetch_factor",
        "packed_locality_batches",
    ):
        if key in data:
            migrated["data"][key] = data[key]
    with CONFIG.open("w", encoding="utf-8") as handle:
        json.dump(migrated, handle, indent=4)
    print(
        "[Stochastic-Conformer-GAN] Installed the architecture config; "
        "speaker count, F0 limits and packed-cache settings were preserved."
    )


def _train_worker(rank, world_size, gpu_ids):
    distributed = world_size > 1
    device = torch.device(
        f"cuda:{gpu_ids[rank]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29543")
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", rank=rank, world_size=world_size)
    torch.backends.cuda.matmul.allow_tf32 = TF32
    torch.backends.cudnn.allow_tf32 = TF32
    torch.backends.cudnn.benchmark = BENCHMARK and not DETERMINISTIC
    torch.backends.cudnn.deterministic = DETERMINISTIC
    random.seed(1234 + rank)
    np.random.seed(1234 + rank)
    torch.manual_seed(1234 + rank)

    config = load_config_from_json(str(CONFIG))
    entries = load_filepaths_and_text(str(EXPERIMENT / "filelist.txt"))
    train_entries, validation_entries = split_entries_by_source(
        entries, VALIDATION_RATIO, int(getattr(config.train, "seed", 1234))
    )
    stats, packed_cache = _prepare_data(
        config, entries, train_entries, validation_entries, rank
    )
    loader, validation_loader = _build_loaders(
        config,
        entries,
        train_entries,
        validation_entries,
        stats,
        packed_cache,
        rank,
        world_size,
    )
    raw_model = StochasticResidualConformerGAN(
        spec_channels=config.data.n_mel_channels,
        mel_channels=config.data.n_mel_channels,
        checkpointing=CHECKPOINTING,
        **vars(config.model),
    ).to(device)
    raw_model.set_statistics(stats)
    discriminator = torch.nn.ModuleDict(
        {
            "random_area": raw_model.random_area_discriminator,
            "voicing": raw_model.voicing_discriminator,
        }
    ).to(device)
    latest_path = EXPERIMENT / "G_latest.pth"
    numbered_resume = latest_checkpoint_path(str(EXPERIMENT), "G_[0-9]*.pth")
    resume_path = (
        latest_path
        if latest_path.is_file()
        else Path(numbered_resume) if numbered_resume else None
    )
    # A fresh run deliberately has no resume checkpoint, but it should still
    # initialize from the selected pretrained G/D files when requested.
    if resume_path is None:
        _load_pretrained_flexible(
            raw_model,
            PRETRAIN_G,
            "generator",
            allowed_prefixes=(
                "emb_g.",
                "phone.",
                "pitch.",
                "f0.",
                "speaker.",
                "onset.",
                "blocks.",
                "speaker_adaln.",
                "base_down.",
                "base_out.",
                "prior_global.",
                "prior_local.",
                "residual_decoder.",
            ),
        )
        # The stochastic architecture normally uses one acoustic pretrain
        # file.  If that file is a full training checkpoint and contains a D
        # state, reuse it; a separate D file remains optional.
        _load_pretrained_flexible(
            discriminator,
            PRETRAIN_D or PRETRAIN_G,
            "discriminator",
        )
    # The discriminators are owned by the generator module for convenient
    # export filtering, but optimized through their own optimizer.
    model_params = [p for n, p in raw_model.named_parameters() if not n.startswith(("random_area_discriminator.", "voicing_discriminator."))]
    d_params = list(discriminator.parameters())
    lr_g = LEARNING_RATE if CUSTOM_LR else float(getattr(config.train, "learning_rate_g", 1e-4))
    lr_d = float(getattr(config.train, "learning_rate_d", lr_g))
    optimizer_g = _make_optimizer(model_params, lr_g, OPTIMIZER)
    optimizer_d = _make_optimizer(d_params, lr_d, OPTIMIZER_D)

    # Resume the complete acoustic/discriminator training state when the
    # experiment already has a latest checkpoint.  The UI's "Fresh Training"
    # checkbox maps to CLEANUP; keeping it disabled is therefore the resume
    # mode for this architecture as it is for Mel-VITS.
    start_epoch = 1
    global_step = 0
    best = float("inf")
    resume_payload = {}
    if resume_path is not None and not CLEANUP:
        try:
            resume_payload = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            raw_model.load_state_dict(resume_payload["model"], strict=True)
            if resume_payload.get("optimizer"):
                optimizer_g.load_state_dict(resume_payload["optimizer"])
            if resume_payload.get("discriminator"):
                discriminator.load_state_dict(
                    resume_payload["discriminator"], strict=False
                )
            if resume_payload.get("optimizer_d"):
                optimizer_d.load_state_dict(resume_payload["optimizer_d"])
            start_epoch = int(resume_payload.get("iteration", 0)) + 1
            global_step = int(
                resume_payload.get(
                    "global_step",
                    max(0, start_epoch - 1) * max(1, len(loader)),
                )
            )
            best = float(resume_payload.get("best_loss", best))
            if rank == 0:
                print(
                    f"[Stochastic-Conformer-GAN] Resuming from {resume_path} "
                    f"(epoch {start_epoch - 1}, step {global_step})."
                )
        except (KeyError, RuntimeError, ValueError, EOFError) as error:
            raise RuntimeError(
                "G_latest.pth exists but is incompatible with the selected "
                "Stochastic-Residual-Conformer-GAN architecture. Enable "
                "'Fresh Training' to intentionally start over, or restore "
                "the matching checkpoint."
            ) from error

    scheduler_g = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer_g, GAMMA)
        if "exp" in SCHEDULER.lower() and not OPTIMIZER.startswith("Sched-Free")
        else None
    )
    scheduler_d = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer_d, GAMMA)
        if "exp" in SCHEDULER.lower() and not OPTIMIZER_D.startswith("Sched-Free")
        else None
    )
    if scheduler_g is not None and resume_payload.get("scheduler_g"):
        scheduler_g.load_state_dict(resume_payload["scheduler_g"])
    if scheduler_d is not None and resume_payload.get("scheduler_d"):
        scheduler_d.load_state_dict(resume_payload["scheduler_d"])
    scaler = GradScaler("cuda", enabled=FP16 and device.type == "cuda")
    if resume_payload.get("gradscaler"):
        scaler.load_state_dict(resume_payload["gradscaler"])
    model = raw_model
    if TORCH_COMPILE and platform.system() == "Linux":
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[gpu_ids[rank]] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    module = raw_model
    ema = EMA(module, EMA_DECAY, cpu=EMA_IN_RAM) if rank == 0 else None
    ema_updates = int(resume_payload.get("ema_updates", 0))
    # Short finetunes should validate/export the trained weights directly
    # until the EMA shadow has received enough updates to be meaningful.
    ema_ready_updates = max(10, 1000 // max(1, EMA_INTERVAL))

    def ema_context():
        if ema is not None and ema_updates >= ema_ready_updates:
            return ema.apply(module)
        return nullcontext()

    writer = SummaryWriter(str(EXPERIMENT / "tensorboard" / "stochastic_conformer_gan")) if rank == 0 else None
    for epoch in range(start_epoch, EPOCHS + 1):
        module.train()
        discriminator.train()
        if hasattr(optimizer_g, "train"):
            optimizer_g.train()
        if hasattr(optimizer_d, "train"):
            optimizer_d.train()
        running = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch", disable=rank != 0)
        for batch_index, batch in enumerate(progress):
            batch = tuple(item.to(device, non_blocking=True) for item in batch)
            phone, phone_lengths, pitch, pitchf, mel, mel_lengths, _, _, sid, _, _, _, _, _ = batch
            with autocast(device_type=device.type, enabled=FP16 and device.type == "cuda", dtype=torch.float16):
                output = model(
                    phone, phone_lengths, pitch, pitchf, mel, mel_lengths, sid,
                    local_prior_mix=_local_mix(global_step, config),
                )
            fake = output["mel"]
            real_patch, f0, t0 = _random_patch(mel)
            fake_patch, _ = _disc_inputs(fake, pitchf, f0, t0)
            real_patch, pitch_patch = _disc_inputs(mel, pitchf, f0, t0)
            if batch_index % ACCUMULATION == 0:
                optimizer_d.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=FP16 and device.type == "cuda", dtype=torch.float16):
                d_loss_random = _disc_step(discriminator["random_area"], real_patch, fake_patch)
                d_loss_voice = _disc_step(discriminator["voicing"], real_patch, fake_patch, pitch_patch, pitch_patch)
                d_loss = d_loss_random + d_loss_voice
            scaler.scale(d_loss / ACCUMULATION).backward()
            if (batch_index + 1) % ACCUMULATION == 0:
                scaler.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(d_params, float(getattr(config.train, "gradient_clip", 5.0)))
                scaler.step(optimizer_d)

            for parameter in d_params:
                parameter.requires_grad_(False)
            with autocast(device_type=device.type, enabled=FP16 and device.type == "cuda", dtype=torch.float16):
                fake_patch, _ = _disc_inputs(fake, pitchf, f0, t0)
                g_adv_r, fm_r = _generator_gan(discriminator["random_area"], real_patch, fake_patch)
                g_adv_v, fm_v = _generator_gan(discriminator["voicing"], real_patch, fake_patch, pitch_patch, pitch_patch)
                base_target = F.avg_pool1d(mel, 3, 1, 1)
                base_loss = _mask_l1(output["base"], base_target, output["mask"])
                final_loss = _mask_l1(fake, mel, output["mask"])
                multi_loss = _multiscale(fake, mel, output["mask"])
                temporal_loss = _mask_l1(fake[..., 1:] - fake[..., :-1], mel[..., 1:] - mel[..., :-1], output["mask"][..., 1:])
                frequency_loss = _mask_l1(fake[:, 1:] - fake[:, :-1], mel[:, 1:] - mel[:, :-1], output["mask"])
                kl_g, _ = _kl_gaussian(output["global_distribution"], free_bits=float(getattr(config.train, "global_free_bits", 0.15)))
                local_mask = F.interpolate(output["mask"], size=output["local_distribution"][0].size(-1), mode="nearest")
                kl_l, _ = _kl_gaussian(output["local_distribution"], local_mask, float(getattr(config.train, "local_free_bits", 0.25)))
                warm = min(1.0, global_step / max(1, int(getattr(config.train, "kl_warmup_steps", 30000))))
                adversarial = g_adv_r + g_adv_v
                feature_matching = fm_r + fm_v
                g_loss = (
                    float(config.loss.base) * base_loss
                    + float(config.loss.final) * final_loss
                    + float(config.loss.multi_scale) * multi_loss
                    + float(config.loss.temporal_delta) * temporal_loss
                    + float(config.loss.frequency_delta) * frequency_loss
                    + float(getattr(config.train, "feature_matching_weight", 1.0)) * feature_matching
                    + float(getattr(config.train, "generator_adversarial_weight", 0.05)) * adversarial
                    + warm * float(getattr(config.train, "global_kl_weight", 0.001)) * kl_g
                    + warm * float(getattr(config.train, "local_kl_weight", 0.001)) * kl_l
                )
            if batch_index % ACCUMULATION == 0:
                optimizer_g.zero_grad(set_to_none=True)
            scaler.scale(g_loss / ACCUMULATION).backward()
            for parameter in d_params:
                parameter.requires_grad_(True)
            if (batch_index + 1) % ACCUMULATION == 0:
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model_params, float(getattr(config.train, "gradient_clip", 5.0)))
                scaler.step(optimizer_g)
                scaler.update()
                if ema is not None and global_step % EMA_INTERVAL == 0:
                    # EMA is updated only every N optimizer steps.  Raise the
                    # decay to N so it has the same time constant as an EMA
                    # updated every step; otherwise the shadow remains far
                    # behind the trained model and exported audio is poor.
                    ema.update(module, EMA_INTERVAL)
                    ema_updates += 1
            global_step += 1
            running += float(g_loss.detach())
            if rank == 0:
                progress.set_postfix(loss=f"{running/(batch_index+1):.4f}", kl=f"{kl_g.item()+kl_l.item():.3f}", prior=f"{_local_mix(global_step, config):.2f}")
                if writer is not None and global_step % LOG_INTERVAL == 0:
                    for tag, value in {
                        "generator": g_loss,
                        "discriminator": d_loss,
                        "quality": final_loss + 0.5 * multi_loss + 0.2 * temporal_loss + 0.2 * frequency_loss,
                        "kl_global": kl_g,
                        "kl_local": kl_l,
                        "adversarial": adversarial,
                        "feature_matching": feature_matching,
                        "local_prior_mix": _local_mix(global_step, config),
                    }.items():
                        writer.add_scalar(f"conformer_gan/{tag}", float(value.detach()) if torch.is_tensor(value) else value, global_step)
        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()

        validation_quality = running / max(1, len(loader))
        if validation_loader is not None:
            if hasattr(optimizer_g, "eval"):
                optimizer_g.eval()
            if hasattr(optimizer_d, "eval"):
                optimizer_d.eval()
            module.eval()
            sums = 0.0
            count = 0
            validation_waveform = 0.0
            waveform_batches = 0
            validation_vocoder = None
            waveform_loss = None
            if VOCODER_VALIDATION and rank == 0:
                print("[Stochastic-Conformer-GAN] Loading pc-NSF for validation only.")
                validation_vocoder = PCNSFHiFiGAN.from_export(
                    ROOT / config.vocoder.checkpoint,
                    ROOT / config.vocoder.config,
                    map_location=device,
                ).to(device).eval()
                validation_vocoder.requires_grad_(False)
                waveform_loss = MultiResolutionSTFTLoss().to(device)
            context = ema_context()
            with context, torch.no_grad():
                for validation_index, batch in enumerate(validation_loader):
                    batch = tuple(item.to(device, non_blocking=True) for item in batch)
                    phone, phone_lengths, pitch, pitchf, mel, mel_lengths, waveform, _, sid, _, _, _, _, _ = batch
                    with autocast(device_type=device.type, enabled=FP16 and device.type == "cuda", dtype=torch.float16):
                        output = module(phone, phone_lengths, pitch, pitchf, mel, mel_lengths, sid, local_prior_mix=1.0, global_prior=True)
                    quality = _mask_l1(output["mel"], mel, output["mask"]) + 0.5 * _multiscale(output["mel"], mel, output["mask"])
                    sums += float(quality)
                    count += 1
                    if rank == 0 and validation_index < VOCODER_VALIDATION_BATCHES:
                        module_for_preview = module
                        valid_frames = int(mel_lengths[0].item())
                        predicted_mel = output["mel"][:1, :, :valid_frames]
                        target_mel = mel[:1, :, :valid_frames]
                        predicted_wave = None
                        target_wave = None
                        if validation_vocoder is not None:
                            # The generator/training loss uses normalized mel;
                            # the external pc-NSF vocoder expects the original
                            # mel scale.  Keep the normalized tensor for the
                            # image and denormalize only at the vocoder edge.
                            predicted_vocoder_mel = (
                                predicted_mel * module.mel_std[None, :, None]
                                + module.mel_mean[None, :, None]
                            )
                            with autocast(
                                device_type=device.type,
                                enabled=FP16 and device.type == "cuda",
                                dtype=torch.float16,
                            ):
                                predicted_wave = validation_vocoder(
                                    predicted_vocoder_mel,
                                    pitchf[:1, :valid_frames],
                                )
                            target_wave = waveform[:1]
                            length = min(predicted_wave.size(-1), target_wave.size(-1))
                            predicted_wave = predicted_wave[..., :length]
                            target_wave = target_wave[..., :length]
                            validation_waveform += float(
                                waveform_loss(predicted_wave, target_wave)
                            )
                            waveform_batches += 1
                        target_mel = (
                            target_mel * module_for_preview.mel_std[None, :, None]
                            + module_for_preview.mel_mean[None, :, None]
                        )
                        predicted_mel = (
                            predicted_mel * module_for_preview.mel_std[None, :, None]
                            + module_for_preview.mel_mean[None, :, None]
                        )
                        _save_validation_preview(
                            writer,
                            epoch,
                            validation_index,
                            global_step,
                            predicted_mel[0],
                            target_mel[0],
                            predicted_wave,
                            target_wave,
                        )
            validation_quality = sums / max(1, count)
            if validation_vocoder is not None:
                del validation_vocoder, waveform_loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if rank == 0:
                    print("[Stochastic-Conformer-GAN] Unloaded validation-only pc-NSF.")
            if hasattr(optimizer_g, "train"):
                optimizer_g.train()
            if hasattr(optimizer_d, "train"):
                optimizer_d.train()
            if rank == 0 and writer is not None:
                writer.add_scalar("validation/quality", validation_quality, global_step)
                if waveform_batches:
                    writer.add_scalar(
                        "validation/waveform_stft",
                        validation_waveform / waveform_batches,
                        global_step,
                    )
        if rank == 0:
            module = model.module if hasattr(model, "module") else module
            is_best = validation_quality < best
            if is_best:
                best = validation_quality
            # SAVE_LATEST selects the filename, while SAVE_EVERY controls the
            # cadence.  Saving the latest file on every epoch made the UI's
            # saving-frequency setting ineffective for this trainer.
            should_save = (
                epoch % max(1, SAVE_EVERY) == 0
                or epoch == EPOCHS
                or (SAVE_BEST and is_best)
            )
            if should_save:
                checkpoint_name = (
                    "G_latest.pth"
                    if SAVE_LATEST
                    else f"G_{global_step}.pth"
                )
                save_checkpoint(
                    raw_model,
                    optimizer_g,
                    optimizer_g.param_groups[0]["lr"],
                    epoch,
                    str(EXPERIMENT / checkpoint_name),
                    scaler,
                    extra={
                        "global_step": global_step,
                        "ema_updates": ema_updates,
                        "best_loss": best,
                        "discriminator": discriminator.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "scheduler_g": scheduler_g.state_dict()
                        if scheduler_g is not None
                        else None,
                        "scheduler_d": scheduler_d.state_dict()
                        if scheduler_d is not None
                        else None,
                    },
                )
            if SAVE_BEST and is_best:
                shutil.copy2(
                    EXPERIMENT / checkpoint_name,
                    EXPERIMENT / "G_best.pth",
                )
            if SAVE_WEIGHTS and (
                epoch % max(1, SAVE_EVERY) == 0
                or epoch == EPOCHS
                or (SAVE_BEST and is_best)
            ):
                context = ema_context()
                with context:
                    extract_model(
                        module.state_dict(),
                        "44.1k",
                        NAME,
                        str(ROOT / "assets" / "weights" / f"{NAME}_{epoch}e_{global_step}s.pth"),
                        epoch,
                        global_step,
                        config,
                        "pc-NSF-HiFiGAN",
                        "Stochastic-Residual-Conformer-GAN",
                        version="stochastic-conformer-gan-1",
                    )
            print(f"[Stochastic-Conformer-GAN] epoch={epoch} quality={validation_quality:.5f}")
    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


def main():
    _ensure_config()
    if CLEANUP:
        for path in EXPERIMENT.glob("G_*.pth"):
            path.unlink()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    gpu_ids = [int(value) for value in GPU_IDS.split("-") if value.isdigit()] if torch.cuda.is_available() and GPU_IDS != "-" else []
    world_size = max(1, len(gpu_ids))
    if world_size > 1:
        mp.spawn(_train_worker, args=(world_size, gpu_ids), nprocs=world_size)
    else:
        _train_worker(0, 1, gpu_ids)


if __name__ == "__main__":
    main()
