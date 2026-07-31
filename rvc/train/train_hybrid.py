"""End-to-end, single-phase trainer for the Hybrid-FSQ acoustic model."""

from __future__ import annotations

import math
import json
import os
import platform
import random
import shutil
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from rvc.lib.algorithm.hybrid_fsq import HybridFSQSynthesizer
from rvc.lib.algorithm.pc_nsf_hifigan import PCNSFHiFiGAN
from rvc.train.data_utils import DistributedBucketSampler, TextAudioLoaderMultiNSFsid
from rvc.train.hybrid_data import (
    HybridFSQCollate,
    HybridFSQDataset,
    attach_evaluation_parents,
    build_hybrid_statistics,
    migrate_mel_caches_to_mmap,
    split_entries_by_source,
)
from rvc.train.process.extract_model import extract_model
from rvc.train.losses import MultiResolutionSTFTLoss
from rvc.train.utils import (
    latest_checkpoint_path,
    load_checkpoint,
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
PRETRAIN = _arg(4, "")
GPU_IDS = _arg(6, "0")
BATCH = _arg(7, 4, int)
SAVE_LATEST = _arg(9, True, bool)
SAVE_WEIGHTS = _arg(10, True, bool)
CLEANUP = _arg(13, False, bool)
OPTIMIZER = _arg(16, "AdamW")
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


class RateController:
    def __init__(self, config):
        self.beta = float(getattr(config, "control_beta_initial", 0.01))
        self.integral = 0.0
        self.minimum = float(getattr(config, "control_beta_min", 1e-4))
        self.maximum = float(getattr(config, "control_beta_max", 0.1))
        self.kp = float(getattr(config, "control_kp", 0.001))
        self.ki = float(getattr(config, "control_ki", 1e-5))
        self.start = float(getattr(config, "global_rate_start_nats", 0.5))
        self.end = float(getattr(config, "global_rate_end_nats", 2.0))
        self.steps = int(getattr(config, "global_rate_ramp_steps", 30000))

    def update(self, rate: float, step: int):
        fraction = min(1.0, step / max(1, self.steps))
        target = self.start + (self.end - self.start) * fraction
        # A rate below target must reduce beta; a rate above target increases it.
        error = rate - target
        self.integral = max(-10.0, min(10.0, self.integral + error))
        self.beta = max(
            self.minimum,
            min(self.maximum, self.beta + self.kp * error + self.ki * self.integral),
        )
        return self.beta, target

    def state_dict(self):
        return {"beta": self.beta, "integral": self.integral}

    def load_state_dict(self, state):
        if state:
            self.beta = float(state.get("beta", self.beta))
            self.integral = float(state.get("integral", self.integral))


class EMA:
    def __init__(self, model, decay, cpu):
        self.decay = decay
        device = "cpu" if cpu else None
        self.shadow = {
            key: value.detach().to(device=device).clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model, elapsed=1):
        decay = self.decay ** elapsed
        for key, average in self.shadow.items():
            value = model.state_dict()[key].detach().to(average)
            if average.is_floating_point():
                average.lerp_(value, 1 - decay)
            else:
                average.copy_(value)

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state):
        if not state:
            return
        for key, value in state.get("shadow", {}).items():
            if key in self.shadow and value.shape == self.shadow[key].shape:
                self.shadow[key].copy_(value)

    @contextmanager
    def apply(self, model):
        current = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(current, strict=True)


def _mask_l1(prediction, target, mask):
    difference = (prediction.float() - target.float()).abs() * mask.float()
    return difference.sum() / (mask.sum() * prediction.size(1)).clamp_min(1)


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


def _kl_global(distribution):
    mu_q, logs_q, mu_p, logs_p = (item.float() for item in distribution)
    variance_ratio = torch.exp(2 * (logs_q - logs_p))
    mean_term = (mu_q - mu_p).square() * torch.exp(-2 * logs_p)
    return (logs_p - logs_q + 0.5 * (variance_ratio + mean_term - 1)).sum(-1).mean()


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
    """Save a small, fixed validation panel and optional waveform pair."""
    preview_dir = EXPERIMENT / "validation_samples" / f"epoch_{epoch:04d}"
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
    shared_low = float(torch.quantile(torch.cat((target_cpu.flatten(), predicted_cpu.flatten())), 0.01))
    shared_high = float(torch.quantile(torch.cat((target_cpu.flatten(), predicted_cpu.flatten())), 0.99))
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
        f"Hybrid-FSQ validation — epoch {epoch}, {stem}",
        fontsize=14,
    )
    figure.canvas.draw()
    composite = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    composite_tensor = torch.from_numpy(composite).permute(2, 0, 1)
    # One colored composite per sample/tag. TensorBoard adds each epoch as a
    # new step instead of creating three unrelated image cards.
    writer.add_image(
        f"validation_previews/mel/{stem}",
        composite_tensor,
        global_step,
        dataformats="CHW",
    )
    figure.savefig(mel_dir / f"{stem}.png", dpi=130)
    plt.close(figure)

    if predicted_wave is None:
        return
    audio_dir.mkdir(parents=True, exist_ok=True)
    predicted_wave = torch.nan_to_num(
        predicted_wave.detach().float().cpu().reshape(-1)
    ).clamp(-1, 1)
    target_wave = torch.nan_to_num(
        target_wave.detach().float().cpu().reshape(-1)
    ).clamp(-1, 1)
    sf.write(
        audio_dir / f"{stem}_generated.wav",
        predicted_wave.numpy(),
        44100,
        subtype="PCM_16",
    )
    sf.write(
        audio_dir / f"{stem}_original.wav",
        target_wave.numpy(),
        44100,
        subtype="PCM_16",
    )
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


def _losses(model, output, batch, controller, step, config, update_controller=True):
    (
        _,
        _,
        _,
        _,
        mel,
        _,
        _,
        _,
        _,
        base_target,
        global_target,
        slow_target,
        fast_target,
        _,
    ) = batch
    mask = output["mask"]
    residual_target = global_target + slow_target + fast_target
    residual_hat = output["global"] + output["slow"] + output["fast"]
    pi, slow_logits, fast_logits = output["prior"]
    slow_ids, fast_ids = output["codes"][:2]
    prior_nll, responsibilities = model.mixture_prior_loss(
        pi, slow_logits, fast_logits, slow_ids, fast_ids
    )
    geometry = model.prior_geometry(output["prior"], output["codes"], responsibilities)
    kl = _kl_global(output["global_distribution"])
    marginal_components = responsibilities.mean(0)
    effective_components = torch.exp(
        -(marginal_components * marginal_components.clamp_min(1e-8).log()).sum()
    )
    slow_histogram = torch.bincount(
        slow_ids.detach().reshape(-1), minlength=512
    ).float()
    fast_histogram = torch.bincount(
        fast_ids.detach().reshape(-1), minlength=125
    ).float()

    def effective_codes(histogram):
        probability = histogram / histogram.sum().clamp_min(1)
        return torch.exp(
            -(probability * probability.clamp_min(1e-8).log()).sum()
        )
    if update_controller:
        beta, target_rate = controller.update(float(kl.detach()), step)
    else:
        beta = controller.beta
        fraction = min(1.0, step / max(1, controller.steps))
        target_rate = controller.start + (controller.end - controller.start) * fraction
    weights = config.loss
    values = {
        "base": _mask_l1(output["base"], base_target, mask),
        "global": _mask_l1(output["global"], global_target, mask),
        "residual": _mask_l1(residual_hat, residual_target, mask),
        "slow": _mask_l1(output["slow"], slow_target, mask),
        "fast": _mask_l1(output["fast"], fast_target, mask),
        "final": _mask_l1(output["mel"], mel, mask),
        "multi_scale": _multiscale(output["mel"], mel, mask),
        "temporal_delta": _mask_l1(
            output["mel"][..., 1:] - output["mel"][..., :-1],
            mel[..., 1:] - mel[..., :-1],
            mask[..., 1:],
        ),
        "local_prior": prior_nll,
        "prior_geometry": geometry,
        "kl": kl,
        "effective_components": effective_components,
        "effective_slow_codes": effective_codes(slow_histogram),
        "effective_fast_codes": effective_codes(fast_histogram),
    }
    total = sum(
        values[key] * float(getattr(weights, key))
        for key in (
            "base",
            "global",
            "residual",
            "slow",
            "fast",
            "final",
            "multi_scale",
            "temporal_delta",
            "local_prior",
            "prior_geometry",
        )
    ) + kl * beta
    values.update(total=total, beta=beta, target_rate=target_rate)
    return values


def _warm_mel_cache(config, entries):
    missing = []
    seen = set()
    for index, entry in enumerate(entries):
        if entry[0] in seen:
            continue
        seen.add(entry[0])
        if not (
            Path(os.path.splitext(entry[0])[0] + ".mel.pt").is_file()
            or Path(os.path.splitext(entry[0])[0] + ".mel.npy").is_file()
        ):
            missing.append(index)
    if missing:
        dataset = TextAudioLoaderMultiNSFsid(
            config.data, entries=entries, augment=False
        )
        print(f"[Hybrid-FSQ] Building {len(missing)} missing pc-NSF mel caches...")
        workers = min(8, max(1, os.cpu_count() or 1))
        cache_loader = DataLoader(
            Subset(dataset, missing),
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        for _ in tqdm(cache_loader, total=len(missing), desc="Mel cache", unit="file"):
            pass


def _optimizer(model, lr, num_batches=1):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER == "RAdam":
        return torch.optim.RAdam(parameters, lr=lr, betas=(0.8, 0.99), eps=1e-9)
    if OPTIMIZER == "AdaBelief":
        from rvc.train.custom_optimizers.adabelief import AdaBelief
        return AdaBelief(parameters, lr=lr, betas=(0.8, 0.999), eps=1e-16)
    if OPTIMIZER == "Sched-Free AdamW":
        from schedulefree import AdamWScheduleFree
        return AdamWScheduleFree(parameters, lr=lr, betas=(0.8, 0.99))
    if OPTIMIZER == "Sched-Free RAdam":
        from schedulefree import RAdamScheduleFree
        return RAdamScheduleFree(parameters, lr=lr, betas=(0.8, 0.99))
    if OPTIMIZER == "Ranger21":
        from rvc.train.custom_optimizers.ranger21 import Ranger21
        return Ranger21(
            parameters,
            lr=lr,
            num_epochs=EPOCHS,
            num_batches_per_epoch=max(1, num_batches),
            use_madgrad=False,
            use_warmup=False,
            warmdown_active=False,
            use_cheb=False,
            lookahead_active=True,
            normloss_active=False,
            using_gc=True,
        )
    kwargs = dict(lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01)
    if next(model.parameters()).is_cuda:
        kwargs["fused"] = True
    return torch.optim.AdamW(parameters, **kwargs)


def _scheduler(optimizer, steps_per_epoch):
    if SCHEDULER == "none" or OPTIMIZER.startswith("Sched-Free"):
        return None, False
    if SCHEDULER == "exp decay step":
        return (
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=GAMMA ** (1 / max(1, steps_per_epoch))
            ),
            True,
        )
    if SCHEDULER == "exp decay epoch":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA), False
    if SCHEDULER == "cosine annealing epoch":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, EPOCHS), eta_min=3e-5
            ),
            False,
        )
    raise ValueError(f"Unsupported scheduler for Hybrid-FSQ: {SCHEDULER}")


def _load_flexible(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source = payload.get("model", payload.get("weight", payload))
    target = model.state_dict()
    compatible = {
        key.removeprefix("module."): value
        for key, value in source.items()
        if key.removeprefix("module.") in target
        and target[key.removeprefix("module.")].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    print(f"[Hybrid-FSQ] Loaded {len(compatible)} compatible tensors.")


def _worker(rank, world_size, gpu_ids):
    distributed = world_size > 1
    device = (
        torch.device(f"cuda:{gpu_ids[rank]}")
        if torch.cuda.is_available() and gpu_ids
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29542")
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", rank=rank, world_size=world_size)
    torch.backends.cuda.matmul.allow_tf32 = TF32
    torch.backends.cudnn.allow_tf32 = TF32
    torch.backends.cudnn.benchmark = BENCHMARK and not DETERMINISTIC
    torch.backends.cudnn.deterministic = DETERMINISTIC
    seed = 1234 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = load_config_from_json(str(CONFIG))
    entries = load_filepaths_and_text(str(EXPERIMENT / "filelist.txt"))
    train_entries, validation_entries = split_entries_by_source(
        entries, VALIDATION_RATIO, int(getattr(config.train, "seed", 1234))
    )
    if rank == 0:
        # Train-only statistics must not see validation data, but validation
        # targets still require the same cached pc-NSF mel representation.
        # Warm every cache before building/attaching either target set.
        _warm_mel_cache(config, entries)
        stats = build_hybrid_statistics(
            train_entries, EXPERIMENT, config.data.n_mel_channels
        )
        if bool(getattr(config.data, "mmap_mel_cache", True)):
            converted = migrate_mel_caches_to_mmap(
                entries,
                workers=int(getattr(config.data, "mmap_conversion_workers", 4)),
                remove_legacy=bool(
                    getattr(config.data, "remove_legacy_mel_cache", True)
                ),
            )
            if converted:
                print(
                    f"[Hybrid-FSQ] Converted {converted} mel caches to "
                    "crop-readable mmap format."
                )
    if distributed:
        dist.barrier()
    if rank != 0:
        stats = torch.load(EXPERIMENT / "hybrid_stats.pt", map_location="cpu", weights_only=False)
    if validation_entries:
        stats = attach_evaluation_parents(stats, validation_entries)

    segment_frames = int(getattr(config.train, "segment_frames", 256))
    dataset = HybridFSQDataset(
        config.data, entries=train_entries, augment=True, stats=stats,
        segment_frames=segment_frames, load_waveform=False
    )
    sampler = DistributedBucketSampler(
        dataset, BATCH, [32, 64, 128, 192, 256, 384, 512, 768, 1200],
        num_replicas=world_size, rank=rank, shuffle=True
    )
    configured_workers = int(getattr(config.data, "loader_workers", 8))
    workers = min(
        max(1, configured_workers),
        max(1, (os.cpu_count() or 2) // world_size),
    )
    prefetch_factor = max(1, int(getattr(config.data, "prefetch_factor", 4)))
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=HybridFSQCollate(),
        num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0, prefetch_factor=prefetch_factor
    )
    validation_loader = None
    if validation_entries:
        validation_dataset = HybridFSQDataset(
            config.data, entries=validation_entries, augment=False, stats=stats,
            segment_frames=segment_frames,
            load_waveform=VOCODER_VALIDATION,
            waveform_items=VOCODER_VALIDATION_BATCHES * BATCH,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=BATCH, collate_fn=HybridFSQCollate(),
            num_workers=max(1, workers // 2),
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )

    model = HybridFSQSynthesizer(
        spec_channels=config.data.n_mel_channels,
        mel_channels=config.data.n_mel_channels,
        checkpointing=CHECKPOINTING,
        **config.model,
    )
    model.set_statistics(stats)
    model.to(device)
    lr = LEARNING_RATE if CUSTOM_LR else config.train.learning_rate_g
    optimizer_steps = max(1, math.ceil(len(loader) / ACCUMULATION))
    optimizer = _optimizer(model, lr, optimizer_steps)
    scheduler, scheduler_per_step = _scheduler(
        optimizer, optimizer_steps
    )
    scaler = GradScaler("cuda", enabled=FP16 and device.type == "cuda")
    controller = RateController(config.train)
    ema = EMA(model, EMA_DECAY, EMA_IN_RAM)
    start_epoch, global_step, best = 1, 0, math.inf
    resume = EXPERIMENT / "G_latest.pth"
    resume_payload = {}
    if resume.is_file() and not CLEANUP:
        try:
            model, optimizer, _, saved_epoch, scaler_state, resume_payload = load_checkpoint(
                str(resume), model, optimizer, strict_load=True, return_extra=True
            )
            start_epoch = int(saved_epoch) + 1
            global_step = int(resume_payload.get("global_step", 0))
            best = float(resume_payload.get("best_loss", best))
            if scaler_state:
                scaler.load_state_dict(scaler_state)
        except (RuntimeError, ValueError) as error:
            print(f"[Hybrid-FSQ] Exact resume unavailable ({error}); loading compatible weights.")
            _load_flexible(model, str(resume))
    elif PRETRAIN and Path(PRETRAIN).is_file():
        _load_flexible(model, PRETRAIN)
    ema.load_state_dict(resume_payload.get("ema"))
    controller.load_state_dict(resume_payload.get("rate_controller"))
    if scheduler is not None and resume_payload.get("scheduler"):
        scheduler.load_state_dict(resume_payload["scheduler"])

    if TORCH_COMPILE and platform.system() == "Linux":
        model.compile(mode="max-autotune", dynamic=True)
        if rank == 0:
            print("[Hybrid-FSQ] torch.compile enabled.")
    elif TORCH_COMPILE and rank == 0:
        print("[Hybrid-FSQ] torch.compile is available only on Linux; using eager mode.")
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[gpu_ids[rank]] if device.type == "cuda" else None,
            broadcast_buffers=False
        )
    writer = SummaryWriter(str(EXPERIMENT)) if rank == 0 else None
    amp = FP16 and device.type == "cuda"
    if rank == 0:
        print(
            f"[Hybrid-FSQ] train={len(dataset)} validation={len(validation_entries)} "
            f"segment={segment_frames} precision={'FP16' if amp else 'FP32'}"
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, EPOCHS + 1):
        sampler.set_epoch(epoch)
        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()
        running = 0.0
        progress = tqdm(loader, disable=rank != 0, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for batch_index, batch in enumerate(progress):
            batch = tuple(item.to(device, non_blocking=True) for item in batch)
            (
                phone, phone_lengths, pitch, pitchf, mel, mel_lengths,
                _, _, sid, _, _, slow_target, fast_target, global_coeff
            ) = batch
            sync_now = (batch_index + 1) % ACCUMULATION == 0 or batch_index + 1 == len(loader)
            context = (
                model.no_sync()
                if distributed and not sync_now
                else nullcontext()
            )
            with context:
                with autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                    output = model(
                        phone, phone_lengths, pitch, pitchf, mel, mel_lengths, sid,
                        slow_target, fast_target, global_coeff
                    )
                    module = model.module if hasattr(model, "module") else model
                    values = _losses(module, output, batch, controller, global_step, config)
                scaler.scale(values["total"] / ACCUMULATION).backward()
            if sync_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None and scheduler_per_step:
                    scheduler.step()
                if global_step % EMA_INTERVAL == 0:
                    module = model.module if hasattr(model, "module") else model
                    ema.update(module, EMA_INTERVAL)
            running += float(values["total"].detach())
            if rank == 0:
                status = {
                    "loss": f"{running / (batch_index + 1):.4f}",
                    "rate": f"{values['kl'].item():.2f}",
                    "beta": f"{values['beta']:.4f}",
                    "modes": f"{values['effective_components'].item():.2f}",
                }
                if device.type == "cuda":
                    status["vram"] = f"{torch.cuda.memory_allocated(device)/2**30:.2f}G"
                progress.set_postfix(status, refresh=False)
                if sync_now and global_step % LOG_INTERVAL == 0:
                    for key, value in values.items():
                        scalar = value if isinstance(value, float) else float(value.detach())
                        writer.add_scalar(f"hybrid/{key}", scalar, global_step)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        epoch_loss = running / max(1, len(loader))
        validation_loss = None
        if validation_loader is not None:
            if hasattr(optimizer, "eval"):
                optimizer.eval()
            model.eval()
            validation_total = 0.0
            validation_waveform = 0.0
            waveform_batches = 0
            validation_vocoder = None
            waveform_loss = None
            if VOCODER_VALIDATION and rank == 0:
                print("[Hybrid-FSQ] Loading pc-NSF for validation only.")
                validation_vocoder = PCNSFHiFiGAN.from_export(
                    ROOT / config.vocoder.checkpoint,
                    ROOT / config.vocoder.config,
                    map_location=device,
                ).to(device).eval()
                validation_vocoder.requires_grad_(False)
                waveform_loss = MultiResolutionSTFTLoss().to(device)
            validation_module = (
                model.module if hasattr(model, "module") else model
            )
            with ema.apply(validation_module), torch.no_grad():
                for validation_index, validation_batch in enumerate(validation_loader):
                    validation_batch = tuple(item.to(device, non_blocking=True) for item in validation_batch)
                    (
                        phone, phone_lengths, pitch, pitchf, mel, mel_lengths,
                        _, _, sid, _, _, slow_target, fast_target, global_coeff
                    ) = validation_batch
                    with autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                        output = model(
                            phone, phone_lengths, pitch, pitchf, mel, mel_lengths, sid,
                            slow_target, fast_target, global_coeff
                        )
                        module = model.module if hasattr(model, "module") else model
                        values = _losses(
                            module,
                            output,
                            validation_batch,
                            controller,
                            global_step,
                            config,
                            update_controller=False,
                        )
                    validation_total += float(values["total"])
                    if (
                        rank == 0
                        and validation_index < VOCODER_VALIDATION_BATCHES
                    ):
                        module = model.module if hasattr(model, "module") else model
                        target_mel = (
                            mel[:1] * module.mel_std[None, :, None]
                            + module.mel_mean[None, :, None]
                        )
                        onset_energy = target_mel.mean(1)
                        source_onset = F.relu(
                            torch.diff(
                                onset_energy,
                                dim=-1,
                                prepend=onset_energy[:, :1],
                            )
                        )
                        source_onset = source_onset / source_onset.amax(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-5)
                        with autocast(
                            device_type=device.type,
                            enabled=amp,
                            dtype=torch.float16,
                        ):
                            rendered_mel = module.infer(
                                phone[:1],
                                phone_lengths[:1],
                                pitch[:1],
                                pitchf[:1],
                                sid[:1],
                                seed=1729 + validation_index,
                                noise_scale=0.5,
                                temperature=0.7,
                                source_onset=source_onset,
                            )[0]
                        valid_frames = int(mel_lengths[0].item())
                        rendered_mel = rendered_mel[..., :valid_frames]
                        target_mel = target_mel[..., :valid_frames]
                        predicted_wave = None
                        target_wave = None
                        if validation_vocoder is not None:
                            with autocast(
                                device_type=device.type,
                                enabled=amp,
                                dtype=torch.float16,
                            ):
                                predicted_wave = validation_vocoder(
                                    rendered_mel,
                                    pitchf[:1, : rendered_mel.size(-1)],
                                )
                            target_wave = validation_batch[6][
                                :1, :, : predicted_wave.size(-1)
                            ]
                            length = min(
                                predicted_wave.size(-1), target_wave.size(-1)
                            )
                            predicted_wave = predicted_wave[..., :length]
                            target_wave = target_wave[..., :length]
                            validation_waveform += float(
                                waveform_loss(predicted_wave, target_wave)
                            )
                            waveform_batches += 1
                        _save_validation_preview(
                            writer,
                            epoch,
                            validation_index,
                            global_step,
                            rendered_mel[0],
                            target_mel[0],
                            predicted_wave,
                            target_wave,
                        )
            validation_loss = validation_total / max(1, len(validation_loader))
            if validation_vocoder is not None:
                del validation_vocoder, waveform_loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if rank == 0:
                    print("[Hybrid-FSQ] Unloaded validation-only pc-NSF.")
            if rank == 0:
                writer.add_scalar("validation/total", validation_loss, global_step)
                if waveform_batches:
                    writer.add_scalar(
                        "validation/waveform_stft",
                        validation_waveform / waveform_batches,
                        global_step,
                    )
                print(
                    f"epoch={epoch} validation={validation_loss:.4f} "
                    f"wave={validation_waveform / max(1, waveform_batches):.4f}"
                )
            if hasattr(optimizer, "train"):
                optimizer.train()
        selection = validation_loss if validation_loss is not None else epoch_loss
        if scheduler is not None and not scheduler_per_step:
            scheduler.step()
        is_best = selection < best
        best = min(best, selection)
        should_save = epoch % SAVE_EVERY == 0 or epoch == EPOCHS or (SAVE_BEST and is_best)
        if rank == 0 and should_save:
            module = model.module if hasattr(model, "module") else model
            checkpoint_path = EXPERIMENT / ("G_latest.pth" if SAVE_LATEST else f"G_{global_step}.pth")
            save_checkpoint(
                model, optimizer, optimizer.param_groups[0]["lr"], epoch,
                str(checkpoint_path), scaler,
                extra={
                    "ema": ema.state_dict(),
                    "global_step": global_step,
                    "best_loss": best,
                    "rate_controller": controller.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                },
            )
            if SAVE_BEST and is_best:
                shutil.copy2(checkpoint_path, EXPERIMENT / "G_best.pth")
            if SAVE_WEIGHTS:
                output_path = ROOT / "assets" / "weights" / f"{NAME}_{epoch}e_{global_step}s.pth"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with ema.apply(module):
                    extract_model(
                        module.state_dict(), "44.1k", NAME, str(output_path), epoch,
                        global_step, config, "pc-NSF-HiFiGAN", "Hybrid-FSQ",
                        version="hybrid-fsq-1"
                    )
    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()


def main():
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    recommended_path = ROOT / "rvc" / "configs" / "hybrid_fsq" / "44100.json"
    with open(recommended_path, "r", encoding="utf-8") as handle:
        recommended_config = json.load(handle)
    if not CONFIG.is_file():
        shutil.copy2(recommended_path, CONFIG)
        current_config = recommended_config
    else:
        with open(CONFIG, "r", encoding="utf-8") as handle:
            current_config = json.load(handle)
        if current_config.get("architecture") != "Hybrid-FSQ":
            hybrid_config = recommended_config
            if "model" in current_config and "spk_embed_dim" in current_config["model"]:
                hybrid_config["model"]["spk_embed_dim"] = current_config["model"]["spk_embed_dim"]
            for key in ("f0_min", "f0_max"):
                if key in current_config.get("data", {}):
                    hybrid_config["data"][key] = current_config["data"][key]
            with open(CONFIG, "w", encoding="utf-8") as handle:
                json.dump(hybrid_config, handle, indent=4)
            print(
                "[Hybrid-FSQ] Replaced the experiment acoustic config while "
                "preserving speaker count and F0 limits."
            )
            current_config = hybrid_config
        else:
            config_changed = False
            for section in ("train", "data", "model", "loss", "vocoder"):
                current_section = current_config.setdefault(section, {})
                for key, value in recommended_config.get(section, {}).items():
                    if key not in current_section:
                        current_section[key] = value
                        config_changed = True
            if config_changed:
                with open(CONFIG, "w", encoding="utf-8") as handle:
                    json.dump(current_config, handle, indent=4)
                print("[Hybrid-FSQ] Added new recommended options to config.json.")
    if CLEANUP:
        for path in EXPERIMENT.glob("G_*.pth"):
            path.unlink()
    gpu_ids = (
        [int(value) for value in GPU_IDS.split("-") if value.isdigit()]
        if torch.cuda.is_available() and GPU_IDS != "-"
        else []
    )
    world_size = max(1, len(gpu_ids))
    if world_size > 1:
        mp.spawn(_worker, args=(world_size, gpu_ids), nprocs=world_size)
    else:
        _worker(0, 1, gpu_ids)


if __name__ == "__main__":
    main()
