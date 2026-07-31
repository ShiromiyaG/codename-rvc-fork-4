"""Training entry point for the mel-VITS voice-conversion architecture.

The positional CLI remains compatible with the UI from fork-4.  Legacy
discriminator/vocoder arguments are accepted but ignored so existing presets
continue to launch; only the acoustic model is optimized.
"""

from __future__ import annotations

import glob
import gc
import math
import os
import platform
import random
import shutil
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

# Must be configured before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from rvc.lib.algorithm.commons import slice_segments
from rvc.lib.algorithm.pc_nsf_hifigan import PCNSFHiFiGAN
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollateMultiNSFsid,
    TextAudioLoaderMultiNSFsid,
)
from rvc.train.losses import MultiResolutionSTFTLoss, kl_loss_fb
from rvc.train.process.extract_model import extract_model
from rvc.train.utils import (
    latest_checkpoint_path,
    load_checkpoint,
    load_config_from_json,
    load_filepaths_and_text,
    save_checkpoint,
)


def _arg(index: int, default, cast=str):
    if len(sys.argv) <= index:
        return default
    value = sys.argv[index]
    if value.strip().lower() in {"", "none", "null"}:
        return default
    if cast is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    return cast(value)


MODEL_NAME = _arg(1, "model")
SAVE_EVERY = _arg(2, 10, int)
EPOCHS = _arg(3, 500, int)
PRETRAIN_G = _arg(4, "")
_LEGACY_PRETRAIN_D = _arg(5, "")
GPU_IDS = _arg(6, "0")
BATCH_SIZE = _arg(7, 4, int)
SAMPLE_RATE = _arg(8, 44100, int)
SAVE_LATEST = _arg(9, True, bool)
SAVE_WEIGHTS = _arg(10, True, bool)
USE_WARMUP = _arg(11, False, bool)
WARMUP_EPOCHS = _arg(12, 0, int)
CLEANUP = _arg(13, False, bool)
_LEGACY_VOCODER = _arg(14, "pc-NSF-HiFiGAN")
_LEGACY_ARCH = _arg(15, "Mel-VITS")
OPTIMIZER_NAME = _arg(16, "AdamW")
_LEGACY_D_OPTIMIZER = _arg(17, "AdamW")
CHECKPOINTING = _arg(18, True, bool)
USE_TF32 = _arg(19, True, bool)
CUDNN_BENCHMARK = _arg(20, True, bool)
DETERMINISTIC = _arg(21, False, bool)
_LOSS_NAME = _arg(22, "L1 Mel Loss")
SCHEDULER_NAME = _arg(23, "exp decay step")
_LEGACY_D_SCHEDULER = _arg(24, "none")
SCHEDULER_GAMMA = _arg(25, 0.999875, float)
_LEGACY_D_GAMMA = _arg(26, 1.0, float)
KL_ANNEALING = _arg(27, True, bool)
KL_CYCLE_EPOCHS = _arg(28, 20, int)
LOG_INTERVAL = _arg(29, 50, int)
CLIP_SCHEDULE = _arg(30, False, bool)
CLIP_STEPS = _arg(31, 0, int)
CLIP_INITIAL = _arg(32, 0.0, float)
_LEGACY_D_CLIP_INITIAL = _arg(33, 0.0, float)
CLIP_FINAL = _arg(34, 0.0, float)
_LEGACY_D_CLIP_FINAL = _arg(35, 0.0, float)
CUSTOM_LR = _arg(36, False, bool)
CUSTOM_LR_G = _arg(37, 1e-4, float)
_LEGACY_CUSTOM_LR_D = _arg(38, 1e-4, float)
TWO_SAMPLE_KL = _arg(39, False, bool)
SAVE_BEST = _arg(40, True, bool)
_LEGACY_DOUBLE_D = _arg(41, False, bool)
USE_TORCH_COMPILE = _arg(
    42, os.environ.get("RVC_TORCH_COMPILE", "0").lower() in {"1", "true", "yes"}, bool
)
USE_FP16 = _arg(43, True, bool)
GRADIENT_ACCUMULATION = max(1, _arg(44, 1, int))
KL_FREE_BITS = max(0.0, _arg(45, 0.5, float))
WAVEFORM_LOSS_WEIGHT = max(0.0, _arg(46, 1.0, float))
WAVEFORM_LOSS_INTERVAL = max(1, _arg(47, 4, int))
WAVEFORM_LOSS_FRAMES = max(16, _arg(48, 128, int))
VALIDATION_RATIO = min(0.25, max(0.0, _arg(49, 0.05, float)))
EMA_DECAY = min(0.99999, max(0.0, _arg(50, 0.999, float)))
SPEAKER_BALANCE_TEMPERATURE = min(1.0, max(0.0, _arg(51, 0.5, float)))
CONTENT_ADVERSARIAL_WEIGHT = max(0.0, _arg(52, 0.1, float))
SPEAKER_CLASSIFICATION_WEIGHT = max(0.0, _arg(53, 0.5, float))
PITCH_AUGMENTATION_PROBABILITY = min(1.0, max(0.0, _arg(54, 0.2, float)))
PITCH_AUGMENTATION_SEMITONES = max(0.0, _arg(55, 2.0, float))
EMA_IN_RAM = _arg(56, True, bool)
EMA_UPDATE_INTERVAL = max(1, _arg(57, 10, int))
BRANCHWISE_TRAINING = _arg(58, True, bool)
WAVEFORM_MICROBATCH_SIZE = max(1, _arg(59, 1, int))
USE_SDPA = _arg(60, True, bool)
VOCODER_VALIDATION_ONLY = _arg(61, False, bool)
VALIDATION_VOCODER_BATCHES = max(1, _arg(62, 1, int))

EXPERIMENT_DIR = ROOT / "logs" / MODEL_NAME
CONFIG_PATH = EXPERIMENT_DIR / "config.json"


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float, device=None):
        self.decay = decay
        self.device = torch.device(device) if device is not None else None
        self.shadow = {
            key: value.detach().to(self.device).clone()
            if self.device is not None
            else value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module, elapsed_steps: int = 1) -> None:
        state = model.state_dict()
        effective_decay = self.decay ** max(1, elapsed_steps)
        for key, average in self.shadow.items():
            value = state[key].detach().to(
                device=average.device, dtype=average.dtype
            )
            if average.is_floating_point():
                average.lerp_(value, 1.0 - effective_decay)
            else:
                average.copy_(value)

    def state_dict(self):
        return {
            "decay": self.decay,
            "device": "cpu" if self.device is not None else "model",
            "shadow": self.shadow,
        }

    def load_state_dict(self, state) -> None:
        if not state:
            return
        self.decay = float(state.get("decay", self.decay))
        saved = state.get("shadow", state)
        for key, value in saved.items():
            if key in self.shadow and self.shadow[key].shape == value.shape:
                self.shadow[key].copy_(value)

    @contextmanager
    def apply(self, model: nn.Module):
        current = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(current, strict=True)


def _split_entries(entries, ratio: float, seed: int):
    if ratio <= 0 or len(entries) < 3:
        return entries, []
    groups = {}
    for entry in entries:
        if "mute" not in Path(entry[0]).name.lower():
            groups.setdefault(entry[4], []).append(entry)
    validation_ids = set()
    rng = random.Random(seed)
    for speaker_entries in groups.values():
        if len(speaker_entries) < 2:
            continue
        shuffled = list(speaker_entries)
        rng.shuffle(shuffled)
        count = min(len(shuffled) - 1, max(1, round(len(shuffled) * ratio)))
        validation_ids.update(id(entry) for entry in shuffled[:count])
    train = [entry for entry in entries if id(entry) not in validation_ids]
    validation = [entry for entry in entries if id(entry) in validation_ids]
    return train, validation


def _load_pretrained_flexible(model: nn.Module, path: str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source = payload.get("model", payload.get("weight", payload))
    source = {key.removeprefix("module."): value for key, value in source.items()}
    target = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in source.items():
        if key not in target:
            skipped.append(key)
            continue
        if value.shape == target[key].shape:
            loaded[key] = value.to(dtype=target[key].dtype)
        elif (
            key == "emb_g.weight"
            and value.ndim == 2
            and value.shape[1] == target[key].shape[1]
        ):
            loaded[key] = value.float().mean(dim=0, keepdim=True).repeat(
                target[key].shape[0], 1
            ).to(dtype=target[key].dtype)
        else:
            skipped.append(key)
    missing, _ = model.load_state_dict(loaded, strict=False)
    print(
        f"[Mel-VITS] Loaded {len(loaded)} pretrained tensors; "
        f"{len(skipped)} incompatible and {len(missing)} newly initialized."
    )


def _make_optimizer(
    model: nn.Module, name: str, lr: float, num_batches: int
) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if name == "AdamW":
        kwargs = dict(lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01)
        if torch.cuda.is_available():
            kwargs["fused"] = True
        return torch.optim.AdamW(params, **kwargs)
    if name == "RAdam":
        return torch.optim.RAdam(
            params, lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.01
        )
    if name == "AdaBelief":
        from rvc.train.custom_optimizers.adabelief import AdaBelief

        return AdaBelief(
            params,
            lr=lr,
            betas=(0.8, 0.999),
            eps=1e-16,
            weight_decay=0,
            rectify=False,
        )
    if name == "Ranger21":
        from rvc.train.custom_optimizers.ranger21 import Ranger21

        return Ranger21(
            params,
            lr=lr,
            num_epochs=EPOCHS,
            num_batches_per_epoch=num_batches,
            use_madgrad=False,
            use_warmup=False,
            warmdown_active=False,
            use_cheb=False,
            lookahead_active=True,
            normloss_active=False,
            using_gc=True,
        )
    if name == "Sched-Free AdamW":
        from schedulefree import AdamWScheduleFree

        return AdamWScheduleFree(
            params,
            lr=lr,
            betas=(0.8, 0.99),
            eps=1e-9,
            weight_decay=0.01,
            warmup_steps=WARMUP_EPOCHS * num_batches if USE_WARMUP else 0,
        )
    if name == "Sched-Free RAdam":
        from schedulefree import RAdamScheduleFree

        return RAdamScheduleFree(
            params, lr=lr, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _make_scheduler(
    optimizer,
    steps_per_epoch: int,
    start_epoch: int,
    global_step: int = 0,
):
    if SCHEDULER_NAME == "none" or OPTIMIZER_NAME.startswith("Sched-Free"):
        return None, False
    if SCHEDULER_NAME == "exp decay step":
        gamma = SCHEDULER_GAMMA ** (1.0 / max(1, steps_per_epoch))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )
        per_step = True
        completed_intervals = max(0, global_step)
    elif SCHEDULER_NAME == "exp decay epoch":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=SCHEDULER_GAMMA,
        )
        per_step = False
        completed_intervals = max(0, start_epoch - 1)
    elif SCHEDULER_NAME == "cosine annealing epoch":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, EPOCHS),
            eta_min=3e-5,
        )
        per_step = False
        completed_intervals = max(0, start_epoch - 1)
    else:
        raise ValueError(f"Unsupported scheduler: {SCHEDULER_NAME}")

    # Construct with PyTorch's fresh-scheduler defaults so it does not require
    # initial_lr or advance the loaded optimizer LR during initialization.
    # Older Mel-VITS checkpoints did not store scheduler state, so position the
    # new scheduler at the already-completed interval without performing a step.
    if completed_intervals:
        scheduler.last_epoch = completed_intervals
        scheduler._step_count = completed_intervals + 1
    return scheduler, per_step


def _configure_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = USE_TF32
    torch.backends.cudnn.allow_tf32 = USE_TF32
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK and not DETERMINISTIC
    torch.backends.cudnn.deterministic = DETERMINISTIC
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if USE_TF32 else "highest")


def _build_model(config) -> Synthesizer:
    return Synthesizer(
        spec_channels=config.data.n_mel_channels,
        segment_size=config.train.segment_size // config.data.hop_length,
        sr=config.data.sample_rate,
        checkpointing=CHECKPOINTING,
        use_2_sample_kl=TWO_SAMPLE_KL,
        **config.model,
    )


def _different_sids(sids: torch.Tensor, speaker_count: int) -> torch.Tensor:
    if speaker_count < 2:
        return sids
    offsets = torch.randint(1, speaker_count, sids.shape, device=sids.device)
    return (sids + offsets) % speaker_count


def _masked_l1(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    predicted = predicted.float()
    target = target.float()
    mask = mask.float()
    channels = predicted.shape[1]
    return ((predicted - target).abs() * mask).sum() / (
        mask.sum().clamp_min(1.0) * channels
    )


def _waveform_objective(
    pc_vocoder,
    waveform_loss,
    mel_hat,
    ids,
    pitchf,
    wave,
    hop_length,
):
    """Render a small batch with recomputation instead of retaining vocoder activations."""
    count = min(WAVEFORM_MICROBATCH_SIZE, mel_hat.size(0))
    if count < mel_hat.size(0):
        selection = torch.randperm(mel_hat.size(0), device=mel_hat.device)[:count]
        mel_hat = mel_hat.index_select(0, selection)
        ids = ids.index_select(0, selection)
        pitchf = pitchf.index_select(0, selection)
        wave = wave.index_select(0, selection)
    frames = min(WAVEFORM_LOSS_FRAMES, mel_hat.size(-1))
    f0_segment = slice_segments(
        pitchf, ids, mel_hat.size(-1), 2
    )[..., :frames]
    predicted_wave = activation_checkpoint(
        pc_vocoder,
        mel_hat[..., :frames],
        f0_segment,
        use_reentrant=False,
    )
    target_wave = slice_segments(
        wave,
        ids * hop_length,
        frames * hop_length,
        3,
    )
    length = min(predicted_wave.size(-1), target_wave.size(-1))
    return activation_checkpoint(
        waveform_loss,
        predicted_wave[..., :length],
        target_wave[..., :length],
        use_reentrant=False,
    )


def _cleanup_old_checkpoints() -> None:
    for pattern in ("G_*.pth", "D_*.pth", "events.out.tfevents.*"):
        for path in EXPERIMENT_DIR.glob(pattern):
            path.unlink()


def _save_small_model(model, config, epoch: int, step: int) -> None:
    if not SAVE_WEIGHTS:
        return
    weights_dir = ROOT / "assets" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    extract_model(
        module.state_dict(),
        "44.1k",
        MODEL_NAME,
        str(weights_dir / f"{MODEL_NAME}_{epoch}e_{step}s.pth"),
        epoch,
        step,
        config,
        "pc-NSF-HiFiGAN",
        "Mel-VITS",
    )


@torch.no_grad()
def _validate(
    model,
    loader,
    device,
    use_amp: bool,
    pc_vocoder=None,
    waveform_loss=None,
    hop_length: int = 512,
) -> dict[str, float]:
    module = model.module if hasattr(model, "module") else model
    module.eval()
    totals = {"total": 0.0, "mel": 0.0, "kl": 0.0, "waveform": 0.0}
    batches = 0
    waveform_batches = 0
    torch.manual_seed(271828)
    for batch in loader:
        batch = [item.to(device, non_blocking=True) for item in batch]
        phone, phone_lengths, pitch, pitchf, mel, mel_lengths, wave, _, sid = batch
        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            mel_hat, ids, _, mel_mask, latent, _ = module(
                phone, phone_lengths, pitch, pitchf, mel, mel_lengths, sid
            )
            _, z_p, z_p2, m_p, logs_p, _, logs_q = latent
            target = slice_segments(mel, ids, mel_hat.size(-1), 3)
            target_mask = slice_segments(mel_mask, ids, mel_hat.size(-1), 3)
            loss_mel = _masked_l1(mel_hat, target, target_mask)
            loss_kl = kl_loss_fb(
                z_p, logs_q, m_p, logs_p, mel_mask, z_p2, free_bits=KL_FREE_BITS
            )
            total = loss_mel + loss_kl * 0.01
            loss_waveform = mel_hat.new_zeros(())
            if (
                pc_vocoder is not None
                and waveform_loss is not None
                and waveform_batches < VALIDATION_VOCODER_BATCHES
            ):
                count = min(WAVEFORM_MICROBATCH_SIZE, mel_hat.size(0))
                frames = min(WAVEFORM_LOSS_FRAMES, mel_hat.size(-1))
                mel_validation = mel_hat[:count, :, :frames]
                ids_validation = ids[:count]
                f0_validation = slice_segments(
                    pitchf[:count],
                    ids_validation,
                    mel_hat.size(-1),
                    2,
                )[..., :frames]
                predicted_wave = pc_vocoder(
                    mel_validation, f0_validation
                )
                target_wave = slice_segments(
                    wave[:count],
                    ids_validation * hop_length,
                    frames * hop_length,
                    3,
                )
                length = min(
                    predicted_wave.size(-1), target_wave.size(-1)
                )
                loss_waveform = waveform_loss(
                    predicted_wave[..., :length],
                    target_wave[..., :length],
                )
                waveform_batches += 1
        totals["total"] += total.float().item()
        totals["mel"] += loss_mel.float().item()
        totals["kl"] += loss_kl.float().item()
        totals["waveform"] += loss_waveform.float().item()
        batches += 1
    module.train()
    result = {
        key: value / max(1, batches)
        for key, value in totals.items()
        if key != "waveform"
    }
    result["waveform"] = totals["waveform"] / max(1, waveform_batches)
    return result


def _train_worker(rank: int, world_size: int, gpu_ids: list[int]) -> None:
    distributed = world_size > 1
    if torch.cuda.is_available() and gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[rank]}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29541")
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    _configure_runtime()
    seed = 1234 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = load_config_from_json(str(CONFIG_PATH))
    config.train.fp16_run = USE_FP16
    config.data.training_files = str(EXPERIMENT_DIR / "filelist.txt")
    config.model.use_sdpa = USE_SDPA
    config.data.pitch_augmentation_probability = PITCH_AUGMENTATION_PROBABILITY
    config.data.pitch_augmentation_semitones = PITCH_AUGMENTATION_SEMITONES
    if config.data.sample_rate != 44100 or config.data.hop_length != 512:
        raise ValueError("Mel-VITS requires the 44.1 kHz / hop 512 pc-NSF frontend")

    entries = load_filepaths_and_text(config.data.training_files)
    train_entries, validation_entries = _split_entries(
        entries, VALIDATION_RATIO, int(getattr(config.train, "seed", 1234))
    )
    dataset = TextAudioLoaderMultiNSFsid(
        config.data, entries=train_entries, augment=True
    )
    sampler = DistributedBucketSampler(
        dataset,
        BATCH_SIZE,
        [32, 50, 100, 200, 300, 400, 500, 700, 900, 1200],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        speaker_balance_temperature=SPEAKER_BALANCE_TEMPERATURE,
    )
    workers = min(8, max(1, (os.cpu_count() or 2) // world_size))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=TextAudioCollateMultiNSFsid(),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        prefetch_factor=2,
    )
    validation_loader = None
    if validation_entries:
        validation_dataset = TextAudioLoaderMultiNSFsid(
            config.data, entries=validation_entries, augment=False
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=TextAudioCollateMultiNSFsid(),
            num_workers=max(1, workers // 2),
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
        )

    model = _build_model(config).to(device)
    lr = CUSTOM_LR_G if CUSTOM_LR else config.train.learning_rate_g
    optimizer_steps_per_epoch = max(1, math.ceil(len(loader) / GRADIENT_ACCUMULATION))
    optimizer = _make_optimizer(
        model, OPTIMIZER_NAME, lr, optimizer_steps_per_epoch
    )
    optimizer_initial_lrs = [group["lr"] for group in optimizer.param_groups]
    start_epoch = 1
    global_step = 0
    amp_enabled = USE_FP16 and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)
    if rank == 0:
        runtime_precision = "FP16 mixed precision" if amp_enabled else "FP32"
        print(f"[Mel-VITS] Training precision: {runtime_precision}")

    latest_path = EXPERIMENT_DIR / "G_latest.pth"
    resume_path = (
        str(latest_path)
        if latest_path.is_file()
        else latest_checkpoint_path(str(EXPERIMENT_DIR), "G_[0-9]*.pth")
    )
    resume_payload = {}
    if resume_path and not CLEANUP:
        try:
            model, optimizer, _, saved_epoch, scaler_state, resume_payload = (
                load_checkpoint(
                    resume_path,
                    model,
                    optimizer,
                    strict_load=True,
                    return_extra=True,
                )
            )
            start_epoch = int(saved_epoch) + 1
            global_step = int(
                resume_payload.get(
                    "global_step",
                    (start_epoch - 1) * optimizer_steps_per_epoch,
                )
            )
            if scaler_state:
                scaler.load_state_dict(scaler_state)
        except (RuntimeError, ValueError) as error:
            print(
                "[Mel-VITS] The checkpoint cannot be resumed exactly after "
                f"the architecture update ({error}). Loading compatible model "
                "weights and restarting optimizer/epoch state."
            )
            optimizer = _make_optimizer(
                model, OPTIMIZER_NAME, lr, optimizer_steps_per_epoch
            )
            _load_pretrained_flexible(model, resume_path)
            resume_payload = {}
    elif PRETRAIN_G and Path(PRETRAIN_G).is_file():
        _load_pretrained_flexible(model, PRETRAIN_G)

    ema = ModelEMA(model, EMA_DECAY, device="cpu" if EMA_IN_RAM else None)
    ema.load_state_dict(resume_payload.get("ema"))
    ema_update_interval = EMA_UPDATE_INTERVAL if EMA_IN_RAM else 1
    last_ema_step = global_step
    for group, initial_lr in zip(
        optimizer.param_groups, optimizer_initial_lrs, strict=True
    ):
        group.setdefault("initial_lr", initial_lr)
    scheduler, per_step_scheduler = _make_scheduler(
        optimizer,
        optimizer_steps_per_epoch,
        start_epoch,
        global_step,
    )
    if scheduler is not None and resume_payload.get("scheduler"):
        scheduler.load_state_dict(resume_payload["scheduler"])
    warmup_steps = (
        WARMUP_EPOCHS * optimizer_steps_per_epoch if USE_WARMUP else 0
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    compile_enabled = USE_TORCH_COMPILE and platform.system() == "Linux"
    if USE_TORCH_COMPILE and not compile_enabled and rank == 0:
        print("[Mel-VITS] torch.compile is supported only on Linux; using eager mode.")
    if compile_enabled:
        model.compile(mode="max-autotune", dynamic=True)
        if rank == 0:
            print("[Mel-VITS] torch.compile enabled (Linux, max-autotune, dynamic).")

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[gpu_ids[rank]] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    writer = SummaryWriter(str(EXPERIMENT_DIR)) if rank == 0 else None
    best_loss = math.inf
    best_loss = float(resume_payload.get("best_loss", best_loss))
    use_amp = scaler.is_enabled()
    branchwise_enabled = BRANCHWISE_TRAINING and not distributed
    if BRANCHWISE_TRAINING and distributed and rank == 0:
        print(
            "[Mel-VITS] Branchwise waveform backward is disabled under DDP "
            "to preserve reducer correctness."
        )
    waveform_loss = MultiResolutionSTFTLoss().to(device)
    pc_vocoder = None
    if WAVEFORM_LOSS_WEIGHT > 0 and not VOCODER_VALIDATION_ONLY:
        pc_vocoder = PCNSFHiFiGAN.from_export(
            ROOT / config.vocoder.checkpoint,
            ROOT / config.vocoder.config,
            map_location=device,
        ).to(device)
    if rank == 0:
        print(
            f"[Mel-VITS] train={len(dataset)} validation="
            f"{len(validation_entries)} accumulation={GRADIENT_ACCUMULATION} "
            f"speaker_temperature={SPEAKER_BALANCE_TEMPERATURE:.2f} "
            f"ema={'RAM' if EMA_IN_RAM else 'GPU'} "
            f"branchwise={branchwise_enabled} "
            f"waveform_microbatch={WAVEFORM_MICROBATCH_SIZE} "
            f"vocoder_mode="
            f"{'validation-only' if VOCODER_VALIDATION_ONLY else ('training' if WAVEFORM_LOSS_WEIGHT > 0 else 'disabled')}"
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sampler.set_epoch(epoch)
        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            disable=rank != 0,
            mininterval=0.2,
        )
        for batch_index, batch in enumerate(progress):
            if warmup_steps and global_step < warmup_steps and not OPTIMIZER_NAME.startswith("Sched-Free"):
                scale = max(1, global_step + 1) / warmup_steps
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * scale
            batch = [item.to(device, non_blocking=True) for item in batch]
            phone, phone_lengths, pitch, pitchf, mel, mel_lengths, wave, _, sid = batch
            last_batch = batch_index + 1 == len(loader)
            step_now = (
                (batch_index + 1) % GRADIENT_ACCUMULATION == 0 or last_batch
            )
            group_start = (batch_index // GRADIENT_ACCUMULATION) * GRADIENT_ACCUMULATION
            accumulation_divisor = min(
                GRADIENT_ACCUMULATION, len(loader) - group_start
            )
            sync_context = (
                model.no_sync()
                if distributed and not step_now
                else nullcontext()
            )
            waveform_due = (
                pc_vocoder is not None
                and step_now
                and global_step % WAVEFORM_LOSS_INTERVAL == 0
            )
            conversion_due = (
                config.model.spk_embed_dim > 1
                and random.random()
                < getattr(config.train, "conversion_probability", 0.5)
            )
            with sync_context:
                loss_conversion = mel.new_zeros(())
                loss_converted_speaker = mel.new_zeros(())
                if conversion_due and branchwise_enabled:
                    module = model.module if hasattr(model, "module") else model
                    target_sid = _different_sids(
                        sid, config.model.spk_embed_dim
                    )
                    with autocast(
                        device_type=device.type,
                        enabled=use_amp,
                        dtype=torch.float16,
                    ):
                        (
                            converted_mel,
                            recovered,
                            source_content,
                            conversion_mask,
                        ) = module.conversion_cycle(
                            phone,
                            phone_lengths,
                            pitch,
                            pitchf,
                            target_sid,
                        )
                        loss_conversion = _masked_l1(
                            recovered, source_content, conversion_mask
                        )
                        converted_logits = module.mel_speaker_classifier(
                            converted_mel, conversion_mask
                        )
                        loss_converted_speaker = F.cross_entropy(
                            converted_logits.float(), target_sid
                        )
                        conversion_branch_total = (
                            loss_conversion
                            * getattr(config.train, "c_conversion", 1.0)
                            + loss_converted_speaker
                            * SPEAKER_CLASSIFICATION_WEIGHT
                        )
                    scaler.scale(
                        conversion_branch_total / accumulation_divisor
                    ).backward()
                    del (
                        converted_mel,
                        recovered,
                        source_content,
                        conversion_mask,
                        converted_logits,
                        conversion_branch_total,
                    )

                with autocast(
                    device_type=device.type,
                    enabled=use_amp,
                    dtype=torch.float16,
                ):
                    mel_hat, ids, x_mask, mel_mask, latent, speaker_logits = model(
                        phone, phone_lengths, pitch, pitchf, mel, mel_lengths, sid
                    )
                    z, z_p, z_p2, m_p, logs_p, _, logs_q = latent
                    target = slice_segments(mel, ids, mel_hat.size(-1), 3)
                    target_mask = slice_segments(
                        mel_mask, ids, mel_hat.size(-1), 3
                    )
                    loss_mel = _masked_l1(mel_hat, target, target_mask)
                    loss_delta = _masked_l1(
                        mel_hat[..., 1:] - mel_hat[..., :-1],
                        target[..., 1:] - target[..., :-1],
                        target_mask[..., 1:],
                    )
                    loss_kl = kl_loss_fb(
                        z_p,
                        logs_q,
                        m_p,
                        logs_p,
                        mel_mask,
                        z_p2,
                        free_bits=KL_FREE_BITS,
                    )
                    if KL_ANNEALING:
                        warmup = max(
                            1, optimizer_steps_per_epoch * KL_CYCLE_EPOCHS
                        )
                        kl_weight = min(1.0, global_step / warmup)
                    else:
                        kl_weight = 1.0
                    loss_source_speaker = F.cross_entropy(
                        speaker_logits[0].float(), sid
                    )
                    loss_target_speaker = F.cross_entropy(
                        speaker_logits[1].float(), sid
                    )
                    if conversion_due and not branchwise_enabled:
                        module = model.module if hasattr(model, "module") else model
                        target_sid = _different_sids(
                            sid, config.model.spk_embed_dim
                        )
                        converted_mel, recovered, source_content, conversion_mask = (
                            module.conversion_cycle(
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                target_sid,
                            )
                        )
                        loss_conversion = _masked_l1(
                            recovered, source_content, conversion_mask
                        )
                        converted_logits = module.mel_speaker_classifier(
                            converted_mel, conversion_mask
                        )
                        loss_converted_speaker = F.cross_entropy(
                            converted_logits.float(), target_sid
                        )
                    loss_waveform = mel_hat.new_zeros(())
                    if waveform_due and not branchwise_enabled:
                        loss_waveform = _waveform_objective(
                            pc_vocoder,
                            waveform_loss,
                            mel_hat,
                            ids,
                            pitchf,
                            wave,
                            config.data.hop_length,
                        )
                    total = (
                        loss_mel * config.train.c_mel
                        + loss_delta * getattr(config.train, "c_delta", 2.0)
                        + loss_kl * config.train.c_kl * kl_weight
                        + loss_source_speaker * CONTENT_ADVERSARIAL_WEIGHT
                        + loss_target_speaker * SPEAKER_CLASSIFICATION_WEIGHT
                        + (
                            loss_conversion
                            * getattr(config.train, "c_conversion", 1.0)
                            + loss_converted_speaker
                            * SPEAKER_CLASSIFICATION_WEIGHT
                            if not branchwise_enabled
                            else 0.0
                        )
                        + (
                            loss_waveform
                            * WAVEFORM_LOSS_WEIGHT
                            * accumulation_divisor
                            if not branchwise_enabled
                            else 0.0
                        )
                    )

                scaler.scale(total / accumulation_divisor).backward()
                if waveform_due and branchwise_enabled:
                    branch_count = min(
                        WAVEFORM_MICROBATCH_SIZE, phone.size(0)
                    )
                    selection = torch.randperm(
                        phone.size(0), device=device
                    )[:branch_count]
                    with autocast(
                        device_type=device.type,
                        enabled=use_amp,
                        dtype=torch.float16,
                    ):
                        branch_output = model(
                            phone.index_select(0, selection),
                            phone_lengths.index_select(0, selection),
                            pitch.index_select(0, selection),
                            pitchf.index_select(0, selection),
                            mel.index_select(0, selection),
                            mel_lengths.index_select(0, selection),
                            sid.index_select(0, selection),
                        )
                        branch_mel_hat, branch_ids = branch_output[:2]
                        loss_waveform = _waveform_objective(
                            pc_vocoder,
                            waveform_loss,
                            branch_mel_hat,
                            branch_ids,
                            pitchf.index_select(0, selection),
                            wave.index_select(0, selection),
                            config.data.hop_length,
                        )
                        del branch_output
                    scaler.scale(
                        loss_waveform
                        * WAVEFORM_LOSS_WEIGHT
                    ).backward()

            grad_norm = torch.zeros((), device=device)
            if step_now:
                scaler.unscale_(optimizer)
                if CLIP_SCHEDULE:
                    clip = CLIP_INITIAL if global_step < CLIP_STEPS else CLIP_FINAL
                    clip = clip if clip > 0 else math.inf
                else:
                    clip = math.inf
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                module = model.module if hasattr(model, "module") else model
                if (
                    scheduler is not None
                    and per_step_scheduler
                    and global_step >= warmup_steps
                ):
                    scheduler.step()
                global_step += 1
                if global_step - last_ema_step >= ema_update_interval:
                    ema.update(
                        module, elapsed_steps=global_step - last_ema_step
                    )
                    last_ema_step = global_step

            total_for_logging = total.detach().float()
            if branchwise_enabled:
                total_for_logging = (
                    total_for_logging
                    + loss_waveform.detach().float() * WAVEFORM_LOSS_WEIGHT
                    + loss_conversion.detach().float()
                    * getattr(config.train, "c_conversion", 1.0)
                    + loss_converted_speaker.detach().float()
                    * SPEAKER_CLASSIFICATION_WEIGHT
                )
            elif waveform_due:
                total_for_logging = (
                    total_for_logging
                    - loss_waveform.detach().float()
                    * WAVEFORM_LOSS_WEIGHT
                    * accumulation_divisor
                    + loss_waveform.detach().float() * WAVEFORM_LOSS_WEIGHT
                )
            batch_loss = total_for_logging.item()
            running += batch_loss
            if rank == 0:
                progress_status = {
                    "loss": f"{running / (batch_index + 1):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "step": global_step,
                }
                if device.type == "cuda":
                    progress_status["vram"] = (
                        f"{torch.cuda.memory_allocated(device) / 2**30:.2f}G"
                    )
                progress.set_postfix(progress_status, refresh=False)
            if rank == 0 and step_now and global_step % LOG_INTERVAL == 0:
                average = running / max(1, batch_index + 1)
                memory_text = ""
                if device.type == "cuda":
                    peak_allocated = (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    )
                    peak_reserved = (
                        torch.cuda.max_memory_reserved(device) / 2**30
                    )
                    memory_text = (
                        f" vram={peak_allocated:.2f}GiB/"
                        f"{peak_reserved:.2f}GiB"
                    )
                progress.write(
                    f"epoch={epoch} step={global_step} loss={average:.4f} "
                    f"mel={loss_mel.item():.4f} kl={loss_kl.item():.4f} "
                    f"wave={loss_waveform.item():.4f} "
                    f"conversion={loss_conversion.item():.4f}{memory_text}"
                )
                writer.add_scalar(
                    "loss/total", total_for_logging.item(), global_step
                )
                writer.add_scalar("loss/mel", loss_mel.item(), global_step)
                writer.add_scalar("loss/kl", loss_kl.item(), global_step)
                writer.add_scalar(
                    "loss/conversion_cycle", loss_conversion.item(), global_step
                )
                writer.add_scalar(
                    "loss/waveform_stft", loss_waveform.item(), global_step
                )
                writer.add_scalar(
                    "loss/content_speaker_adversarial",
                    loss_source_speaker.item(),
                    global_step,
                )
                writer.add_scalar(
                    "loss/target_speaker",
                    loss_target_speaker.item(),
                    global_step,
                )
                writer.add_scalar(
                    "loss/converted_target_speaker",
                    loss_converted_speaker.item(),
                    global_step,
                )
                writer.add_scalar("train/kl_weight", kl_weight, global_step)
                if device.type == "cuda":
                    writer.add_scalar(
                        "memory/peak_allocated_gib",
                        peak_allocated,
                        global_step,
                    )
                    writer.add_scalar(
                        "memory/peak_reserved_gib",
                        peak_reserved,
                        global_step,
                    )
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                writer.add_scalar(
                    "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )

        if (
            scheduler is not None
            and not per_step_scheduler
            and epoch >= max(start_epoch, WARMUP_EPOCHS)
        ):
            scheduler.step()

        module = model.module if hasattr(model, "module") else model
        if rank == 0 and device.type == "cuda":
            print(
                f"epoch={epoch} peak_vram_allocated="
                f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB "
                f"peak_vram_reserved="
                f"{torch.cuda.max_memory_reserved(device) / 2**30:.2f}GiB"
            )
        if global_step > last_ema_step:
            ema.update(module, elapsed_steps=global_step - last_ema_step)
            last_ema_step = global_step
        epoch_loss = running / max(1, len(loader))
        validation = None
        if validation_loader is not None:
            if hasattr(optimizer, "eval"):
                optimizer.eval()
            validation_vocoder = pc_vocoder
            lazy_validation_vocoder = False
            if VOCODER_VALIDATION_ONLY:
                if rank == 0:
                    print(
                        "[Mel-VITS] Loading pc-NSF for validation only."
                    )
                validation_vocoder = PCNSFHiFiGAN.from_export(
                    ROOT / config.vocoder.checkpoint,
                    ROOT / config.vocoder.config,
                    map_location=device,
                ).to(device)
                lazy_validation_vocoder = True
            with ema.apply(module):
                validation = _validate(
                    model,
                    validation_loader,
                    device,
                    use_amp,
                    pc_vocoder=validation_vocoder,
                    waveform_loss=waveform_loss,
                    hop_length=config.data.hop_length,
                )
            if lazy_validation_vocoder:
                del validation_vocoder
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if rank == 0:
                    print(
                        "[Mel-VITS] Unloaded validation-only pc-NSF."
                    )
            if hasattr(optimizer, "train"):
                optimizer.train()
            if rank == 0:
                writer.add_scalar(
                    "validation/total", validation["total"], global_step
                )
                writer.add_scalar(
                    "validation/mel", validation["mel"], global_step
                )
                writer.add_scalar(
                    "validation/kl", validation["kl"], global_step
                )
                writer.add_scalar(
                    "validation/waveform_stft",
                    validation["waveform"],
                    global_step,
                )
                print(
                    f"epoch={epoch} validation={validation['total']:.4f} "
                    f"mel={validation['mel']:.4f} kl={validation['kl']:.4f} "
                    f"wave={validation['waveform']:.4f}"
                )
        selection_loss = validation["total"] if validation else epoch_loss
        is_best = selection_loss < best_loss
        if is_best:
            best_loss = selection_loss
        should_save = (
            epoch % SAVE_EVERY == 0
            or epoch == EPOCHS
            or (SAVE_BEST and is_best)
        )
        if rank == 0 and should_save:
            path = EXPERIMENT_DIR / (
                "G_latest.pth" if SAVE_LATEST else f"G_{global_step}.pth"
            )
            save_checkpoint(
                model,
                optimizer,
                optimizer.param_groups[0]["lr"],
                epoch,
                str(path),
                scaler,
                extra={
                    "ema": ema.state_dict(),
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "last_ema_step": last_ema_step,
                    "scheduler": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                },
            )
            if SAVE_BEST and is_best:
                shutil.copy2(path, EXPERIMENT_DIR / "G_best.pth")
            with ema.apply(module):
                _save_small_model(model, config, epoch, global_step)

    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    if SAMPLE_RATE != 44100:
        raise ValueError("The pc-NSF-HiFiGAN architecture supports only 44100 Hz")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    if CLEANUP:
        _cleanup_old_checkpoints()
    gpu_ids = (
        [int(item) for item in GPU_IDS.split("-") if item.strip().isdigit()]
        if torch.cuda.is_available() and GPU_IDS != "-"
        else []
    )
    world_size = max(1, len(gpu_ids))
    if world_size > 1:
        mp.spawn(_train_worker, args=(world_size, gpu_ids), nprocs=world_size)
    else:
        _train_worker(0, 1, gpu_ids)


if __name__ == "__main__":
    if _LEGACY_ARCH == "Hybrid-FSQ":
        from rvc.train.train_hybrid import main as hybrid_main

        hybrid_main()
    elif _LEGACY_ARCH == "Stochastic-Residual-Conformer-GAN":
        from rvc.train.train_conformer_gan import main as conformer_gan_main

        conformer_gan_main()
    elif _LEGACY_ARCH == "Raw-NSF-Waveform-GAN":
        from rvc.train.train_raw_nsf_gan import main as raw_nsf_main

        raw_nsf_main()
    else:
        main()
