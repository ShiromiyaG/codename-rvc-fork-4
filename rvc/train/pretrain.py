"""
Pretrain Pipeline for Codename-RVC-Fork — 3-Phase Training

Phase 1: Isolated vocoder (decoder) pretraining
    - Input: mel-spectrogram + F0 → waveform
    - Losses: spectral + adversarial + feature matching (no KL)
    - Saves decoder-only + full G/D checkpoints

Phase 2: Full VITS pretrain (base SR, default 48kHz)
    - Loads Phase 1 decoder weights into net_g.dec (frozen initially)
    - Gradual decoder unfreeze after --decoder_freeze_steps
    - KL annealing: linear 0→1 over --kl_anneal_steps
    - Free bits: kl_loss = max(kl_per_dim, --kl_free_bits)

Phase 3: Sample rate adaptation (32k / 40k from 48k checkpoint)
    - Loads Phase 2 checkpoint, overrides sample rate
    - Continues training with same losses, shorter schedule

Usage:
    python rvc/train/pretrain.py --phase 1 --vocoder ChouwaGAN --sample_rate 48000 ...
    python rvc/train/pretrain.py --phase 2 --vocoder ChouwaGAN --phase1_ckpt_g <path> ...
    python rvc/train/pretrain.py --phase 3 --vocoder ChouwaGAN --phase2_ckpt_g <path> --sample_rate 40000 ...
"""

import os
import sys
import signal
import datetime
import glob
import argparse
import json
import math
import re

pid_data = {"process_pids": []}
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from collections import deque
from itertools import islice
from random import randint, shuffle
from time import time as ttime

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.multiprocessing as mp
import auraloss

now_dir = os.getcwd()
sys.path.append(now_dir)

from utils import (
    HParams,
    plot_spectrogram_to_numpy,
    summarize,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    load_wav_to_torch,
    load_config_from_json,
    mel_spec_similarity,
    flush_writer,
    block_tensorboard_flush_on_exit,
    wave_to_mel,
    small_model_naming,
    old_session_cleanup,
    train_loader_safety,
    verify_spk_dim,
    early_stopper,
)
from losses import (
    discriminator_loss,
    generator_loss,
    discriminator_tprls_loss,
    generator_tprls_loss,
    HingeAdversarialLoss,
    SoftHingeAdversarialLoss,
    LeCamRegularization,
    feature_loss,
    kl_loss,
    kl_loss_floored,
)
from mel_processing import (
    spec_to_mel_torch,
    MultiScaleMelSpectrogramLoss,
    mel_spectrogram_torch,
)
from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.train.utils import replace_keys_in_dict, _unwrap_model, _strip_compile_prefix

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="RVC Pretrain Pipeline (3-Phase)")

    # Phase
    p.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                   help="Training phase: 1=decoder-only, 2=full-VITS, 3=SR-adapt")

    # Model identity
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--vocoder", type=str, default="ChouwaGAN",
                   choices=["ChouwaGAN", "HiFi-GAN", "RefineGAN", "RingFormer_v1",
                            "RingFormer_v2", "APEX-GAN"])
    p.add_argument("--architecture", type=str, default="Fork")
    p.add_argument("--sample_rate", type=int, default=48000)

    # Training basics
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--total_epochs", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="AdamW",
                   choices=["AdamW", "RAdam"])

    # Precision & performance
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--use_tf32", action="store_true", default=True)
    p.add_argument("--use_checkpointing", action="store_true", default=False)
    p.add_argument("--use_torch_compile", action="store_true", default=False)

    # Loss config
    p.add_argument("--spectral_loss", type=str, default="L1 Mel Loss",
                   choices=["L1 Mel Loss", "Multi-Scale Mel Loss", "Multi-Res STFT Loss"])
    p.add_argument("--adversarial_loss", type=str, default="lsgan",
                   choices=["lsgan", "hinge", "soft_hinge", "tprls"])

    # Phase 2 specific: KL annealing
    p.add_argument("--kl_anneal_steps", type=int, default=50000,
                   help="Steps over which KL weight linearly ramps 0→1")
    p.add_argument("--kl_free_bits", type=float, default=0.25,
                   help="Free bits threshold per KL dimension (nats)")

    # Phase 2 specific: decoder freeze
    p.add_argument("--decoder_freeze_steps", type=int, default=10000,
                   help="Steps to keep decoder frozen at start of Phase 2")

    # Checkpoint loading
    p.add_argument("--phase1_ckpt_g", type=str, default="",
                   help="Phase 1 generator checkpoint for Phase 2 init")
    p.add_argument("--phase1_ckpt_d", type=str, default="",
                   help="Phase 1 discriminator checkpoint for Phase 2 init")
    p.add_argument("--phase2_ckpt_g", type=str, default="",
                   help="Phase 2 generator checkpoint for Phase 3 init")
    p.add_argument("--phase2_ckpt_d", type=str, default="",
                   help="Phase 2 discriminator checkpoint for Phase 3 init")

    # Pretrain from existing weights
    p.add_argument("--pretrain_g", type=str, default="",
                   help="Optional pretrained G weights to init from")
    p.add_argument("--pretrain_d", type=str, default="",
                   help="Optional pretrained D weights to init from")

    # Grad clipping
    p.add_argument("--grad_clip_g", type=float, default=1000.0)
    p.add_argument("--grad_clip_d", type=float, default=1000.0)

    # Logging
    p.add_argument("--rolling_loss_steps", type=int, default=50)
    p.add_argument("--preview_interval", type=int, default=500,
                   help="Steps between generating audio previews in TensorBoard")

    # Misc
    p.add_argument("--save_only_latest", action="store_true", default=True)
    p.add_argument("--no_save_only_latest", dest="save_only_latest", action="store_false")
    p.add_argument("--cleanup", action="store_true", default=False)
    p.add_argument("--spk_embed_dim", type=int, default=0,
                   help="Number of speakers. 0 = auto-detect from model_info.json")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

class EarlyStopSignalHandler:
    def __init__(self):
        self.stop_triggered = False
        signal.signal(signal.SIGINT, self._handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._handler)

    def _handler(self, signum, frame):
        self.stop_triggered = True
        print(f"\n[PRETRAIN] Early stop signal received! Saving and exiting...")


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now = ttime()
        elapsed = now - self.last_time
        self.last_time = now
        return f"Time: {datetime.datetime.now().strftime('%H:%M:%S')} | Per epoch: {str(datetime.timedelta(seconds=int(elapsed)))}"


def eval_infer(net_g, reference):
    net_g.eval()
    with torch.no_grad():
        model = net_g.module if hasattr(net_g, "module") else net_g
        o, *_ = model.infer(*reference)
    net_g.train()
    return o


def get_vocoder_config_dir(vocoder):
    mapping = {
        "ChouwaGAN": "chouwa_gan",
        "HiFi-GAN": "hifi_refine",
        "RefineGAN": "hifi_refine",
        "RingFormer_v1": "ringformer_v1",
        "RingFormer_v2": "ringformer_v2",
        "APEX-GAN": "apex_gan",
    }
    return mapping.get(vocoder, "hifi_refine")


def kl_weight_linear(global_step, anneal_steps):
    """Linear KL annealing: 0 → 1 over anneal_steps."""
    if anneal_steps <= 0:
        return 1.0
    return min(1.0, global_step / anneal_steps)


# ═══════════════════════════════════════════════════════════════════
# Model Factories
# ═══════════════════════════════════════════════════════════════════

def get_g_model(config, sample_rate, vocoder, use_checkpointing):
    from rvc.lib.algorithm.synthesizers import Synthesizer
    return Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=True,
        sr=sample_rate,
        vocoder=vocoder,
        checkpointing=use_checkpointing,
    )


def get_d_model(config, vocoder, use_checkpointing, sample_rate):
    if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined
        return MPD_MSD_MRD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            **dict(config.mrd) if hasattr(config, 'mrd') else {}
        )
    elif vocoder == "APEX-GAN":
        from rvc.lib.algorithm.discriminators.multi import CoMBD_SBD_UnivHD_Combined
        return CoMBD_SBD_UnivHD_Combined(
            sample_rate=sample_rate,
            segment_size_samples=config.train.segment_size,
            use_spectral_norm=config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
        )
    elif vocoder == "RefineGAN":
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined_RefineGan
        return MPD_MSD_MRD_Combined_RefineGan(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
        )
    elif vocoder == "ChouwaGAN":
        from rvc.lib.algorithm.discriminators.multi import ChouwaGAN_Combined
        return ChouwaGAN_Combined(
            sample_rate=sample_rate,
            use_spectral_norm=config.model.use_spectral_norm,
            use_san=True,
            use_checkpointing=use_checkpointing,
        )
    else:
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined
        return MPD_MSD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
        )


def get_decoder_only(config, vocoder, sample_rate, use_checkpointing):
    """Instantiate just the vocoder decoder for Phase 1."""
    if vocoder == "ChouwaGAN":
        from rvc.lib.algorithm.generators import ChouwaGANGenerator
        return ChouwaGANGenerator(
            initial_channel=config.model.inter_channels,
            upsample_rates=config.model.upsample_rates,
            upsample_initial_channel=config.model.upsample_initial_channel,
            upsample_kernel_sizes=config.model.upsample_kernel_sizes,
            resblock_kernel_sizes=config.model.resblock_kernel_sizes,
            resblock_dilation_sizes=config.model.resblock_dilation_sizes,
            gin_channels=config.model.gin_channels,
            sr=sample_rate,
            checkpointing=use_checkpointing,
        )
    elif vocoder == "RefineGAN":
        from rvc.lib.algorithm.generators import RefineGANGenerator
        return RefineGANGenerator(
            sample_rate=sample_rate,
            downsample_rates=config.model.upsample_rates[::-1],
            upsample_rates=config.model.upsample_rates,
            start_channels=16,
            num_mels=config.model.inter_channels,
            checkpointing=use_checkpointing,
        )
    elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        from rvc.lib.algorithm.generators import RingFormerGenerator
        return RingFormerGenerator(
            initial_channel=config.model.inter_channels,
            resblock_kernel_sizes=config.model.resblock_kernel_sizes,
            resblock_dilation_sizes=config.model.resblock_dilation_sizes,
            upsample_rates=config.model.upsample_rates,
            upsample_initial_channel=config.model.upsample_initial_channel,
            upsample_kernel_sizes=config.model.upsample_kernel_sizes,
            gen_istft_n_fft=getattr(config.model, 'gen_istft_n_fft', 120),
            gen_istft_hop_size=getattr(config.model, 'gen_istft_hop_size', 30),
            gin_channels=config.model.gin_channels,
            sr=sample_rate,
            checkpointing=use_checkpointing,
        )
    elif vocoder == "APEX-GAN":
        from rvc.lib.algorithm.generators import APEX_GAN_Generator
        return APEX_GAN_Generator(
            config.model.inter_channels,
            config.model.resblock_kernel_sizes,
            config.model.resblock_dilation_sizes,
            config.model.upsample_rates,
            config.model.upsample_initial_channel,
            config.model.upsample_kernel_sizes,
            gin_channels=config.model.gin_channels,
            sr=sample_rate,
        )
    else:  # HiFi-GAN
        from rvc.lib.algorithm.generators import HiFiGANNSFGenerator
        return HiFiGANNSFGenerator(
            config.model.inter_channels,
            config.model.resblock_kernel_sizes,
            config.model.resblock_dilation_sizes,
            config.model.upsample_rates,
            config.model.upsample_initial_channel,
            config.model.upsample_kernel_sizes,
            gin_channels=config.model.gin_channels,
            sr=sample_rate,
            checkpointing=use_checkpointing,
        )


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Decoder-Only Pretraining
# ═══════════════════════════════════════════════════════════════════

def run_phase1(rank, n_gpus, args, config, device, device_id):
    """
    Phase 1: Train only the vocoder (decoder) with discriminators.
    Input: spec → Conv1d projection → z → decoder(z, f0, g) → waveform
    Uses the full VITS data pipeline but only trains decoder + disc.
    """
    global_step = 0
    stopper = EarlyStopSignalHandler()

    experiment_dir = os.path.join(now_dir, "logs", args.model_name)
    config_save_path = os.path.join(experiment_dir, "config.json")

    # Setup distributed
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # Dataloaders
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )
    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    train_sampler = DistributedBucketSampler(
        train_dataset, args.batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset, num_workers=4, shuffle=False, pin_memory=True,
        collate_fn=TextAudioCollateMultiNSFsid(),
        batch_sampler=train_sampler,
        persistent_workers=True, prefetch_factor=8,
    )
    train_loader_safety(train_loader)

    # Models
    decoder = get_decoder_only(config, args.vocoder, args.sample_rate, args.use_checkpointing)

    # Auto-detect speaker count from model_info.json (written during preprocessing)
    model_info_path = os.path.join(experiment_dir, "model_info.json")
    spk_dim = args.spk_embed_dim if args.spk_embed_dim > 0 else config.model.spk_embed_dim
    if rank == 0:
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
                spk_dim = model_info["speakers_id"]
        except Exception as e:
            print(f"Could not read speakers_id from model_info.json: {e}. Using {spk_dim}.")
        print(f"    ██████  Phase 1: Initializing decoder with {spk_dim} speakers.")

    # Broadcast spk_dim to all ranks
    if n_gpus > 1:
        spk_dim_tensor = torch.tensor([spk_dim], device=device)
        dist.broadcast(spk_dim_tensor, src=0)
        spk_dim = int(spk_dim_tensor.item())

    emb_g = nn.Embedding(spk_dim, config.model.gin_channels)
    net_d = get_d_model(config, args.vocoder, args.use_checkpointing, args.sample_rate)

    # Wrap decoder + emb_g + projection into a module for clean ckpt saving
    spec_channels = config.data.filter_length // 2 + 1
    inter_channels = config.model.inter_channels

    class DecoderWrapper(nn.Module):
        def __init__(self, dec, emb_g, spec_channels, inter_channels):
            super().__init__()
            self.dec = dec
            self.emb_g = emb_g
            # Project spec (1025-dim) → inter_channels (192-dim) for decoder input
            self.spec_proj = nn.Conv1d(spec_channels, inter_channels, 1)

        def forward(self, spec, f0, sid):
            g = self.emb_g(sid).unsqueeze(-1)
            z = self.spec_proj(spec)
            return self.dec(z, f0, g=g)

    net_g_wrapper = DecoderWrapper(decoder, emb_g, spec_channels, inter_channels)

    # Resume from checkpoint if available
    def get_highest_checkpoint(prefix, directory):
        pattern = re.compile(rf"^{prefix}(\d+)\.pth$")
        files = []
        for f in os.listdir(directory):
            match = pattern.match(f)
            if match:
                files.append((int(match.group(1)), os.path.join(directory, f)))
        return sorted(files, key=lambda x: x[0], reverse=True)[0][1] if files else None

    # Move to device
    if device.type == "cuda":
        net_g_wrapper = net_g_wrapper.to(device_id)
        net_d = net_d.to(device_id)
    else:
        net_g_wrapper = net_g_wrapper.to(device)
        net_d = net_d.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g_wrapper = DDP(net_g_wrapper, device_ids=[device_id])
        net_d = DDP(net_d, device_ids=[device_id])

    if args.use_torch_compile and sys.platform == "linux":
        try:
            net_g_wrapper = torch.compile(net_g_wrapper, mode="max-autotune-no-cudagraphs")
            net_d = torch.compile(net_d, mode="max-autotune-no-cudagraphs")
            if rank == 0:
                print("    ██████  torch.compile enabled for G and D (max-autotune-no-cudagraphs)")
        except Exception as e:
            if rank == 0:
                print(f"    ██████  torch.compile failed, falling back to eager mode: {e}")

    # Optimizers
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g_wrapper.parameters()),
        lr=args.lr_g, betas=(0.8, 0.99), eps=1e-9,
    )
    optim_d = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_d.parameters()),
        lr=args.lr_d, betas=(0.8, 0.99), eps=1e-9,
    )

    epoch_str = 1
    gradscaler_dict = {}

    # Try resume
    try:
        g_ckpt = get_highest_checkpoint("G_phase1_", experiment_dir)
        d_ckpt = get_highest_checkpoint("D_phase1_", experiment_dir)
        if g_ckpt and d_ckpt:
            _, _, _, _, gradscaler_dict = load_checkpoint(g_ckpt, net_g_wrapper, optim_g, strict_load=True)
            load_checkpoint(d_ckpt, net_d, optim_d, strict_load=True)
            global_step = int(os.path.basename(g_ckpt).split("_")[-1].split(".")[0])
            epoch_str = (global_step // len(train_loader)) + 1
            if rank == 0:
                print(f"[Phase 1] Resuming from step {global_step}, epoch {epoch_str - 1}")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, Exception):
        if rank == 0:
            print("[Phase 1] Starting from scratch")

    # Spectral loss
    if args.spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = torch.nn.L1Loss()
    elif args.spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(sample_rate=args.sample_rate)
    elif args.spectral_loss == "Multi-Res STFT Loss":
        fn_spectral_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 4096], hop_sizes=[256, 512, 1024],
            win_lengths=[1024, 2048, 4096], window="hann_window",
            scale="mel", n_bins=128, sample_rate=args.sample_rate,
            perceptual_weighting=True, device=device,
        )

    # Adversarial loss
    if args.adversarial_loss == "hinge":
        fn_hinge = HingeAdversarialLoss()
    elif args.adversarial_loss == "soft_hinge":
        fn_hinge = SoftHingeAdversarialLoss()
    else:
        fn_hinge = None

    fn_lecam = LeCamRegularization(decay=0.9999).to(device) if args.vocoder == "ChouwaGAN" else None

    # AMP
    train_dtype = torch.float16 if args.fp16 else torch.float32
    use_amp = args.fp16 and device.type == "cuda"
    gradscaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))
    if gradscaler_dict:
        gradscaler.load_state_dict(gradscaler_dict)

    # Tensorboard
    if rank == 0:
        writer = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval_pretrain_phase1"),
            flush_secs=86400, purge_step=global_step + 1,
        )
        block_tensorboard_flush_on_exit(writer)
        print(f"[Phase 1] Decoder-only pretraining: {args.vocoder}")
        print(f"[Phase 1] Epochs: {args.total_epochs}, Batch: {args.batch_size}, LR: {args.lr_g}/{args.lr_d}")

    epoch_recorder = EpochRecorder()
    avg_rolling = {
        "loss_disc": deque(maxlen=args.rolling_loss_steps),
        "loss_adv": deque(maxlen=args.rolling_loss_steps),
        "loss_mel": deque(maxlen=args.rolling_loss_steps),
        "loss_fm": deque(maxlen=args.rolling_loss_steps),
        "loss_total": deque(maxlen=args.rolling_loss_steps),
    }

    for epoch in range(epoch_str, args.total_epochs + 1):
        train_sampler.set_epoch(epoch)
        net_g_wrapper.train()
        net_d.train()

        current_epoch_start = (epoch - 1) * len(train_loader)
        start_batch = max(0, global_step - current_epoch_start)
        remaining = len(train_loader) - start_batch
        data_iter = islice(enumerate(train_loader), remaining)

        from tqdm import tqdm
        with tqdm(total=len(train_loader), leave=False, initial=start_batch,
                  desc=f"Phase1 E{epoch}") as pbar:
            for batch_idx, info in data_iter:
                global_step += 1

                if device.type == "cuda":
                    info = [t.cuda(device_id, non_blocking=True) for t in info]
                else:
                    info = [t.to(device) for t in info]

                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid = info

                # Posterior encoder simulation: use spec directly as "z"
                # We generate mel from spec, then feed to decoder
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    # Random segment slicing on spec
                    seg_size = config.train.segment_size // config.data.hop_length
                    spec_slice, ids_slice = commons.rand_slice_segments(spec, spec_lengths, seg_size)
                    pitchf_slice = commons.slice_segments(pitchf, ids_slice, seg_size, dim=2)

                    # Decoder forward: spec → proj → z → waveform
                    g_model = net_g_wrapper.module if hasattr(net_g_wrapper, "module") else net_g_wrapper
                    y_hat = g_model(spec_slice, pitchf_slice, sid)

                    # Slice ground truth waveform
                    y_sliced = commons.slice_segments(
                        y, ids_slice * config.data.hop_length,
                        config.train.segment_size, dim=3,
                    )

                # ─── Discriminator step ───
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat.detach())

                with autocast(device_type="cuda", enabled=False):
                    if args.adversarial_loss == "lsgan":
                        loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    elif args.adversarial_loss == "tprls":
                        loss_disc = discriminator_tprls_loss(y_d_hat_r, y_d_hat_g)
                    elif args.adversarial_loss in ("hinge", "soft_hinge"):
                        loss_fake, loss_real = fn_hinge(y_d_hat_g, y_d_hat_r)
                        loss_disc = loss_fake + loss_real

                    if fn_lecam is not None:
                        loss_disc = loss_disc + 0.2 * fn_lecam(y_d_hat_r, y_d_hat_g)
                        fn_lecam.update_ema(y_d_hat_r, y_d_hat_g)

                optim_d.zero_grad(set_to_none=True)
                if train_dtype == torch.float16:
                    gradscaler.scale(loss_disc).backward()
                    gradscaler.unscale_(optim_d)
                    clip_grad_norm_(net_d.parameters(), max_norm=args.grad_clip_d)
                    gradscaler.step(optim_d)
                else:
                    loss_disc.backward()
                    clip_grad_norm_(net_d.parameters(), max_norm=args.grad_clip_d)
                    optim_d.step()

                # ─── Generator step ───
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    _, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)

                with autocast(device_type="cuda", enabled=False):
                    # Spectral loss
                    if args.spectral_loss == "L1 Mel Loss":
                        y_mel = wave_to_mel(config, y_sliced, half=train_dtype)
                        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)
                        loss_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                    elif args.spectral_loss == "Multi-Scale Mel Loss":
                        loss_mel = fn_spectral_loss(y_sliced, y_hat) * config.train.c_mel / 3.0
                    elif args.spectral_loss == "Multi-Res STFT Loss":
                        loss_mel = fn_spectral_loss(y_hat.float(), y_sliced.float()) * 21.0

                    # Feature matching
                    loss_fm = feature_loss(fmap_r, fmap_g)

                    # Generator adv loss
                    if args.adversarial_loss == "lsgan":
                        loss_adv = generator_loss(y_d_hat_g)
                    elif args.adversarial_loss == "tprls":
                        y_d_hat_r_det = [i.detach() for i in y_d_hat_r]
                        loss_adv = generator_tprls_loss(y_d_hat_r_det, y_d_hat_g)
                    elif args.adversarial_loss in ("hinge", "soft_hinge"):
                        loss_adv = fn_hinge(y_d_hat_g)

                    # Normalize by disc count for ChouwaGAN
                    if args.vocoder == "ChouwaGAN":
                        n_disc = len(y_d_hat_g)
                        loss_adv = loss_adv / n_disc
                        loss_fm = loss_fm / n_disc

                    loss_gen_total = loss_adv + loss_fm + loss_mel

                optim_g.zero_grad(set_to_none=True)
                if train_dtype == torch.float16:
                    gradscaler.scale(loss_gen_total).backward()
                    gradscaler.unscale_(optim_g)
                    clip_grad_norm_(net_g_wrapper.parameters(), max_norm=args.grad_clip_g)
                    gradscaler.step(optim_g)
                    gradscaler.update()
                else:
                    loss_gen_total.backward()
                    clip_grad_norm_(net_g_wrapper.parameters(), max_norm=args.grad_clip_g)
                    optim_g.step()

                # Rolling loss tracking
                avg_rolling["loss_disc"].append(loss_disc.detach())
                avg_rolling["loss_adv"].append(loss_adv.detach())
                avg_rolling["loss_mel"].append(loss_mel.detach())
                avg_rolling["loss_fm"].append(loss_fm.detach())
                avg_rolling["loss_total"].append(loss_gen_total.detach())

                if rank == 0 and global_step % args.rolling_loss_steps == 0:
                    scalars = {}
                    for key, queue in avg_rolling.items():
                        if len(queue) > 0:
                            val = torch.stack(list(queue)).mean().item()
                            scalars[f"phase1_rolling/{key}"] = val
                    scalars["phase1/lr_g"] = optim_g.param_groups[0]["lr"]
                    scalars["phase1/lr_d"] = optim_d.param_groups[0]["lr"]
                    summarize(writer=writer, global_step=global_step, scalars=scalars)
                    flush_writer(writer, rank)

                pbar.update(1)

                # Early stop check
                if stopper.stop_triggered:
                    if rank == 0:
                        _save_phase1_checkpoint(
                            net_g_wrapper, net_d, optim_g, optim_d,
                            args, config, epoch, global_step, experiment_dir, gradscaler,
                        )
                    if n_gpus > 1:
                        dist.barrier()
                    return

        # End of epoch
        if n_gpus > 1 and device.type == "cuda":
            dist.barrier()
        torch.cuda.empty_cache()

        if rank == 0:
            record = f"[Phase 1] {args.model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
            print(record)

            if epoch % args.save_every == 0 or epoch == args.total_epochs:
                _save_phase1_checkpoint(
                    net_g_wrapper, net_d, optim_g, optim_d,
                    args, config, epoch, global_step, experiment_dir, gradscaler,
                )

            if epoch == args.total_epochs:
                print(f"[Phase 1] Training complete! step={global_step}")
                writer.flush()
                writer.close()
                os._exit(0)


def _save_phase1_checkpoint(net_g_wrapper, net_d, optim_g, optim_d,
                            args, config, epoch, global_step, experiment_dir, gradscaler):
    """Save Phase 1 checkpoints."""
    g_path = os.path.join(experiment_dir, f"G_phase1_{global_step}.pth")
    d_path = os.path.join(experiment_dir, f"D_phase1_{global_step}.pth")

    if args.save_only_latest:
        for pattern in ["G_phase1_*.pth", "D_phase1_*.pth"]:
            for f in glob.glob(os.path.join(experiment_dir, pattern)):
                try:
                    os.remove(f)
                except:
                    pass

    save_checkpoint(net_g_wrapper, optim_g, args.lr_g, global_step, g_path, gradscaler)
    save_checkpoint(net_d, optim_d, args.lr_d, global_step, d_path, gradscaler)

    # Also save decoder-only weights for easy loading in Phase 2
    dec_model = _unwrap_model(net_g_wrapper)
    dec_only_path = os.path.join(experiment_dir, "decoder_phase1.pth")
    torch.save({
        "dec": _strip_compile_prefix(dec_model.dec.state_dict()),
        "emb_g": _strip_compile_prefix(dec_model.emb_g.state_dict()),
        "spec_proj": _strip_compile_prefix(dec_model.spec_proj.state_dict()),
    }, dec_only_path)
    print(f"[Phase 1] Saved decoder-only weights to {dec_only_path}")


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Full VITS Pretraining
# ═══════════════════════════════════════════════════════════════════

def run_phase2(rank, n_gpus, args, config, device, device_id):
    """
    Phase 2: Full VITS pretrain with:
    - Phase 1 decoder weights loaded + frozen initially
    - KL annealing (linear 0→1)
    - Free bits KL loss
    """
    global_step = 0
    stopper = EarlyStopSignalHandler()

    experiment_dir = os.path.join(now_dir, "logs", args.model_name)

    # Distributed setup
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # Dataloaders
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )
    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    train_sampler = DistributedBucketSampler(
        train_dataset, args.batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset, num_workers=4, shuffle=False, pin_memory=True,
        collate_fn=TextAudioCollateMultiNSFsid(),
        batch_sampler=train_sampler,
        persistent_workers=True, prefetch_factor=8,
    )
    train_loader_safety(train_loader)

    # Spk dim
    model_info_path = os.path.join(experiment_dir, "model_info.json")
    spk_dim = verify_spk_dim(config, model_info_path, experiment_dir, latest_checkpoint_path, rank, args.pretrain_g)
    config.model.spk_embed_dim = spk_dim

    # Models
    net_g = get_g_model(config, args.sample_rate, args.vocoder, args.use_checkpointing)
    net_d = get_d_model(config, args.vocoder, args.use_checkpointing, args.sample_rate)

    # ─── Load Phase 1 decoder weights ───
    decoder_frozen = False
    decoder_was_frozen = False

    phase1_dec_path = os.path.join(experiment_dir, "decoder_phase1.pth")
    if args.phase1_ckpt_g and os.path.isfile(args.phase1_ckpt_g):
        phase1_dec_path = args.phase1_ckpt_g

    if os.path.isfile(phase1_dec_path):
        if rank == 0:
            print(f"[Phase 2] Loading Phase 1 decoder from: {phase1_dec_path}")
        ckpt = torch.load(phase1_dec_path, map_location="cpu", weights_only=True)

        if "dec" in ckpt:
            net_g.dec.load_state_dict(ckpt["dec"], strict=False)
            if "emb_g" in ckpt:
                net_g.emb_g.load_state_dict(ckpt["emb_g"], strict=False)
        elif "model" in ckpt:
            # Full G_phase1 checkpoint — extract dec and emb_g keys
            state = _strip_compile_prefix(ckpt["model"])
            dec_state = {k.replace("dec.", ""): v for k, v in state.items() if k.startswith("dec.")}
            emb_state = {k.replace("emb_g.", ""): v for k, v in state.items() if k.startswith("emb_g.")}
            if dec_state:
                net_g.dec.load_state_dict(dec_state, strict=False)
            if emb_state:
                net_g.emb_g.load_state_dict(emb_state, strict=False)

        # Freeze decoder
        if args.decoder_freeze_steps > 0:
            for p in net_g.dec.parameters():
                p.requires_grad = False
            decoder_frozen = True
            decoder_was_frozen = True
            if rank == 0:
                print(f"[Phase 2] Decoder frozen for {args.decoder_freeze_steps} steps")
    else:
        if rank == 0:
            print("[Phase 2] No Phase 1 checkpoint found, training from scratch")

    # Load Phase 1 discriminator if available
    if args.phase1_ckpt_d and os.path.isfile(args.phase1_ckpt_d):
        if rank == 0:
            print(f"[Phase 2] Loading Phase 1 discriminator: {args.phase1_ckpt_d}")
        ckpt_d = torch.load(args.phase1_ckpt_d, map_location="cpu", weights_only=True)
        state_d = ckpt_d["model"] if "model" in ckpt_d else ckpt_d
        state_d = _strip_compile_prefix(state_d)
        net_d.load_state_dict(state_d, strict=False)

    # Move to device
    if device.type == "cuda":
        net_g = net_g.to(device_id)
        net_d = net_d.to(device_id)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id], find_unused_parameters=decoder_frozen)
        net_d = DDP(net_d, device_ids=[device_id])

    if args.use_torch_compile and sys.platform == "linux":
        try:
            net_g = torch.compile(net_g, mode="max-autotune-no-cudagraphs")
            net_d = torch.compile(net_d, mode="max-autotune-no-cudagraphs")
            if rank == 0:
                print("    ██████  torch.compile enabled for G and D (max-autotune-no-cudagraphs)")
        except Exception as e:
            if rank == 0:
                print(f"    ██████  torch.compile failed, falling back to eager mode: {e}")

    # Optimizers
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        lr=args.lr_g, betas=(0.8, 0.99), eps=1e-9,
    )
    optim_d = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_d.parameters()),
        lr=args.lr_d, betas=(0.8, 0.99), eps=1e-9,
    )

    epoch_str = 1
    gradscaler_dict = {}

    # Try resume
    def get_highest_checkpoint(prefix, directory):
        pattern = re.compile(rf"^{prefix}(\d+)\.pth$")
        files = []
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                match = pattern.match(f)
                if match:
                    files.append((int(match.group(1)), os.path.join(directory, f)))
        return sorted(files, key=lambda x: x[0], reverse=True)[0][1] if files else None

    try:
        g_ckpt = get_highest_checkpoint("G_phase2_", experiment_dir)
        d_ckpt = get_highest_checkpoint("D_phase2_", experiment_dir)
        if g_ckpt and d_ckpt:
            _, _, _, _, gradscaler_dict = load_checkpoint(g_ckpt, net_g, optim_g, strict_load=True)
            load_checkpoint(d_ckpt, net_d, optim_d, strict_load=True)
            global_step = int(os.path.basename(g_ckpt).split("_")[-1].split(".")[0])
            epoch_str = (global_step // len(train_loader)) + 1
            # If we resumed past freeze point, ensure decoder is unfrozen
            if global_step >= args.decoder_freeze_steps and decoder_frozen:
                model_ref = net_g.module if hasattr(net_g, "module") else net_g
                for p in model_ref.dec.parameters():
                    p.requires_grad = True
                decoder_frozen = False
                # Rebuild optimizer with all params
                optim_g = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, net_g.parameters()),
                    lr=args.lr_g, betas=(0.8, 0.99), eps=1e-9,
                )
            if rank == 0:
                print(f"[Phase 2] Resuming from step {global_step}, epoch {epoch_str - 1}")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, Exception):
        if rank == 0:
            print(f"[Phase 2] Starting from epoch 1")

    # Losses
    if args.spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = torch.nn.L1Loss()
    elif args.spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(sample_rate=args.sample_rate)
    elif args.spectral_loss == "Multi-Res STFT Loss":
        fn_spectral_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 4096], hop_sizes=[256, 512, 1024],
            win_lengths=[1024, 2048, 4096], window="hann_window",
            scale="mel", n_bins=128, sample_rate=args.sample_rate,
            perceptual_weighting=True, device=device,
        )

    if args.adversarial_loss == "hinge":
        fn_hinge = HingeAdversarialLoss()
    elif args.adversarial_loss == "soft_hinge":
        fn_hinge = SoftHingeAdversarialLoss()
    else:
        fn_hinge = None

    fn_lecam = LeCamRegularization(decay=0.9999).to(device) if args.vocoder == "ChouwaGAN" else None

    train_dtype = torch.float16 if args.fp16 else torch.float32
    use_amp = args.fp16 and device.type == "cuda"
    gradscaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))
    if gradscaler_dict:
        gradscaler.load_state_dict(gradscaler_dict)

    # Tensorboard
    if rank == 0:
        writer = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval_pretrain_phase2"),
            flush_secs=86400, purge_step=global_step + 1,
        )
        block_tensorboard_flush_on_exit(writer)
        print(f"[Phase 2] Full VITS pretrain: {args.vocoder} @ {args.sample_rate}Hz")
        print(f"[Phase 2] KL anneal steps: {args.kl_anneal_steps}, free bits: {args.kl_free_bits}")
        print(f"[Phase 2] Decoder freeze steps: {args.decoder_freeze_steps}")

    # Reference for eval infer
    info = next(iter(train_loader))
    if device.type == "cuda":
        info = [t.cuda(device_id, non_blocking=True) for t in info]
    phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
    reference = (phone, phone_lengths, pitch, pitchf, sid, config.train.seed)

    epoch_recorder = EpochRecorder()
    avg_rolling = {
        "loss_disc": deque(maxlen=args.rolling_loss_steps),
        "loss_adv": deque(maxlen=args.rolling_loss_steps),
        "loss_mel": deque(maxlen=args.rolling_loss_steps),
        "loss_fm": deque(maxlen=args.rolling_loss_steps),
        "loss_kl": deque(maxlen=args.rolling_loss_steps),
        "loss_total": deque(maxlen=args.rolling_loss_steps),
    }

    for epoch in range(epoch_str, args.total_epochs + 1):
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()

        current_epoch_start = (epoch - 1) * len(train_loader)
        start_batch = max(0, global_step - current_epoch_start)
        remaining = len(train_loader) - start_batch
        data_iter = islice(enumerate(train_loader), remaining)

        from tqdm import tqdm
        with tqdm(total=len(train_loader), leave=False, initial=start_batch,
                  desc=f"Phase2 E{epoch}") as pbar:
            for batch_idx, info in data_iter:
                global_step += 1

                # ─── Unfreeze decoder after N steps ───
                if decoder_frozen and global_step >= args.decoder_freeze_steps:
                    model_ref = net_g.module if hasattr(net_g, "module") else net_g
                    for p in model_ref.dec.parameters():
                        p.requires_grad = True
                    decoder_frozen = False
                    # Rebuild optimizer to include decoder params
                    optim_g = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, net_g.parameters()),
                        lr=args.lr_g, betas=(0.8, 0.99), eps=1e-9,
                    )
                    if rank == 0:
                        print(f"\n[Phase 2] Decoder UNFROZEN at step {global_step}")

                if device.type == "cuda":
                    info = [t.cuda(device_id, non_blocking=True) for t in info]
                else:
                    info = [t.to(device) for t in info]

                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid = info

                # ─── Generator forward ───
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)

                    if args.vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (mag, _), pitch_pred = model_output
                    elif args.vocoder == "APEX-GAN":
                        y_hat_list, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pitch_pred = model_output
                        y_hat = y_hat_list[-1]
                    else:
                        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pitch_pred = model_output

                    y_sliced = commons.slice_segments(
                        y, ids_slice * config.data.hop_length,
                        config.train.segment_size, dim=3,
                    )

                # ─── Discriminator step ───
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    if args.vocoder == "APEX-GAN":
                        y_hat_d = [o.detach() for o in y_hat_list]
                    else:
                        y_hat_d = y_hat.detach()
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat_d)

                with autocast(device_type="cuda", enabled=False):
                    if args.adversarial_loss == "lsgan":
                        loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    elif args.adversarial_loss == "tprls":
                        loss_disc = discriminator_tprls_loss(y_d_hat_r, y_d_hat_g)
                    elif args.adversarial_loss in ("hinge", "soft_hinge"):
                        loss_fake, loss_real = fn_hinge(y_d_hat_g, y_d_hat_r)
                        loss_disc = loss_fake + loss_real
                    if fn_lecam is not None:
                        loss_disc = loss_disc + 0.2 * fn_lecam(y_d_hat_r, y_d_hat_g)
                        fn_lecam.update_ema(y_d_hat_r, y_d_hat_g)

                optim_d.zero_grad(set_to_none=True)
                if train_dtype == torch.float16:
                    gradscaler.scale(loss_disc).backward()
                    gradscaler.unscale_(optim_d)
                    clip_grad_norm_(net_d.parameters(), max_norm=args.grad_clip_d)
                    gradscaler.step(optim_d)
                else:
                    loss_disc.backward()
                    clip_grad_norm_(net_d.parameters(), max_norm=args.grad_clip_d)
                    optim_d.step()

                # ─── Generator losses ───
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    if args.vocoder == "APEX-GAN":
                        _, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat_list)
                    else:
                        _, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)

                with autocast(device_type="cuda", enabled=False):
                    # Spectral
                    if args.spectral_loss == "L1 Mel Loss":
                        y_mel = wave_to_mel(config, y_sliced, half=train_dtype)
                        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)
                        loss_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                    elif args.spectral_loss == "Multi-Scale Mel Loss":
                        loss_mel = fn_spectral_loss(y_sliced, y_hat) * config.train.c_mel / 3.0
                    elif args.spectral_loss == "Multi-Res STFT Loss":
                        loss_mel = fn_spectral_loss(y_hat.float(), y_sliced.float()) * 21.0

                    # Feature matching
                    loss_fm = feature_loss(fmap_r, fmap_g)

                    # Adversarial
                    if args.adversarial_loss == "lsgan":
                        loss_adv = generator_loss(y_d_hat_g)
                    elif args.adversarial_loss == "tprls":
                        y_d_hat_r_det = [i.detach() for i in y_d_hat_r]
                        loss_adv = generator_tprls_loss(y_d_hat_r_det, y_d_hat_g)
                    elif args.adversarial_loss in ("hinge", "soft_hinge"):
                        loss_adv = fn_hinge(y_d_hat_g)

                    if args.vocoder == "ChouwaGAN":
                        n_disc = len(y_d_hat_g)
                        loss_adv = loss_adv / n_disc
                        loss_fm = loss_fm / n_disc

                    # KL loss with annealing + free bits
                    kl_w = kl_weight_linear(global_step, args.kl_anneal_steps)

                    if args.vocoder == "ChouwaGAN":
                        loss_kl = kl_loss_floored(z_p, logs_q, m_p, logs_p, z_mask,
                                                  free_bits=args.kl_free_bits) * config.train.c_kl
                    else:
                        loss_kl = kl_loss_floored(z_p, logs_q, m_p, logs_p, z_mask,
                                                  free_bits=args.kl_free_bits) * config.train.c_kl

                    loss_gen_total = loss_adv + loss_fm + loss_mel + loss_kl * kl_w

                optim_g.zero_grad(set_to_none=True)
                if train_dtype == torch.float16:
                    gradscaler.scale(loss_gen_total).backward()
                    gradscaler.unscale_(optim_g)
                    clip_grad_norm_(net_g.parameters(), max_norm=args.grad_clip_g)
                    gradscaler.step(optim_g)
                    gradscaler.update()
                else:
                    loss_gen_total.backward()
                    clip_grad_norm_(net_g.parameters(), max_norm=args.grad_clip_g)
                    optim_g.step()

                # Rolling losses
                avg_rolling["loss_disc"].append(loss_disc.detach())
                avg_rolling["loss_adv"].append(loss_adv.detach())
                avg_rolling["loss_mel"].append(loss_mel.detach())
                avg_rolling["loss_fm"].append(loss_fm.detach())
                avg_rolling["loss_kl"].append(loss_kl.detach())
                avg_rolling["loss_total"].append(loss_gen_total.detach())

                if rank == 0 and global_step % args.rolling_loss_steps == 0:
                    scalars = {}
                    for key, queue in avg_rolling.items():
                        if len(queue) > 0:
                            val = torch.stack(list(queue)).mean().item()
                            scalars[f"phase2_rolling/{key}"] = val
                    scalars["phase2/kl_weight"] = kl_w
                    scalars["phase2/decoder_frozen"] = 1.0 if decoder_frozen else 0.0
                    scalars["phase2/lr_g"] = optim_g.param_groups[0]["lr"]
                    summarize(writer=writer, global_step=global_step, scalars=scalars)
                    flush_writer(writer, rank)

                # Audio preview
                if rank == 0 and global_step % args.preview_interval == 0:
                    o = eval_infer(net_g, reference)
                    audio_dict = {f"phase2/audio_{global_step}s": o[0, :, :]}
                    summarize(writer=writer, global_step=global_step,
                              audios=audio_dict, audio_sample_rate=args.sample_rate)
                    flush_writer(writer, rank)
                    torch.cuda.empty_cache()

                pbar.update(1)

                if stopper.stop_triggered:
                    if rank == 0:
                        _save_phase2_checkpoint(
                            net_g, net_d, optim_g, optim_d, args, config,
                            epoch, global_step, experiment_dir, gradscaler,
                        )
                    if n_gpus > 1:
                        dist.barrier()
                    return

        # End of epoch
        if n_gpus > 1 and device.type == "cuda":
            dist.barrier()
        torch.cuda.empty_cache()

        if rank == 0:
            print(f"[Phase 2] {args.model_name} | epoch={epoch} | step={global_step} | kl_w={kl_w:.4f} | {epoch_recorder.record()}")

            if epoch % args.save_every == 0 or epoch == args.total_epochs:
                _save_phase2_checkpoint(
                    net_g, net_d, optim_g, optim_d, args, config,
                    epoch, global_step, experiment_dir, gradscaler,
                )

            if epoch == args.total_epochs:
                print(f"[Phase 2] Training complete! step={global_step}")
                writer.flush()
                writer.close()
                os._exit(0)


def _save_phase2_checkpoint(net_g, net_d, optim_g, optim_d,
                            args, config, epoch, global_step, experiment_dir, gradscaler):
    g_path = os.path.join(experiment_dir, f"G_phase2_{global_step}.pth")
    d_path = os.path.join(experiment_dir, f"D_phase2_{global_step}.pth")

    if args.save_only_latest:
        for pattern in ["G_phase2_*.pth", "D_phase2_*.pth"]:
            for f in glob.glob(os.path.join(experiment_dir, pattern)):
                try:
                    os.remove(f)
                except:
                    pass

    save_checkpoint(net_g, optim_g, args.lr_g, global_step, g_path, gradscaler)
    save_checkpoint(net_d, optim_d, args.lr_d, global_step, d_path, gradscaler)

    # Also save as standard G/D format for RVC compatibility
    ckpt = _strip_compile_prefix(_unwrap_model(net_g).state_dict())
    rvc_g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
    rvc_d_path = os.path.join(experiment_dir, f"D_{global_step}.pth")
    save_checkpoint(net_g, optim_g, args.lr_g, global_step, rvc_g_path, gradscaler)
    save_checkpoint(net_d, optim_d, args.lr_d, global_step, rvc_d_path, gradscaler)

    # Save extractable weight model
    weight_name = small_model_naming(args.model_name, epoch, global_step)
    weight_path = os.path.join(experiment_dir, weight_name)
    if not os.path.exists(weight_path):
        extract_model(
            ckpt=ckpt, sr=args.sample_rate, name=args.model_name,
            model_path=weight_path, epoch=epoch, step=global_step,
            hps=config, vocoder=args.vocoder, architecture=args.architecture,
        )


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Sample Rate Adaptation
# ═══════════════════════════════════════════════════════════════════

def run_phase3(rank, n_gpus, args, config, device, device_id):
    """
    Phase 3: Load Phase 2 checkpoint and adapt to a different sample rate.
    Same training loop as Phase 2, but without KL annealing (kl_w=1.0)
    and without decoder freezing.
    """
    global_step = 0
    stopper = EarlyStopSignalHandler()

    experiment_dir = os.path.join(now_dir, "logs", args.model_name)

    # Distributed setup
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # Dataloaders
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )
    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    train_sampler = DistributedBucketSampler(
        train_dataset, args.batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset, num_workers=4, shuffle=False, pin_memory=True,
        collate_fn=TextAudioCollateMultiNSFsid(),
        batch_sampler=train_sampler,
        persistent_workers=True, prefetch_factor=8,
    )
    train_loader_safety(train_loader)

    model_info_path = os.path.join(experiment_dir, "model_info.json")
    spk_dim = verify_spk_dim(config, model_info_path, experiment_dir, latest_checkpoint_path, rank, args.pretrain_g)
    config.model.spk_embed_dim = spk_dim

    # Models
    net_g = get_g_model(config, args.sample_rate, args.vocoder, args.use_checkpointing)
    net_d = get_d_model(config, args.vocoder, args.use_checkpointing, args.sample_rate)

    # Load Phase 2 checkpoint
    if args.phase2_ckpt_g and os.path.isfile(args.phase2_ckpt_g):
        if rank == 0:
            print(f"[Phase 3] Loading Phase 2 G: {args.phase2_ckpt_g}")
        ckpt = torch.load(args.phase2_ckpt_g, map_location="cpu", weights_only=True)
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = _strip_compile_prefix(state)
        net_g.load_state_dict(state, strict=False)

    if args.phase2_ckpt_d and os.path.isfile(args.phase2_ckpt_d):
        if rank == 0:
            print(f"[Phase 3] Loading Phase 2 D: {args.phase2_ckpt_d}")
        ckpt = torch.load(args.phase2_ckpt_d, map_location="cpu", weights_only=True)
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = _strip_compile_prefix(state)
        net_d.load_state_dict(state, strict=False)

    # Move to device
    if device.type == "cuda":
        net_g = net_g.to(device_id)
        net_d = net_d.to(device_id)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id])
        net_d = DDP(net_d, device_ids=[device_id])

    if args.use_torch_compile and sys.platform == "linux":
        try:
            net_g = torch.compile(net_g, mode="max-autotune-no-cudagraphs")
            net_d = torch.compile(net_d, mode="max-autotune-no-cudagraphs")
            if rank == 0:
                print("    ██████  torch.compile enabled for G and D (max-autotune-no-cudagraphs)")
        except Exception as e:
            if rank == 0:
                print(f"    ██████  torch.compile failed, falling back to eager mode: {e}")

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        lr=args.lr_g, betas=(0.8, 0.99), eps=1e-9,
    )
    optim_d = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_d.parameters()),
        lr=args.lr_d, betas=(0.8, 0.99), eps=1e-9,
    )

    epoch_str = 1
    gradscaler_dict = {}

    # Resume
    def get_highest_checkpoint(prefix, directory):
        pattern = re.compile(rf"^{prefix}(\d+)\.pth$")
        files = []
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                match = pattern.match(f)
                if match:
                    files.append((int(match.group(1)), os.path.join(directory, f)))
        return sorted(files, key=lambda x: x[0], reverse=True)[0][1] if files else None

    try:
        g_ckpt = get_highest_checkpoint("G_phase3_", experiment_dir)
        d_ckpt = get_highest_checkpoint("D_phase3_", experiment_dir)
        if g_ckpt and d_ckpt:
            _, _, _, _, gradscaler_dict = load_checkpoint(g_ckpt, net_g, optim_g, strict_load=True)
            load_checkpoint(d_ckpt, net_d, optim_d, strict_load=True)
            global_step = int(os.path.basename(g_ckpt).split("_")[-1].split(".")[0])
            epoch_str = (global_step // len(train_loader)) + 1
            if rank == 0:
                print(f"[Phase 3] Resuming from step {global_step}")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, Exception):
        if rank == 0:
            print(f"[Phase 3] Starting SR adaptation @ {args.sample_rate}Hz")

    # Losses
    if args.spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = torch.nn.L1Loss()
    elif args.spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(sample_rate=args.sample_rate)
    elif args.spectral_loss == "Multi-Res STFT Loss":
        fn_spectral_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 4096], hop_sizes=[256, 512, 1024],
            win_lengths=[1024, 2048, 4096], window="hann_window",
            scale="mel", n_bins=128, sample_rate=args.sample_rate,
            perceptual_weighting=True, device=device,
        )

    if args.adversarial_loss == "hinge":
        fn_hinge = HingeAdversarialLoss()
    elif args.adversarial_loss == "soft_hinge":
        fn_hinge = SoftHingeAdversarialLoss()
    else:
        fn_hinge = None

    fn_lecam = LeCamRegularization(decay=0.9999).to(device) if args.vocoder == "ChouwaGAN" else None

    train_dtype = torch.float16 if args.fp16 else torch.float32
    use_amp = args.fp16 and device.type == "cuda"
    gradscaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))
    if gradscaler_dict:
        gradscaler.load_state_dict(gradscaler_dict)

    if rank == 0:
        writer = SummaryWriter(
            log_dir=os.path.join(experiment_dir, f"eval_pretrain_phase3_{args.sample_rate}"),
            flush_secs=86400, purge_step=global_step + 1,
        )
        block_tensorboard_flush_on_exit(writer)
        print(f"[Phase 3] SR adaptation: {args.vocoder} → {args.sample_rate}Hz")

    # Reference
    info = next(iter(train_loader))
    if device.type == "cuda":
        info = [t.cuda(device_id, non_blocking=True) for t in info]
    phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
    reference = (phone, phone_lengths, pitch, pitchf, sid, config.train.seed)

    epoch_recorder = EpochRecorder()
    avg_rolling = {
        "loss_disc": deque(maxlen=args.rolling_loss_steps),
        "loss_adv": deque(maxlen=args.rolling_loss_steps),
        "loss_mel": deque(maxlen=args.rolling_loss_steps),
        "loss_fm": deque(maxlen=args.rolling_loss_steps),
        "loss_kl": deque(maxlen=args.rolling_loss_steps),
        "loss_total": deque(maxlen=args.rolling_loss_steps),
    }

    for epoch in range(epoch_str, args.total_epochs + 1):
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()

        current_epoch_start = (epoch - 1) * len(train_loader)
        start_batch = max(0, global_step - current_epoch_start)
        remaining = len(train_loader) - start_batch
        data_iter = islice(enumerate(train_loader), remaining)

        from tqdm import tqdm
        with tqdm(total=len(train_loader), leave=False, initial=start_batch,
                  desc=f"Phase3 E{epoch}") as pbar:
            for batch_idx, info in data_iter:
                global_step += 1

                if device.type == "cuda":
                    info = [t.cuda(device_id, non_blocking=True) for t in info]
                else:
                    info = [t.to(device) for t in info]

                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, y_lengths, sid = info

                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)

                    if args.vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (mag, _), pitch_pred = model_output
                    elif args.vocoder == "APEX-GAN":
                        y_hat_list, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pitch_pred = model_output
                        y_hat = y_hat_list[-1]
                    else:
                        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pitch_pred = model_output

                    y_sliced = commons.slice_segments(
                        y, ids_slice * config.data.hop_length,
                        config.train.segment_size, dim=3,
                    )

                # Disc step
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    if args.vocoder == "APEX-GAN":
                        y_hat_d = [o.detach() for o in y_hat_list]
                    else:
                        y_hat_d = y_hat.detach()
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat_d)

                with autocast(device_type="cuda", enabled=False):
                    if args.adversarial_loss == "lsgan":
                        loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    elif args.adversarial_loss == "tprls":
                        loss_disc = discriminator_tprls_loss(y_d_hat_r, y_d_hat_g)
                    elif args.adversarial_loss in ("hinge", "soft_hinge"):
                        loss_fake, loss_real = fn_hinge(y_d_hat_g, y_d_hat_r)
                        loss_disc = loss_fake + loss_real
                    if fn_lecam is not None:
                        loss_disc = loss_disc + 0.2 * fn_lecam(y_d_hat_r, y_d_hat_g)
                        fn_lecam.update_ema(y_d_hat_r, y_d_hat_g)

                optim_d.zero_grad(set_to_none=True)
                if train_dtype == torch.float16:
                    gradscaler.scale(loss_disc).backward()
                    gradscaler.unscale_(optim_d)
                    clip_grad_norm_(net_d.parameters(), max_norm=args.grad_clip_d)
                    gradscaler.step(optim_d)
                else:
                    loss_disc.backward()
                    clip_grad_norm_(net_d.parameters(), max_norm=args.grad_clip_d)
                    optim_d.step()

                # Gen losses
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    if args.vocoder == "APEX-GAN":
                        _, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat_list)
                    else:
                        _, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)

                with autocast(device_type="cuda", enabled=False):
                    if args.spectral_loss == "L1 Mel Loss":
                        y_mel = wave_to_mel(config, y_sliced, half=train_dtype)
                        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)
                        loss_mel = fn_spectral_loss(y_mel, y_hat_mel) * config.train.c_mel
                    elif args.spectral_loss == "Multi-Scale Mel Loss":
                        loss_mel = fn_spectral_loss(y_sliced, y_hat) * config.train.c_mel / 3.0
                    elif args.spectral_loss == "Multi-Res STFT Loss":
                        loss_mel = fn_spectral_loss(y_hat.float(), y_sliced.float()) * 21.0

                    loss_fm = feature_loss(fmap_r, fmap_g)

                    if args.adversarial_loss == "lsgan":
                        loss_adv = generator_loss(y_d_hat_g)
                    elif args.adversarial_loss == "tprls":
                        y_d_hat_r_det = [i.detach() for i in y_d_hat_r]
                        loss_adv = generator_tprls_loss(y_d_hat_r_det, y_d_hat_g)
                    elif args.adversarial_loss in ("hinge", "soft_hinge"):
                        loss_adv = fn_hinge(y_d_hat_g)

                    if args.vocoder == "ChouwaGAN":
                        n_disc = len(y_d_hat_g)
                        loss_adv = loss_adv / n_disc
                        loss_fm = loss_fm / n_disc

                    # Full KL (no annealing in Phase 3)
                    if args.vocoder == "ChouwaGAN":
                        loss_kl = kl_loss_floored(z_p, logs_q, m_p, logs_p, z_mask,
                                                  free_bits=args.kl_free_bits) * config.train.c_kl
                    else:
                        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl

                    loss_gen_total = loss_adv + loss_fm + loss_mel + loss_kl

                optim_g.zero_grad(set_to_none=True)
                if train_dtype == torch.float16:
                    gradscaler.scale(loss_gen_total).backward()
                    gradscaler.unscale_(optim_g)
                    clip_grad_norm_(net_g.parameters(), max_norm=args.grad_clip_g)
                    gradscaler.step(optim_g)
                    gradscaler.update()
                else:
                    loss_gen_total.backward()
                    clip_grad_norm_(net_g.parameters(), max_norm=args.grad_clip_g)
                    optim_g.step()

                avg_rolling["loss_disc"].append(loss_disc.detach())
                avg_rolling["loss_adv"].append(loss_adv.detach())
                avg_rolling["loss_mel"].append(loss_mel.detach())
                avg_rolling["loss_fm"].append(loss_fm.detach())
                avg_rolling["loss_kl"].append(loss_kl.detach())
                avg_rolling["loss_total"].append(loss_gen_total.detach())

                if rank == 0 and global_step % args.rolling_loss_steps == 0:
                    scalars = {}
                    for key, queue in avg_rolling.items():
                        if len(queue) > 0:
                            val = torch.stack(list(queue)).mean().item()
                            scalars[f"phase3_rolling/{key}"] = val
                    scalars["phase3/lr_g"] = optim_g.param_groups[0]["lr"]
                    summarize(writer=writer, global_step=global_step, scalars=scalars)
                    flush_writer(writer, rank)

                if rank == 0 and global_step % args.preview_interval == 0:
                    o = eval_infer(net_g, reference)
                    audio_dict = {f"phase3/audio_{global_step}s": o[0, :, :]}
                    summarize(writer=writer, global_step=global_step,
                              audios=audio_dict, audio_sample_rate=args.sample_rate)
                    flush_writer(writer, rank)
                    torch.cuda.empty_cache()

                pbar.update(1)

                if stopper.stop_triggered:
                    if rank == 0:
                        _save_phase3_checkpoint(
                            net_g, net_d, optim_g, optim_d, args, config,
                            epoch, global_step, experiment_dir, gradscaler,
                        )
                    if n_gpus > 1:
                        dist.barrier()
                    return

        if n_gpus > 1 and device.type == "cuda":
            dist.barrier()
        torch.cuda.empty_cache()

        if rank == 0:
            print(f"[Phase 3] {args.model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}")

            if epoch % args.save_every == 0 or epoch == args.total_epochs:
                _save_phase3_checkpoint(
                    net_g, net_d, optim_g, optim_d, args, config,
                    epoch, global_step, experiment_dir, gradscaler,
                )

            if epoch == args.total_epochs:
                print(f"[Phase 3] SR adaptation complete! step={global_step}")
                writer.flush()
                writer.close()
                os._exit(0)


def _save_phase3_checkpoint(net_g, net_d, optim_g, optim_d,
                            args, config, epoch, global_step, experiment_dir, gradscaler):
    g_path = os.path.join(experiment_dir, f"G_phase3_{global_step}.pth")
    d_path = os.path.join(experiment_dir, f"D_phase3_{global_step}.pth")

    if args.save_only_latest:
        for pattern in ["G_phase3_*.pth", "D_phase3_*.pth"]:
            for f in glob.glob(os.path.join(experiment_dir, pattern)):
                try:
                    os.remove(f)
                except:
                    pass

    save_checkpoint(net_g, optim_g, args.lr_g, global_step, g_path, gradscaler)
    save_checkpoint(net_d, optim_d, args.lr_d, global_step, d_path, gradscaler)

    # RVC-compatible checkpoint
    rvc_g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
    rvc_d_path = os.path.join(experiment_dir, f"D_{global_step}.pth")
    save_checkpoint(net_g, optim_g, args.lr_g, global_step, rvc_g_path, gradscaler)
    save_checkpoint(net_d, optim_d, args.lr_d, global_step, rvc_d_path, gradscaler)

    ckpt = _strip_compile_prefix(_unwrap_model(net_g).state_dict())
    weight_name = small_model_naming(args.model_name, epoch, global_step)
    weight_path = os.path.join(experiment_dir, weight_name)
    if not os.path.exists(weight_path):
        extract_model(
            ckpt=ckpt, sr=args.sample_rate, name=args.model_name,
            model_path=weight_path, epoch=epoch, step=global_step,
            hps=config, vocoder=args.vocoder, architecture=args.architecture,
        )


# ═══════════════════════════════════════════════════════════════════
# Main Entrypoint
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    # Setup experiment dir and config
    experiment_dir = os.path.join(now_dir, "logs", args.model_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Load or create config
    config_path = os.path.join(experiment_dir, "config.json")
    if os.path.isfile(config_path):
        config = load_config_from_json(config_path)
    else:
        # Load from vocoder template
        voc_dir = get_vocoder_config_dir(args.vocoder)
        template_path = os.path.join(now_dir, "rvc", "configs", voc_dir, f"{args.sample_rate}.json")
        if not os.path.isfile(template_path):
            print(f"[ERROR] Config template not found: {template_path}")
            sys.exit(1)
        config = load_config_from_json(template_path)
        # Save to experiment dir
        import shutil
        shutil.copy(template_path, config_path)
        print(f"[INIT] Created config from template: {template_path}")

    config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

    # Override sample rate for Phase 3
    if args.phase == 3:
        config.data.sample_rate = args.sample_rate

    # Torch settings
    torch.backends.cuda.matmul.allow_tf32 = args.use_tf32
    torch.backends.cudnn.allow_tf32 = args.use_tf32
    torch.backends.cudnn.benchmark = True

    # GPU setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(g) for g in args.gpu.split("-")]
        n_gpus = len(gpus)
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        print("[WARNING] No GPU detected, training on CPU (very slow)")

    # Phase dispatcher
    phase_fn = {1: run_phase1, 2: run_phase2, 3: run_phase3}[args.phase]

    children = []
    for rank, device_id in enumerate(gpus):
        p = mp.Process(
            target=phase_fn,
            args=(rank, n_gpus, args, config, device, device_id),
        )
        children.append(p)
        p.start()
        pid_data["process_pids"].append(p.pid)

    for p in children:
        p.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
