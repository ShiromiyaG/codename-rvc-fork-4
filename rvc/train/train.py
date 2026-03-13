import os
import io
import signal
import datetime
import glob
import itertools
import json
import math
import re
import subprocess
import sys

pid_data = {"process_pids": []}
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
# Reduce CUDA allocator fragmentation — critical on small VRAM GPUs (<= 8 GB)
# when R1 penalty or other ops cause irregular allocation patterns.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Suppress _POSIX_C_SOURCE redefinition noise emitted by GCC when Triton
# JIT-compiles its C stubs. Conda's pyconfig.h and the system's features.h
# both define the macro to different values; -w silences all GCC warnings
# for that translation unit without affecting PyTorch/CUDA compilation.
# Use append (not setdefault) so that conda's existing CFLAGS are preserved.
os.environ["CFLAGS"] = os.environ.get("CFLAGS", "") + " -w"
os.environ["CXXFLAGS"] = os.environ.get("CXXFLAGS", "") + " -w"
import warnings
# torch.inductor falls back to eager for complex-valued ops (STFT in MRD).
# This is expected for our training setup; suppress the per-step noise.
warnings.filterwarnings(
    "ignore",
    message=".*Torchinductor does not support code generation for complex operators.*",
    category=UserWarning,
)
from typing import Tuple, Optional
from collections import deque
from distutils.util import strtobool
from random import randint, shuffle
from time import time as ttime, sleep

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import psutil
from tqdm import tqdm
from pesq import pesq

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
sys.path.append(os.path.join(now_dir))

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
    si_sdr,
    wave_to_mel,
    small_model_naming,
    old_session_cleanup,
    print_init_setup,
    train_loader_safety,
    verify_spk_dim,
    early_stopper,
    WeightTrajectoryVisualizer
)
from losses import (
    discriminator_loss,
    generator_loss,
    discriminator_loss_v2,
    generator_loss_v2,
    HingeAdversarialLoss,
    feature_loss,
    kl_loss,
    kl_loss_clamped,
    phase_loss
)
from mel_processing import (
    spec_to_mel_torch,
    MultiScaleMelSpectrogramLoss
)
from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.train.utils import replace_keys_in_dict

# Parse command line arguments start region ===========================

model_name = sys.argv[1]
epoch_save_frequency = int(sys.argv[2])
total_epoch_count = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])
sample_rate = int(sys.argv[8])
save_only_latest_net_models = strtobool(sys.argv[9])
save_weight_models = strtobool(sys.argv[10])
use_warmup = strtobool(sys.argv[11])
warmup_duration = int(sys.argv[12])
cleanup = strtobool(sys.argv[13])
vocoder = sys.argv[14]
architecture = sys.argv[15]
optimizer_choice = sys.argv[16]
adversarial_loss = sys.argv[17]
use_checkpointing = strtobool(sys.argv[18])
use_tf32 = bool(strtobool(sys.argv[19]))
use_benchmark = bool(strtobool(sys.argv[20]))
use_deterministic = bool(strtobool(sys.argv[21]))
spectral_loss = sys.argv[22]
lr_scheduler = sys.argv[23]
exp_decay_gamma = float(sys.argv[24])
use_validation = strtobool(sys.argv[25])
use_kl_annealing = strtobool(sys.argv[26])
kl_annealing_cycle_duration = int(sys.argv[27])
vits_version = sys.argv[28]
rolling_loss_steps = int(sys.argv[29])
use_tstp = bool(strtobool(sys.argv[30]))
use_custom_lr = strtobool(sys.argv[31])
custom_lr_g, custom_lr_d = (float(sys.argv[32]), float(sys.argv[33])) if use_custom_lr else (None, None)
use_compile = bool(strtobool(sys.argv[34]))
compile_mode = sys.argv[35]
assert not use_custom_lr or (custom_lr_g and custom_lr_d), "Invalid custom LR values."

# Parse command line arguments end region ===========================

current_dir = os.getcwd()

# Derive experiment directory from model name
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")


# Load the config from json
config = load_config_from_json(config_save_path)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")


# AMP precision / dtype init
if config.train.bf16_run:
    train_dtype = torch.bfloat16
elif config.train.fp16_run: 
    train_dtype = torch.float16
else:
    train_dtype = torch.float32

# Torch backends config
torch.backends.cuda.matmul.allow_tf32 = use_tf32
torch.backends.cudnn.allow_tf32 = use_tf32
torch.backends.cudnn.benchmark = use_benchmark
torch.backends.cudnn.deterministic = use_deterministic
# Enable TF32 Tensor Cores for torch.matmul() (nn.Linear, ConvNeXt pointwise layers).
# 'high' = TF32 precision (10-bit mantissa, ~2x faster on Ampere+).
# 'highest' = full FP32 (default PyTorch behaviour when not using this).
torch.set_float32_matmul_precision('high' if use_tf32 else 'highest')

# Globals ( Do not alter these )
global_step = 0
warmup_completed = False
from_scratch = False
use_lr_scheduler = lr_scheduler != "none"

# Globals ( tweakable~ )
randomized = True
benchmark_mode = False
enable_persistent_workers = True

c_stft = 21.0 # Seems close enough to multi-scale mel loss's magnitude, but needs more testing.

pretrain_preview = True
pretrain_preview_interval = 100 # Measured in steps.

override_pretrain_lr = False
new_pretrain_lr = 5e-5 # If you changed it, it needs to be re-adjusted to the most recent lr ~ anytime you resume!

# EXPERIMENTAL
use_trajectory = False

# torch.compile — Ampere+ (RTX 3090 / A100 / H100) with PyTorch 2.x
# 'default'  — inductor kernel fusion + op optimization, no CUDA Graphs.
#              Best balance of speed and compatibility for GAN training
#              (variable-length batches from bucket sampler, complex STFT in MRD).
# 'max-autotune' — profiles kernels at startup (~5 min), best for >12 h runs.
#              Also avoids CUDA Graphs (uses inductor's autotuned triton kernels).
# NOTE: 'reduce-overhead' / 'max-autotune' enable CUDA Graphs which are
#       incompatible with this setup (variable shapes, multi-model steps,
#       complex ops → stalls, empty-graph warnings, tensor aliasing crashes).
#       'max-autotune-no-cudagraphs' runs Triton autotuning for best kernel
#       tile sizes without requiring fixed shapes.  First step is slow (~2-5
#       min while autotuning), steady-state is faster than 'default'.
# use_compile and compile_mode are now parsed from sys.argv above.

use_sid_swap = False
custom_sid = 1


##################################################################

import logging
logging.getLogger("torch").setLevel(logging.ERROR)

class EarlyStopSignalHandler:
    def __init__(self):
        self.stop_triggered = False
        signal.signal(signal.SIGINT, self._handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._handler)

    def _handler(self, signum, frame):
        self.stop_triggered = True
        print(f"\n[TRAINING] Early Stopping signal received! Finishing current step and saving...")



def eval_infer(net_g, reference):
    net_g.eval()
    with torch.no_grad():
        if hasattr(net_g, "module"):
            o, *_ = net_g.module.infer(*reference)
        else:
            o, *_ = net_g.infer(*reference)
    net_g.train()
    return o

class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        return f"Current time: {current_time} | Time per epoch: {elapsed_time_str}"

def setup_env_and_distr(rank, n_gpus, device, device_id, config):
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

def prepare_dataloaders(config, n_gpus, rank, batch_size, use_validation, benchmark_mode):
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid
    )

    if not benchmark_mode and use_validation:
        full_dataset = TextAudioLoaderMultiNSFsid(config.data)
        train_len = int(0.90 * len(full_dataset))
        val_len = len(full_dataset) - train_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(config.train.seed)
        )
        train_dataset.lengths = [full_dataset.lengths[i] for i in train_dataset.indices]
        val_dataset.lengths = [full_dataset.lengths[i] for i in val_dataset.indices]
    else:
        train_dataset = TextAudioLoaderMultiNSFsid(config.data)
        val_dataset = None

    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )

    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=enable_persistent_workers,
        prefetch_factor=8
    )
    val_loader = None
    if val_dataset:
        val_sampler = DistributedBucketSampler(
            val_dataset,
            batch_size * n_gpus,
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_sampler=val_sampler, shuffle=False, collate_fn=collate_fn,
            num_workers=1, pin_memory=True
        )
    
    train_loader_safety(benchmark_mode, train_loader)

    return train_loader, val_loader

def get_g_model(config, sample_rate, vocoder, use_checkpointing, randomized):
    from rvc.lib.algorithm.synthesizers import Synthesizer
    return Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0 = True,
        sr = sample_rate,
        vocoder = vocoder,
        checkpointing = use_checkpointing,
        randomized = randomized,
        vits_version = vits_version,
    )

def get_d_model(config, vocoder, use_checkpointing):
    if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined
        # MPD + MSD + MRD ( unified ) - RingFormer architecture v1 and v2
        return MPD_MSD_MRD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            **dict(config.mrd)
        )
    elif vocoder == "PCPH-GAN":
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined
        mrd_config = dict(config.mrd) if hasattr(config, "mrd") else {}
        return MPD_MSD_MRD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            **mrd_config
        )
    elif vocoder == "ChouwaGAN":
        from rvc.lib.algorithm.discriminators.multi import ChouwaGANDiscriminator
        # MS-STFT + FastMPD + UnivHD (frequency + time-domain, ~2.7M params)
        chouwa_cfg = dict(config.mrd) if hasattr(config, "mrd") else {}
        sample_rate = config.data.sample_rate if hasattr(config.data, "sample_rate") else 48000
        return ChouwaGANDiscriminator(
            use_spectral_norm=config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing,
            sample_rate=sample_rate,
            **chouwa_cfg
        )
    elif vocoder == "RefineGAN":
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_MRD_Combined_RefineGan
        # Trimmed MPD + MSD + MRD ( unified )
        return MPD_MSD_MRD_Combined_RefineGan(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing
        )
    else: # For NSF HiFi-GAN
        from rvc.lib.algorithm.discriminators.multi import MPD_MSD_Combined
        # MPD + MSD ( unified ) - Original RVC Setup
        return MPD_MSD_Combined(
            config.model.use_spectral_norm,
            use_checkpointing=use_checkpointing
        )

def get_optimizers(
    net_g,
    net_d,
    config,
    optimizer_choice,
    custom_lr_g,
    custom_lr_d,
    use_custom_lr,
    total_epoch_count,
    train_loader
):
    # Common args for optims
    common_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
    )
    common_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
    )
    adamwspd_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.5,
    )
    adamwspd_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.5,
    )
    radam_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.01,
        decoupled_weight_decay=True,
        foreach=True,
    )
    radam_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0.01,
        decoupled_weight_decay=True,
        foreach=True,
    )

    d_lr_factor = 1.0  # ChouwaGAN uses TTUR: equal LR, R1 regulates D
    if d_lr_factor != 1.0:
        for d_args in (common_args_d, adamwspd_args_d, radam_args_d):
            d_args["lr"] *= d_lr_factor
        print(f"    ██████  D LR factor: {d_lr_factor:.2f}x  (lr_d = {common_args_d['lr']:.2e}  lr_g = {common_args_g['lr']:.2e})")

    # For exotic optimizers
    ranger_args = dict(
        num_epochs=total_epoch_count,
        num_batches_per_epoch=len(train_loader),
        use_madgrad=False,
        use_warmup=False,
        warmdown_active=False,
        use_cheb=False,
        lookahead_active=True,
        normloss_active=False,
        normloss_factor=1e-4,
        softplus=False,
        use_adaptive_gradient_clipping=True,
        agc_clipping_value=0.01,
        agc_eps=1e-3,
        using_gc=True,
        gc_conv_only=True,
        using_normgc=False,
    )

    if optimizer_choice == "AdamW":
        optim_g = torch.optim.AdamW(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g, fused=True)
        optim_d = torch.optim.AdamW(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d, fused=True)

    elif optimizer_choice == "AdamW BF16":
        from optimi import AdamW as AdamW_BF16
        optim_g = AdamW_BF16(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g, kahan_sum=True, foreach=True)
        optim_d = AdamW_BF16(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d, kahan_sum=True, foreach=True)

    elif optimizer_choice == "RAdam":
        optim_g = torch.optim.RAdam(filter(lambda p: p.requires_grad, net_g.parameters()), **radam_args_g)
        optim_d = torch.optim.RAdam(filter(lambda p: p.requires_grad, net_d.parameters()), **radam_args_d)

    elif optimizer_choice == "DiffGrad":
        from rvc.train.custom_optimizers.diffgrad import diffgrad
        optim_g = diffgrad(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g)
        optim_d = diffgrad(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d)

    elif optimizer_choice == "Ranger21":
        from rvc.train.custom_optimizers.ranger21 import Ranger21
        optim_g = Ranger21(filter(lambda p: p.requires_grad, net_g.parameters()), **common_args_g, **ranger_args)
        optim_d = Ranger21(filter(lambda p: p.requires_grad, net_d.parameters()), **common_args_d, **ranger_args)

    elif optimizer_choice == "AdamSPD":
        import copy
        from rvc.train.custom_optimizers.adamspd import AdamSPD

        # Get trainable parameters and cache the pre-trained weights
        params_to_opt_g = [p for p in net_g.parameters() if p.requires_grad]
        params_anchor_g = copy.deepcopy(params_to_opt_g) 
        # Parameter group with the anchor
        param_group_g = [{'params': params_to_opt_g, 'pre': params_anchor_g}]

        # Get trainable parameters and cache the pre-trained weights
        params_to_opt_d = [p for p in net_d.parameters() if p.requires_grad]
        params_anchor_d = copy.deepcopy(params_to_opt_d) 
        # Parameter group with the anchor
        param_group_d = [{'params': params_to_opt_d, 'pre': params_anchor_d}]

        optim_g = AdamSPD(param_group_g, **adamwspd_args_g)
        optim_d = AdamSPD(param_group_d, **adamwspd_args_d,)

        proj_strength_mult_g = adamwspd_args_g['weight_decay']
        proj_strength_mult_d = adamwspd_args_d['weight_decay']
        print(f"    ██████  Proj. Strength Mult. for AdamSPD: G; {proj_strength_mult_g}, D; {proj_strength_mult_d}")
    else:
        raise ValueError(f"Unknown optimizer choice: {optimizer_choice}")
    return optim_g, optim_d

def setup_models_for_training(net_g, net_d, device, device_id, n_gpus, train_dtype=None):
    net_g = net_g.to(device_id) if device.type == "cuda" else net_g.to(device)
    net_d = net_d.to(device_id) if device.type == "cuda" else net_d.to(device)

    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id]) # find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[device_id]) # find_unused_parameters=True)

    return net_g, net_d

def load_models_and_optimizers(config, pretrainG, pretrainD, vocoder, use_checkpointing, randomized, sample_rate, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader, device, device_id, n_gpus, rank):
    # Init the models
    net_g = get_g_model(config, sample_rate, vocoder, use_checkpointing, randomized)
    net_d = get_d_model(config, vocoder, use_checkpointing)
    try:
        print("    ██████  Starting the training ...")

        # Get latest G and D based on the highest steps count in the filename
        def get_highest_checkpoint(prefix):
            pattern = re.compile(rf"^{prefix}(\d+)\.pth$")
            files = []
            for f in os.listdir(experiment_dir):
                match = pattern.match(f)
                if match:
                    files.append((int(match.group(1)), os.path.join(experiment_dir, f)))
            return sorted(files, key=lambda x: x[0], reverse=True)[0][1] if files else None

        # Confirm presence of checkpoints
        # If they exist, attempt to resume the training
        g_checkpoint_path = get_highest_checkpoint("G_")
        d_checkpoint_path = get_highest_checkpoint("D_")
        if g_checkpoint_path and d_checkpoint_path:

            # Init the optimizers
            optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)
            # Move the models to an appropriate device ( And optionally wrap with DDP for multi-gpu )
            net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus, train_dtype)

            # Load the model and optim states
            _, _, _, epoch_str, gradscaler_dict, _, chouwa_balancer_state = load_checkpoint(g_checkpoint_path, net_g, optim_g)
            _, _, _, epoch_str, _, _, _ = load_checkpoint(d_checkpoint_path, net_d, optim_d)

            if override_pretrain_lr:
                new_lr_for_pretrain = new_pretrain_lr
                for param_group in optim_g.param_groups:
                    param_group['lr'] = new_lr_for_pretrain
                    param_group['initial_lr'] = new_lr_for_pretrain
                for param_group in optim_d.param_groups:
                    param_group['lr'] = new_lr_for_pretrain
                    param_group['initial_lr'] = new_lr_for_pretrain
                print(f"[OVERRIDE] Pretrain LR Override: {new_lr_for_pretrain}")

            #epoch_str += 1
            #global_step = (epoch_str - 1) * len(train_loader)

            global_step = int(os.path.basename(g_checkpoint_path).split("_")[-1].split(".")[0])
            epoch_str = (global_step // len(train_loader)) + 1
            print(f"[RESUMING] (G) & (D) at global_step: {global_step} and epoch count: {epoch_str - 1}")
        else:
            raise FileNotFoundError("No checkpoints found.")

    except FileNotFoundError:
    # If no checkpoints are available, using the Pretrains directly
        epoch_str = 1
        global_step = 0
        gradscaler_dict = {}
        chouwa_balancer_state = None

        # Loading the pretrained Generator model
        if pretrainG not in ["", "None"]:
            if rank == 0:
                print(f"[ ] Loading pretrained (G) '{pretrainG}'")
            checkpoint = torch.load(pretrainG, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            net_g.load_state_dict(state_dict, strict=True)

            if use_sid_swap and custom_sid != 0:
                total_sids = net_g.emb_g.weight.size(0)

                if custom_sid >= total_sids:
                    print(f"[SID SWAP] {custom_sid} is out of bounds!")
                    print(f"[SID SWAP] Currently chosen pretrains only support SIDs from 0 to {total_sids - 1}.")
                    sys.exit("Invalid SID Selection. Please choose a lower custom_sid.")
                if rank == 0:
                    print(f"███ [SID SWAP] Swapping SID: 0 with SID: {custom_sid}")

                with torch.no_grad():
                    temp_sid_0 = net_g.emb_g.weight[0].clone()
                    net_g.emb_g.weight[0].copy_(net_g.emb_g.weight[custom_sid])
                    net_g.emb_g.weight[custom_sid].copy_(temp_sid_0)
                if rank == 0:
                    print(f"███ [SID SWAP] Swap successful. Model is ready for fine-tuning.")

        # Loading the pretrained Discriminator model
        if pretrainD not in ["", "None"]:
            if rank == 0:
                print(f"[ ] Loading pretrained (D) '{pretrainD}'")
            checkpoint = torch.load(pretrainD, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            net_d.load_state_dict(state_dict, strict=True)

        # Load the models and optionally wrap with DDP
        net_g, net_d = setup_models_for_training(net_g, net_d, device, device_id, n_gpus, train_dtype)

        # Init the optimizers
        optim_g, optim_d = get_optimizers(net_g, net_d, config, optimizer_choice, custom_lr_g, custom_lr_d, use_custom_lr, total_epoch_count, train_loader)

    return net_g, net_d, optim_g, optim_d, epoch_str, global_step, gradscaler_dict, None, chouwa_balancer_state

def prepare_schedulers(optim_g, optim_d, use_warmup, warmup_duration, use_lr_scheduler, lr_scheduler, exp_decay_gamma, total_epoch_count, epoch_str, global_step, train_loader):
    warmup_scheduler_g, warmup_scheduler_d = None, None
    scheduler_g, scheduler_d = None, None

    num_batches_per_epoch = len(train_loader)

    if override_pretrain_lr:
        scheduler_resume_epoch = -1
        scheduler_resume_step = -1
    else:
        scheduler_resume_epoch = epoch_str - 1
        scheduler_resume_step = global_step - 1

    if use_warmup:
        warmup_scheduler_g = torch.optim.lr_scheduler.LambdaLR(
            optim_g, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_duration)
        )
        warmup_scheduler_d = torch.optim.lr_scheduler.LambdaLR(
            optim_d, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_duration)
        )

    if not use_warmup:
        for param_group in optim_g.param_groups: # For Generator
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
        for param_group in optim_d.param_groups: # For Discriminator
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

    if use_lr_scheduler:
        if lr_scheduler == "exp decay step":
            exp_decay_gamma_step = exp_decay_gamma ** (1.0 / num_batches_per_epoch)

        if lr_scheduler == "exp decay epoch":
            # Exponential decay lr scheduler per epoch
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch)
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=exp_decay_gamma, last_epoch=scheduler_resume_epoch)

        elif lr_scheduler == "exp decay step":
            # Exponential-decay style lr scheduler per step
            exp_decay_gamma_step = exp_decay_gamma ** (1.0 / num_batches_per_epoch)
            class DynamicStepLR(torch.optim.lr_scheduler.MultiplicativeLR):
                def __init__(self, optimizer, gamma, last_epoch=-1):
                    self.gamma = gamma
                    super().__init__(optimizer, lr_lambda=lambda step: self.gamma, last_epoch=last_epoch)
            scheduler_g = DynamicStepLR(optim_g, gamma=exp_decay_gamma_step, last_epoch=scheduler_resume_step)
            scheduler_d = DynamicStepLR(optim_d, gamma=exp_decay_gamma_step, last_epoch=scheduler_resume_step)



        elif lr_scheduler == "cosine annealing epoch":
            # Cosine annealing lr scheduler per epoch
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=total_epoch_count, eta_min=2e-5, last_epoch=scheduler_resume_epoch)
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=total_epoch_count, eta_min=2e-5, last_epoch=scheduler_resume_epoch)

    return warmup_scheduler_g, warmup_scheduler_d, scheduler_g, scheduler_d

def get_reference_sample(train_loader, device, config):
    reference_path = os.path.join("logs", "reference")
    use_custom_ref = all([
        os.path.isfile(os.path.join(reference_path, "ref_feats.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0c.npy")),
        os.path.isfile(os.path.join(reference_path, "ref_f0f.npy")),
    ])

    if use_custom_ref:
        print("[REFERENCE] Using custom reference input from 'logs\\reference\\'")

        phone = torch.FloatTensor(np.repeat(np.load(os.path.join(reference_path, "ref_feats.npy")), 2, axis=0)).unsqueeze(0).to(device)
        pitch = torch.LongTensor(np.load(os.path.join(reference_path, "ref_f0c.npy"))).unsqueeze(0).to(device)
        pitchf = torch.FloatTensor(np.load(os.path.join(reference_path, "ref_f0f.npy"))).unsqueeze(0).to(device)

        min_len = min(phone.shape[1], pitch.shape[1], pitchf.shape[1])

        phone, pitch, pitchf = phone[:, :min_len, :], pitch[:, :min_len], pitchf[:, :min_len]
        phone_lengths = torch.LongTensor([phone.shape[1]]).to(device)

        sid = torch.LongTensor([0]).to(device)
    else:
        print("[REFERENCE] No custom reference found. Fetching from the first batch of the train_loader.")

        info = next(iter(train_loader))
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
        phone, phone_lengths, pitch, pitchf, sid = phone.to(device), phone_lengths.to(device), pitch.to(device), pitchf.to(device), sid.to(device)

        batch_indices = []
        for batch in train_loader.batch_sampler:
            batch_indices = batch
            break

        if isinstance(train_loader.dataset, torch.utils.data.Subset):
            file_paths = train_loader.dataset.dataset.get_file_paths(batch_indices)
        else:
            file_paths = train_loader.dataset.get_file_paths(batch_indices)

        file_name = os.path.basename(file_paths[0])
        print(f"[REFERENCE] Origin of the ref: {file_name}")

    return (phone, phone_lengths, pitch, pitchf, sid, config.train.seed)

def main():
    """
    Main function to start the training process.
    """
    global gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    wavs = glob.glob(os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*.wav"))
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            print(f"Error: Pretrained model sample rate ({sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz).")
            os._exit(1)
    else:
        print("No wav file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus) 
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        print("No GPU detected, fallback to CPU. This will take a very long time ...")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []

        for rank, device_id in enumerate(gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    rank,
                    n_gpus,
                    experiment_dir,
                    pretrainG,
                    pretrainD,
                    total_epoch_count,
                    epoch_save_frequency,
                    save_weight_models,
                    save_only_latest_net_models,
                    config,
                    device,
                    device_id,
                ),
            )
            children.append(subproc)
            subproc.start()
            pid_data["process_pids"].append(subproc.pid)

        try:
            for i in range(n_gpus):
                children[i].join()
        except KeyboardInterrupt:
            # Main process received SIGINT (Early Stop from GUI).
            # Workers spawn separately and don't inherit the signal, so
            # we forward SIGINT to each worker so their EarlyStopSignalHandler
            # triggers the save-and-exit path.
            print("[TRAINING] Forwarding Early Stop signal to workers...")
            for child in children:
                if child.is_alive():
                    os.kill(child.pid, signal.SIGINT)
            # Wait for workers to finish saving (generous timeout)
            for child in children:
                child.join(timeout=180)
                if child.is_alive():
                    print(f"[TRAINING] Worker {child.pid} did not exit in time, terminating.")
                    child.terminate()

    if cleanup:
        old_session_cleanup(now_dir, model_name)
    start()

def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    total_epoch_count,
    epoch_save_frequency,
    save_weight_models,
    save_only_latest_net_models,
    config,
    device,
    device_id,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, warmup_completed, optimizer_choice, from_scratch

    # Set expandable CUDA memory segments before the first cudaMalloc.
    # Must be here (in the spawned subprocess, before any CUDA op) — the
    # module-level os.environ.setdefault() fires before CUDA is initialized,
    # but mp.Process(spawn) may already have allocated CUDA state by the time
    # user code runs.  The programmatic API call after dist init is the
    # authoritative fallback.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    stopper = EarlyStopSignalHandler()

    if 'warmup_completed' not in globals():
        warmup_completed = False

    # Initial print / session info for console
    print_init_setup(
        warmup_duration,
        rank,
        use_warmup,
        config,
        optimizer_choice,
        use_validation,
        lr_scheduler,
        exp_decay_gamma,
        use_kl_annealing,
        kl_annealing_cycle_duration,
        spectral_loss,
        adversarial_loss,
        vits_version,
        use_tstp
    )

    # Initial setup
    setup_env_and_distr(
        rank,
        n_gpus,
        device,
        device_id,
        config
    )

    # Programmatic API call to ensure expandable_segments is active after
    # dist.init_process_group (which performs the first real cudaMalloc).
    # The env-var set above is a hint; this call is authoritative.
    if device.type == "cuda":
        try:
            torch.cuda.memory.set_allocator_settings("expandable_segments:True")
        except Exception:
            pass  # PyTorch < 2.2 — env-var fallback still applies

    # Dataloading and loaders preparation
    train_loader, val_loader = prepare_dataloaders(
        config,
        n_gpus,
        rank,
        batch_size,
        use_validation,
        benchmark_mode
    )

    # Spk dim verif
    spk_dim = verify_spk_dim(config, model_info_path, experiment_dir, latest_checkpoint_path, rank, pretrainG)
    config.model.spk_embed_dim = spk_dim

    # Spectral loss init
    if spectral_loss == "L1 Mel Loss":
        fn_spectral_loss = torch.nn.L1Loss()
    elif spectral_loss == "Multi-Scale Mel Loss":
        fn_spectral_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
    elif spectral_loss == "Multi-Res STFT Loss":
        fn_spectral_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes = [1024, 2048, 4096],
            hop_sizes = [256, 512, 1024],
            win_lengths = [1024, 2048, 4096],
            window = "hann_window",
            scale = "mel",
            n_bins = 128,
            sample_rate = sample_rate,
            perceptual_weighting = True,
            device=device,
        )
    else:
        print("ERROR: Chosen spectral loss is undefined. Exiting.")
        sys.exit(1)

    # Hinge adversarial loss
    fn_hinge_loss = HingeAdversarialLoss() if adversarial_loss == "hinge" else None

    # Loading of models and optims
    net_g, net_d, optim_g, optim_d, epoch_str, global_step, gradscaler_dict, _, chouwa_balancer_state = load_models_and_optimizers(
        config,
        pretrainG,
        pretrainD,
        vocoder,
        use_checkpointing,
        randomized,
        sample_rate, 
        optimizer_choice,
        custom_lr_g,
        custom_lr_d,
        use_custom_lr, 
        total_epoch_count,
        train_loader,
        device,
        device_id,
        n_gpus,
        rank
    )

    # Number of sub-discriminators — used to normalize losses for ChouwaGAN
    # (prevents gradient scale explosion when more sub-discs are active).
    # Other vocoders use n_disc=1 (no change in behaviour).
    if vocoder == "ChouwaGAN":
        _actual_d = net_d.module if hasattr(net_d, 'module') else net_d
        n_disc = len(_actual_d.discriminators) if hasattr(_actual_d, 'discriminators') else 5
        # Add 1 for the UnivHD discriminator (replaces hf_disc)
        if hasattr(_actual_d, 'univhd'):
            n_disc += 1
    else:
        n_disc = 1

    # ChouwaGAN-exclusive: Adaptive Balancer + HF Recon Loss
    chouwa_balancer = None
    hf_recon_loss = None
    if vocoder == "ChouwaGAN":
        from rvc.train.chouwa_gan_training import AdaptiveBalancer, HighFrequencyReconstructionLoss
        
        chouwa_balancer = AdaptiveBalancer(ema_decay=0.99, skip_threshold=8.0, resume_threshold=3.0)
        
        # Load balancer state if resuming from checkpoint
        if chouwa_balancer_state is not None:
            chouwa_balancer.d_real_ema = chouwa_balancer_state.get("d_real_ema", 0.0)
            chouwa_balancer.d_fake_ema = chouwa_balancer_state.get("d_fake_ema", 0.0)
            chouwa_balancer._d_skipping = chouwa_balancer_state.get("_d_skipping", False)
            chouwa_balancer._warmup = chouwa_balancer_state.get("_warmup", 100)
            print("    ██████  ChouwaGAN: Loaded Adaptive Balancer state from checkpoint")
        
        # Use device_id for multi-GPU to avoid device mismatch
        target_device = device_id if device.type == "cuda" else device
        hf_recon_loss = HighFrequencyReconstructionLoss(sr=config.data.sample_rate).to(target_device)
        print("    ██████  ChouwaGAN: Adaptive D/G balancer active")
        print("    ██████  ChouwaGAN: HF Reconstruction Loss active")

    # Gradient clip ceiling for G — aligned with D so the adversarial
    # signal isn't overwhelmed after per-disc normalization.
    grad_clip_g = 10.0 if vocoder == "ChouwaGAN" else 200.0

    # Tensorboard handling
    if rank == 0:
        writer_eval = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval"),
            flush_secs=86400,
            purge_step=global_step
        )

        if use_trajectory:
            trajectory_tracker = WeightTrajectoryVisualizer(history_limit=100)
        else:
            trajectory_tracker = None

        block_tensorboard_flush_on_exit(writer_eval)

        if global_step != 0:
            print(f"[INIT] TensorBoard writer initialized. Purging logs after step: {global_step}")
        else:
            print(f"[INIT] TensorBoard writer initialized.")

    # from-scratch checker ( disables average loss )
    if pretrainG in ["", "None"] and pretrainD in ["", "None"]:
        from_scratch = True
        if rank == 0:
            print("[INIT] No pretrains used: Average loss disabled!")

    # Prepare the schedulers
    warmup_scheduler_g, warmup_scheduler_d, scheduler_g, scheduler_d = prepare_schedulers(
        optim_g,
        optim_d,
        use_warmup,
        warmup_duration,
        use_lr_scheduler, 
        lr_scheduler,
        exp_decay_gamma,
        total_epoch_count,
        epoch_str,
        global_step,
        train_loader
    )

    # Hann window for stft ( for RingFormer only. )
    hann_window = torch.hann_window(config.model.gen_istft_n_fft).to(device) if vocoder in ["RingFormer_v1", "RingFormer_v2"] else None

    # GradScaler for FP16 training
    gradscaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and train_dtype == torch.float16))
    if len(gradscaler_dict) > 0:
        gradscaler.load_state_dict(gradscaler_dict)
        print("[INIT] Loading gradscaler state dict - FP16")

    training_loop.encoders_frozen = False

    # torch.compile opt-in (requires PyTorch >= 2.0 and CUDA Ampere+)
    if use_compile and device.type == "cuda" and torch.cuda.is_available():
        # Suppress harmless GCC preprocessor warnings from Triton JIT
        # (_POSIX_C_SOURCE redefined between Python's pyconfig.h and glibc features.h)
        os.environ["CFLAGS"] = os.environ.get("CFLAGS", "") + " -Wno-cpp"
        try:
            if rank == 0:
                print(f"    ██████  torch.compile: mode='{compile_mode}' — compiling G and D...")
            _compile_opts = dict(mode=compile_mode, fullgraph=False, dynamic=True)
            if isinstance(net_g, torch.nn.parallel.DistributedDataParallel):
                net_g.module = torch.compile(net_g.module, **_compile_opts)
            else:
                net_g = torch.compile(net_g, **_compile_opts)
            if isinstance(net_d, torch.nn.parallel.DistributedDataParallel):
                net_d.module = torch.compile(net_d.module, **_compile_opts)
            else:
                net_d = torch.compile(net_d, **_compile_opts)
            if rank == 0:
                print("    ██████  torch.compile: done. First step will be slower (tracing).")
        except Exception as e:
            if rank == 0:
                print(f"    ██████  torch.compile: failed ({e}), continuing without it.")

    # Reference sample for live-infer
    reference = get_reference_sample(train_loader, device, config)

    # Cache for training with " cache " enabled
    cache = []

    def _shutdown_loader(loader):
        """Force DataLoader worker shutdown to avoid semaphore leaks on exit."""
        if loader is None:
            return
        try:
            it = getattr(loader, "_iterator", None)
            if it is not None:
                it._shutdown_workers()
                loader._iterator = None
        except Exception:
            pass

    try:
        for epoch in range(epoch_str, total_epoch_count + 1):
            should_stop = training_loop(
                rank,
                epoch,
                config,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                train_loader,
                val_loader if use_validation else None,
                [writer_eval],
                cache,
                total_epoch_count,
                epoch_save_frequency,
                save_weight_models,
                save_only_latest_net_models,
                device,
                device_id,
                reference,
                fn_spectral_loss,
                n_gpus,
                gradscaler,
                fn_hinge_loss,
                hann_window,
                stopper=stopper,
                trajectory_tracker=trajectory_tracker,
                n_disc=n_disc,
                grad_clip_g=grad_clip_g,
                chouwa_balancer=chouwa_balancer,
                hf_recon_loss=hf_recon_loss,
            )

            if use_warmup and epoch <= warmup_duration:
                if warmup_scheduler_g:
                    warmup_scheduler_g.step()
                if warmup_scheduler_d:
                    warmup_scheduler_d.step()

                # Logging of finished warmup
                if epoch == warmup_duration:
                    warmup_completed = True
                    print(f"    ██████  Warmup completed at epochs: {warmup_duration}")
                    print(f"    ██████  LR G: {optim_g.param_groups[0]['lr']}")
                    print(f"    ██████  LR D: {optim_d.param_groups[0]['lr']}")
                    # scheduler:
                    if lr_scheduler == "exp decay epoch":
                        print(f"    ██████  Starting the per-epoch exponential lr decay with gamma of {exp_decay_gamma}")
                    elif lr_scheduler == "cosine annealing epoch":
                        print("    ██████  Starting per-epoch cosine annealing scheduler " )

            if use_lr_scheduler and (not use_warmup or warmup_completed):
                # Once the warmup phase is completed, uses exponential lr decay
                if lr_scheduler in ["exp decay epoch", "cosine annealing epoch"]:
                    scheduler_g.step()
                    scheduler_d.step()
    finally:
        # Explicitly shut down DataLoader worker processes so their semaphores,
        # file descriptors and shared-memory segments are released before the
        # spawned subprocess exits. Without this, Python's resource_tracker
        # reports N leaked semaphores at shutdown (persistent_workers=True +
        # mp.Process is the common trigger).
        _shutdown_loader(train_loader)
        _shutdown_loader(val_loader)

def training_loop(
    rank,
    epoch,
    config,
    nets,
    optims,
    schedulers,
    train_loader,
    val_loader,
    writers,
    cache,
    total_epoch_count,
    epoch_save_frequency,
    save_weight_models,
    save_only_latest_net_models,
    device,
    device_id,
    reference,
    fn_spectral_loss,
    n_gpus,
    gradscaler,
    fn_hinge_loss=None,
    hann_window=None,
    stopper=None,
    trajectory_tracker=None,
    n_disc=1,
    grad_clip_g=200.0,
    chouwa_balancer=None,
    hf_recon_loss=None,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        config (object): Configuration object containing training parameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, net_d].
        train_loader: training dataloader.
        val_loader: validation dataloader.
        writers (list): List of TensorBoard writers [writer_eval].
        cache (list): List to cache data in GPU memory.
        total_epoch_count (int): The total number of epochs for training.
        epoch_save_frequency (int): Frequency of saving epochs.
        save_weight_models (int): Whether to save small weight models. 0 for no, 1 for yes.
        save_only_latest_net_models (int): Whether to save only latest G/D or for each epoch.
        device (torch.device): The device to use for training (CPU or GPU).
        reference (list): Contains reference sample. Either custom or from train loader.
        fn_spectral_loss: spectral loss;  can be l1, multi-scale or ms-stft.
        gradscaler: gradscaler for fp16
        hann_window: hann window used for RingFormer
    """
    global global_step, warmup_completed, use_lr_scheduler, lr_scheduler, use_warmup

    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers if schedulers is not None else (None, None)

    train_loader = train_loader if train_loader is not None else None
    if not benchmark_mode and use_validation:
        val_loader = val_loader if val_loader is not None else None

    if writers is not None:
        writer = writers[0]

    fn_hinge_loss = fn_hinge_loss if fn_hinge_loss is not None else None
    
    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()

    if not from_scratch:
        # Tensors init for averaged losses:
        # Tensors init for averaged losses:
        if vocoder == "ChouwaGAN":
            tensor_count = 7
        elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
            tensor_count = 7
        else:
            tensor_count = 6
        epoch_loss_tensor = torch.zeros(tensor_count, device=device)
        num_batches_in_epoch = 0

    avg_rolling_cache = {
        "grad_norm_d": deque(maxlen=rolling_loss_steps),
        "grad_norm_g": deque(maxlen=rolling_loss_steps),
        "loss_disc": deque(maxlen=rolling_loss_steps),
        "loss_adv": deque(maxlen=rolling_loss_steps),
        "loss_gen_total": deque(maxlen=rolling_loss_steps),
        "loss_fm": deque(maxlen=rolling_loss_steps),
        "loss_mel": deque(maxlen=rolling_loss_steps),
        "loss_kl": deque(maxlen=rolling_loss_steps),
    }
    if vocoder == "ChouwaGAN":
        avg_rolling_cache["loss_hf"] = deque(maxlen=rolling_loss_steps)
    elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
        avg_rolling_cache["loss_sd"] = deque(maxlen=rolling_loss_steps)

    # Constants computed once per epoch (not per batch)
    if vocoder == "ChouwaGAN":
        from rvc.train.chouwa_gan_training import get_chouwa_config
        _cc = get_chouwa_config(from_scratch)
        grad_clip_d, c_fm = _cc["grad_clip_d"], _cc["c_fm"]
        c_hf, c_mel = _cc["c_hf"], _cc["c_mel"]
        c_stft = _cc["c_stft"]
        r1_gamma, r1_interval = _cc["r1_gamma"], _cc["r1_interval"]
        d_real_label = _cc["d_real_label"]
        grad_accum_steps = _cc["grad_accum_steps"]
    else:
        grad_clip_d = 150.0
        c_fm = 1.0
        r1_gamma = 0.0
        r1_interval = 16
        d_real_label = 1.0
        grad_accum_steps = 1

    use_amp = (config.train.bf16_run or config.train.fp16_run) and device.type == "cuda"

    # Partial resume aligning
    current_epoch_start_step = (epoch - 1) * len(train_loader)
    start_batch_idx = global_step - current_epoch_start_step
    start_batch_idx = max(0, start_batch_idx)
    
    #with tqdm(total=len(train_loader), leave=False) as pbar:
    with tqdm(total=len(train_loader), leave=False, initial=start_batch_idx) as pbar:
        for batch_idx, info in data_iterator:

            # Catch up with previously processed steps if resuming from partial ckpts
            if batch_idx < start_batch_idx:
                continue

            global_step += 1

            if not from_scratch:
                num_batches_in_epoch += 1

            if device.type == "cuda":
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                y,
                y_lengths,
                sid,
            ) = info

            # Keep the full-length waveform before slicing so extra G steps
            # can re-slice with a fresh ids_slice from a new G forward.
            y_full = y

            # Generator forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                # Unpacking:
                if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (mag, phase) = (model_output)
                else:
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (model_output)

                # Slice the original waveform ( y ) to match the generated slice:
                if randomized:
                    y = commons.slice_segments(
                        y,
                        ids_slice * config.data.hop_length,
                        config.train.segment_size,
                        dim=3,
                    )
                    
                # Ensure exact length match. Interpolation-based generators
                # can output 1-2 frames fewer than expected target due to padding
                # differences vs ConvTranspose. Trim BOTH `y` and `y_hat` so ALL 
                # subsequent discriminator and loss calculations match precisely.
                min_len = min(y.shape[-1], y_hat.shape[-1])
                if y.shape[-1] > min_len:
                    y = y[..., :min_len]
                if y_hat.shape[-1] > min_len:
                    y_hat = y_hat[..., :min_len]

            if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                reshaped_y = y.view(-1, y.size(-1))
                reshaped_y_hat = y_hat.view(-1, y_hat.size(-1))
                y_stft = torch.stft(reshaped_y, n_fft=config.model.gen_istft_n_fft, hop_length=config.model.gen_istft_hop_size, win_length=config.model.gen_istft_n_fft, window=hann_window, return_complex=True)
                y_hat_stft = torch.stft(reshaped_y_hat, n_fft=config.model.gen_istft_n_fft, hop_length=config.model.gen_istft_hop_size, win_length=config.model.gen_istft_n_fft, window=hann_window, return_complex=True)
                target_magnitude = torch.abs(y_stft)  # shape: [B, F, T]

            # ── Discriminator step ──────────────────────────────────────────
            # No fmaps needed here — saves memory.
            with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                # Detach y_hat early to free generator's computation graph
                y_hat_detached = y_hat.detach()
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat_detached, compute_fmaps=False)

            with autocast(device_type="cuda", enabled=False):
                # Compute discriminator loss:
                d_real_mean, d_fake_mean = 0.0, 0.0
                if adversarial_loss == "softplus":
                    from rvc.train.chouwa_gan_training import softplus_d_loss
                    loss_disc, d_real_mean, d_fake_mean = softplus_d_loss(y_d_hat_r, y_d_hat_g)
                elif adversarial_loss == "lsgan":
                    loss_disc, d_real_mean, d_fake_mean = discriminator_loss(
                        [d.float() for d in y_d_hat_r],
                        [d.float() for d in y_d_hat_g],
                        real_label=d_real_label
                    )
                elif adversarial_loss == "tprls":
                    loss_disc = discriminator_loss_v2(y_d_hat_r, y_d_hat_g, real_label=d_real_label)
                elif adversarial_loss == "hinge":
                    loss_fake, loss_real = fn_hinge_loss(y_d_hat_g, y_d_hat_r)
                    loss_disc = loss_fake + loss_real
                # Normalize by sub-discriminator count
                loss_disc = loss_disc / n_disc

                loss_disc_val = loss_disc.detach()

                # Update adaptive balancer with D scores
                if chouwa_balancer is not None:
                    chouwa_balancer.update(d_real_mean, d_fake_mean, global_step)

                # R1 lazy gradient penalty (ChouwaGAN only)
                if r1_gamma > 0.0:
                    from rvc.train.chouwa_gan_training import r1_penalty_eager
                    loss_disc = r1_penalty_eager(
                        net_d.module if hasattr(net_d, 'module') else net_d,
                        y, y_hat_detached, loss_disc,
                        r1_gamma, r1_interval, global_step,
                    )

            # Discriminator backward and update (with gradient accumulation):
            # Adaptive balancing: skip D optim step if D is too confident
            skip_d = chouwa_balancer.should_skip_d(global_step) if chouwa_balancer else False
            loss_disc_for_backward = loss_disc / grad_accum_steps
            micro_step = ((global_step - 1) % grad_accum_steps) + 1
            if micro_step == 1:
                optim_d.zero_grad(set_to_none=True)
            # Separate step flags: D can be skipped, but G always updates when accumulated
            do_d_step = (micro_step == grad_accum_steps) and not skip_d
            do_g_step = (micro_step == grad_accum_steps)
            if train_dtype == torch.float16:
                gradscaler.scale(loss_disc_for_backward).backward()
                if do_d_step:
                    gradscaler.unscale_(optim_d)
                    scale = gradscaler.get_scale()
                    grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=grad_clip_d)
                    
                    if chouwa_balancer is not None:
                        _d_scale = chouwa_balancer.d_lr_scale()
                        for _pg in optim_d.param_groups:
                            _pg["lr"] *= _d_scale
                            
                    gradscaler.step(optim_d)
                    
                    if chouwa_balancer is not None:
                        for _pg in optim_d.param_groups:
                            _pg["lr"] /= _d_scale
                else:
                    scale = gradscaler.get_scale()
                    grad_norm_d = torch.tensor(0.0, device=device)
            else:
                loss_disc_for_backward.backward()
                if do_d_step:
                    grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=grad_clip_d)
                    
                    if chouwa_balancer is not None:
                        _d_scale = chouwa_balancer.d_lr_scale()
                        for _pg in optim_d.param_groups:
                            _pg["lr"] *= _d_scale
                            
                    optim_d.step()
                    
                    if chouwa_balancer is not None:
                        for _pg in optim_d.param_groups:
                            _pg["lr"] /= _d_scale
                else:
                    grad_norm_d = torch.tensor(0.0, device=device)
                scale = 1.0

            # Free D-step computation graph before G steps to reduce peak VRAM
            del y_d_hat_r, y_d_hat_g, loss_disc, y_hat_detached

            # Empty cache only when D step actually happens and for ChouwaGAN
            # ChouwaGAN: 6 sub-discs leave ~30-50 MB of reserved-but-unallocated
            # fragments after the D backward. The G forward (SnakeBeta resblocks)
            # then OOMs trying to allocate the next ~16 MB contiguous block even
            # with 60-70 MB "free".  empty_cache() returns those fragments to CUDA
            # before the G forward runs.  Cost: ~1-2 ms/step.
            # Other vocoders don't need this as frequently.
            if device.type == "cuda" and vocoder == "ChouwaGAN" and do_d_step:
                torch.cuda.empty_cache()

            # ── Generator step ──────────────────────────────────────────────
            # D provides adversarial + feature-matching signal; its parameter
            # gradients are never used inside the G step.  Freezing D prevents
            # PyTorch from allocating gradient memory for D's params during G
            # backward.  Unfreeze after for the next batch's D step.
            d_for_g = net_d.module if hasattr(net_d, 'module') else net_d
            d_for_g.requires_grad_(False)

            for _g_step in range(1):

                # Run discriminator on generated output (fmaps needed for G losses)
                with autocast(device_type="cuda", enabled=use_amp, dtype=train_dtype):
                    y_d_hat_r_g, y_d_hat_g, fmap_r, fmap_g = d_for_g(y, y_hat)
                    
                    # Detach real fmaps — G doesn't need gradients through real activations
                    # This saves 5-10% backward compute time with zero impact on quality
                    fmap_r = [[f.detach() for f in fm] for fm in fmap_r]

                # Compute generator losses:
                with autocast(device_type="cuda", enabled=False):

                    # Spectral loss (referenced as "loss_mel" for log consistency):
                    
                    # Ensure exact length match. Interpolation-based generators
                    # can output 1-2 frames fewer than expected target due to padding
                    # differences vs ConvTranspose. Trim the target 'y' to match 'y_hat'.
                    min_len = min(y.shape[-1], y_hat.shape[-1])
                    if y.shape[-1] > min_len:
                        y = y[..., :min_len]
                    if y_hat.shape[-1] > min_len:
                        y_hat = y_hat[..., :min_len]
                        
                    if spectral_loss == "L1 Mel Loss":
                        y_mel = wave_to_mel(config, y, half=train_dtype)
                        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)
                        
                        # Guarantee exact frame match for STFT loss. A single sample difference
                        # in the time domain (from interpolation padding) can occasionally result
                        # in a 1-frame difference in the Mel-spectrogram output.
                        min_mel_len = min(y_mel.shape[-1], y_hat_mel.shape[-1])
                        if y_mel.shape[-1] > min_mel_len:
                            y_mel = y_mel[..., :min_mel_len]
                        if y_hat_mel.shape[-1] > min_mel_len:
                            y_hat_mel = y_hat_mel[..., :min_mel_len]
                            
                        active_c_mel = c_mel if vocoder == "ChouwaGAN" else config.train.c_mel
                        loss_mel = fn_spectral_loss(y_mel, y_hat_mel) * active_c_mel
                    elif spectral_loss == "Multi-Scale Mel Loss":
                        active_c_mel = c_mel if vocoder == "ChouwaGAN" else config.train.c_mel
                        loss_mel = fn_spectral_loss(y, y_hat) * active_c_mel / 3.0
                    elif spectral_loss == "Multi-Res STFT Loss":
                        active_c_stft = 2.0 if vocoder == "ChouwaGAN" else c_stft
                        loss_mel = fn_spectral_loss(y_hat.float(), y.float()) * active_c_stft

                    # Feature Matching loss (normalized per sub-disc, 2× for ChouwaGAN)
                    loss_fm = feature_loss(
                        [[f.float() for f in fm] for fm in fmap_r],
                        [[f.float() for f in fm] for fm in fmap_g]
                    ) / n_disc * c_fm

                    # Generator loss (normalized per sub-discriminator)
                    if adversarial_loss == "softplus":
                        from rvc.train.chouwa_gan_training import softplus_g_loss
                        loss_adv = softplus_g_loss(y_d_hat_g, n_disc)
                    elif adversarial_loss == "lsgan":
                        loss_adv = generator_loss(
                            [d.float() for d in y_d_hat_g],
                            real_label=d_real_label
                        ) / n_disc
                    elif adversarial_loss == "tprls":
                        y_d_hat_r_detached = [i.detach() for i in y_d_hat_r_g]
                        loss_adv = generator_loss_v2(y_d_hat_g, y_d_hat_r_detached) / n_disc
                    elif adversarial_loss == "hinge":
                        loss_adv = fn_hinge_loss(y_d_hat_g) / n_disc

                    # Adaptive balancing: scale adv weight when D is too strong/weak
                    if chouwa_balancer is not None:
                        loss_adv = loss_adv * chouwa_balancer.adv_weight_scale()

                    # Kl annealing handler
                    if use_kl_annealing:
                        annealing_cycle_steps = len(train_loader) * kl_annealing_cycle_duration
                        kl_beta = 0.5 * (1 - math.cos((global_step % annealing_cycle_steps) * (math.pi / annealing_cycle_steps)))
                    else:
                        kl_beta = 1.0

                    # RingFormer related;  Phase, Magnitude and SD:
                    if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                        loss_magnitude = torch.nn.functional.l1_loss(mag, target_magnitude)
                        loss_phase = phase_loss(y_stft, y_hat_stft)
                        loss_sd = (loss_magnitude + loss_phase) * 0.7

                    # Total generator loss + kl ( encoders )
                    if not training_loop.encoders_frozen: # For when encoders aren't frozen yet
                        # KL loss — logger/exp in fp16 loses precision
                        loss_kl = kl_loss_clamped(
                            z_p.float(), logs_q.float(),
                            m_p.float(), logs_p.float(),
                            z_mask.float()
                        ) * config.train.c_kl # KL ( Kullback–Leibler divergence ) loss
                        if vocoder == "ChouwaGAN":
                            loss_hf = hf_recon_loss(y_hat, y) * c_hf
                            loss_gen_total = loss_adv + loss_fm + loss_mel + loss_kl * kl_beta + loss_hf
                        elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                            loss_gen_total = loss_adv + loss_fm + loss_mel + loss_kl * kl_beta + loss_sd
                        else:
                            loss_gen_total = loss_adv + loss_fm + loss_mel + loss_kl * kl_beta
                    else:
                        loss_kl = torch.tensor(0.0, device=device) # KL loss dummy for logs
                        if vocoder == "ChouwaGAN":
                            loss_hf = hf_recon_loss(y_hat, y) * c_hf
                            loss_gen_total = loss_adv + loss_fm + loss_mel + loss_hf
                        elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                            loss_gen_total = loss_adv + loss_fm + loss_mel + loss_sd
                        else:
                            loss_gen_total = loss_adv + loss_fm + loss_mel

                # Generator backward and update (with gradient accumulation):
                loss_gen_for_backward = loss_gen_total / grad_accum_steps
                if micro_step == 1:
                    optim_g.zero_grad(set_to_none=True)
                if train_dtype == torch.float16:
                    gradscaler.scale(loss_gen_for_backward).backward()
                    if do_g_step:
                        gradscaler.unscale_(optim_g)
                        grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=grad_clip_g)
                        gradscaler.step(optim_g)
                        gradscaler.update()
                        skip_lr_sched = (scale > gradscaler.get_scale())
                    else:
                        grad_norm_g = torch.tensor(0.0, device=device)
                        skip_lr_sched = False
                else:
                    loss_gen_for_backward.backward()
                    if do_g_step:
                        grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=grad_clip_g)
                        optim_g.step()
                    else:
                        grad_norm_g = torch.tensor(0.0, device=device)
                    skip_lr_sched = False

            # Unfreeze D for the next iteration's D step
            d_for_g.requires_grad_(True)

            # Per step exp lr decay for generator
            if not skip_lr_sched: # We skip lr scheduler step if there were nans / infs due to gradscaler's scaling.
                if use_lr_scheduler and (not use_warmup or warmup_completed) and lr_scheduler == "exp decay step":
                    scheduler_d.step()
                    scheduler_g.step()

            if not from_scratch:
                # Loss accumulation for epoch-avg
                epoch_loss_tensor[0].add_(loss_disc_val)
                epoch_loss_tensor[1].add_(loss_adv.detach())
                epoch_loss_tensor[2].add_(loss_gen_total.detach())
                epoch_loss_tensor[3].add_(loss_fm.detach())
                epoch_loss_tensor[4].add_(loss_mel.detach())
                epoch_loss_tensor[5].add_(loss_kl.detach())

                if vocoder == "ChouwaGAN":
                    epoch_loss_tensor[6].add_(loss_hf.detach())
                elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                    epoch_loss_tensor[6].add_(loss_sd.detach())

            # Loss accumulation for rolling-avg
            # Grads: only log when step actually happened (avoid polluting with zeros)
            if do_d_step and torch.isfinite(grad_norm_d):
                avg_rolling_cache["grad_norm_d"].append(grad_norm_d)
            elif not torch.isfinite(grad_norm_d):
                writer.add_scalar("Grad_Norm/D_Skipped", 1, global_step)

            if do_g_step and torch.isfinite(grad_norm_g):
                avg_rolling_cache["grad_norm_g"].append(grad_norm_g)
            elif not torch.isfinite(grad_norm_g):
                writer.add_scalar("Grad_Norm/G_Skipped", 1, global_step)

            # Losses:
            avg_rolling_cache["loss_disc"].append(loss_disc_val)
            avg_rolling_cache["loss_adv"].append(loss_adv.detach()) 
            avg_rolling_cache["loss_gen_total"].append(loss_gen_total.detach())
            avg_rolling_cache["loss_fm"].append(loss_fm.detach())
            avg_rolling_cache["loss_mel"].append(loss_mel.detach())
            avg_rolling_cache["loss_kl"].append(loss_kl.detach())
            if "loss_hf" in avg_rolling_cache:
                avg_rolling_cache["loss_hf"].append(loss_hf.detach())
            elif "loss_sd" in avg_rolling_cache:
                avg_rolling_cache["loss_sd"].append(loss_sd.detach())
            
            # Free G-step computation graph to reduce peak VRAM
            # IMPORTANT: Delete AFTER logging to avoid UnboundLocalError
            del y_d_hat_r_g, y_d_hat_g, fmap_r, fmap_g
            del loss_adv, loss_fm, loss_mel, loss_kl, loss_gen_total
            if vocoder == "ChouwaGAN":
                del loss_hf
            elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                del loss_sd, loss_phase, loss_mag


            if rank == 0 and global_step % rolling_loss_steps == 0:
                scalar_dict_rolling = {}

                # Learning rate retrieval for rolling logging
                if from_scratch:
                    scalar_dict_rolling.update({
                        "learning_rate/lr_d": optim_d.param_groups[0]["lr"],
                        "learning_rate/lr_g": optim_g.param_groups[0]["lr"],
                    })

                # logging rolling averages
                for key, queue in avg_rolling_cache.items():
                    if len(queue) > 0:
                        category = "loss" if "loss" in key else "grad"
                        label = f"{category}_avg_{rolling_loss_steps}/{key}_{rolling_loss_steps}"
                        val = torch.stack(list(queue)).mean().item() if torch.is_tensor(queue[0]) else sum(queue)/len(queue)
                        scalar_dict_rolling[label] = val

                # ChouwaGAN: log D(real)/D(fake) monitoring
                if chouwa_balancer is not None:
                    scalar_dict_rolling.update(chouwa_balancer.get_log_dict())

                summarize(writer=writer, global_step=global_step, scalars=scalar_dict_rolling)
                flush_writer(writer, rank)

            if from_scratch and pretrain_preview and rank == 0 and global_step % pretrain_preview_interval == 0:
                print(f"    ██████  Generating pretrain-preview at step: {global_step}...")
                o = eval_infer(net_g, reference)
                audio_dict = {f"gen/audio_pretrain_{global_step}s": o[0, :, :]}
                summarize(
                    writer=writer,
                    global_step=global_step,
                    audios=audio_dict,
                    audio_sample_rate=config.data.sample_rate,
                )
                flush_writer(writer, rank)
                torch.cuda.empty_cache()

            pbar.update(1)

            if early_stopper(
                stopper, rank, global_step, epoch, architecture, 
                [net_g, net_d], [optim_g, optim_d], config, 
                experiment_dir, gradscaler, save_weight_models,
                model_name, vocoder, vits_version, n_gpus,
                chouwa_balancer=chouwa_balancer,
            ):
                return True

        # end of batch train
    # end of tqdm

    if n_gpus > 1 and device.type == 'cuda':
        dist.barrier()

    with torch.no_grad():
        torch.cuda.empty_cache()

    # Logging and checkpointing
    if rank == 0:
        # Used for tensorboard chart - all/mel
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )

        # For fp16 we need to .half() the mel spec
        if train_dtype == torch.float16:
            mel = mel.half()

        # Used for tensorboard chart - slice/mel_org
        if randomized:
            y_mel = commons.slice_segments(
                mel,
                ids_slice,
                config.train.segment_size // config.data.hop_length,
                dim=3,
            )
        else:
            y_mel = mel

        # used for tensorboard chart - slice/mel_gen
        y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype)

        # Mel similarity metric:
        mel_similarity = mel_spec_similarity(y_hat_mel, y_mel)
        print(f'Mel Spectrogram Similarity: {mel_similarity:.2f}%')
        writer.add_scalar('Metric/Mel_Spectrogram_Similarity', mel_similarity, global_step)

        # Learning rate retrieval for avg-epoch variation:
        lr_d = optim_d.param_groups[0]["lr"]
        lr_g = optim_g.param_groups[0]["lr"]

        # At each epoch completion
        if global_step % len(train_loader) == 0 and not from_scratch:

            # Calculate the avg epoch loss:
            avg_epoch_loss = epoch_loss_tensor / num_batches_in_epoch

            # Two-Stage Training protocol ( Heavily Experimental )
            if use_tstp:
                KL_FREEZE_THRESHOLD = 0.1 # Freezing threshold   ( If all goes well with this experiment, I will add an UI handler for this. )
                current_avg_kl = avg_epoch_loss[5].item() # get the loss
                if not training_loop.encoders_frozen:
                    if current_avg_kl < KL_FREEZE_THRESHOLD:
                        if rank == 0:
                            print(f"\n[TSTP Phase 1]  KL Loss reached current threshold: {KL_FREEZE_THRESHOLD}")
                            print("[TSTP Phase 1]  Freezing Encoders, Flow and Spk emb. Only Decoder will train now.")

                        # Handle DDP wrapper ( if present )
                        model_ref = net_g.module if hasattr(net_g, "module") else net_g

                        # Freezing:  TextEncoder (enc_p), PosteriorEncoder (enc_q), Flow and spk embedding
                        modules_to_freeze = [model_ref.enc_p, model_ref.enc_q, model_ref.flow, model_ref.emb_g]

                        for module in modules_to_freeze:
                            for param in module.parameters():
                                param.requires_grad = False # No grads required = frozen.
                        
                        training_loop.encoders_frozen = True # Flag as frozen

                        # Accelerating the learning rate decay
                        for sched in [scheduler_g, scheduler_d]:
                            if sched is not None and hasattr(sched, 'gamma'):
                                old_gamma = sched.gamma # Get current gamma
                                sched.gamma = old_gamma ** 2 # Apply acceleration
                                if rank == 0:
                                    print(f"[TSTP Phase 2] Speeding up the learning rate decay by 50%")
                                    print(f"Gamma change: {old_gamma:.4f} ---> {sched.gamma:.4f}")

            # metrics dict
            scalar_dict_avg = {
            "loss_avg/loss_disc": avg_epoch_loss[0].item(),
            "loss_avg/loss_adv": avg_epoch_loss[1].item(),
            "loss_avg/loss_gen_total": avg_epoch_loss[2].item(),
            "loss_avg/loss_fm": avg_epoch_loss[3].item(),
            "loss_avg/loss_mel": avg_epoch_loss[4].item(),
            "loss_avg/loss_kl": avg_epoch_loss[5].item(),
            "learning_rate/lr_d": lr_d,
            "learning_rate/lr_g": lr_g,
            }
            if vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                scalar_dict_avg.update({"loss_avg/loss_sd": avg_epoch_loss[6].item()})
            elif vocoder == "ChouwaGAN":
                scalar_dict_avg.update({"loss_avg/loss_hf": avg_epoch_loss[6].item()})

            summarize(writer=writer, global_step=global_step, scalars=scalar_dict_avg)
            flush_writer(writer, rank)
            num_batches_in_epoch = 0
            epoch_loss_tensor.zero_()

        # Determine the plot data type
        if train_dtype == torch.float16:
            plot_dtype = torch.float16
        else:
            plot_dtype = torch.float32

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].detach().cpu().to(plot_dtype).numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].detach().cpu().to(plot_dtype).numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].detach().cpu().to(plot_dtype).numpy()),
        }

        # At each epoch save point:
        if epoch % epoch_save_frequency == 0:

            if trajectory_tracker is not None:
                # Update tracker with current model weights
                trajectory_tracker.update(net_g, epoch)
                # Get the PCA image
                traj_img = trajectory_tracker.get_plot()
                # Log to TensorBoard
                if traj_img is not None:
                    writer.add_image("Training/Weight_Trajectory", traj_img, global_step, dataformats='HWC')

            if not benchmark_mode:
                if use_validation:
                # Run validation
                    validation_loop(
                        net_g.module if hasattr(net_g, "module") else net_g,
                        val_loader,
                        device,
                        config,
                        writer,
                        global_step,
                    )
            # Inferencing on reference sample
            o = eval_infer(net_g, reference)
            audio_dict = {f"gen/audio_{epoch}e_{global_step}s": o[0, :, :]} # Eval-infer samples
            # Logging
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )
            flush_writer(writer, rank)
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
            )
            flush_writer(writer, rank)

    # Save checkpoint
    model_add = []
    done = False

    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)

        # Save weights every N epochs
        if epoch % epoch_save_frequency == 0:
            g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
            d_path = os.path.join(experiment_dir, f"D_{global_step}.pth")

            if save_only_latest_net_models:
                old_files = glob.glob(os.path.join(experiment_dir, "G_*.pth")) + glob.glob(os.path.join(experiment_dir, "D_*.pth"))
                for f in old_files:
                    try:
                        os.remove(f)
                    except:
                        pass

            # Save Generator checkpoint
            save_checkpoint(net_g, optim_g, config.train.learning_rate, epoch, g_path, gradscaler, chouwa_balancer)
            # Save Discriminator checkpoint
            save_checkpoint(net_d, optim_d, config.train.learning_rate, epoch, d_path, gradscaler)

            # Save small weight model
            if save_weight_models:
                weight_model_name = small_model_naming(model_name, epoch, global_step)
                model_add.append(os.path.join(experiment_dir, weight_model_name))

        # Check completion
        if epoch >= total_epoch_count:
            print(f"Training has been successfully completed with {epoch} epoch and {global_step} steps.")
            # Final model
            weight_model_name = small_model_naming(model_name, epoch, global_step)
            model_add.append(os.path.join(experiment_dir, weight_model_name))
            done = True

        if model_add:
            ckpt = (net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict())

            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=sample_rate,
                        name=model_name,
                        model_path=m,
                        epoch=epoch,
                        step=global_step,
                        hps=config,
                        vocoder=vocoder,
                        architecture=architecture,
                        vits_version=vits_version,
                    )
        if done:
            # Clean-up process IDs from memory
            pid_data["process_pids"].clear()  # Clear the PID list when done

            if rank == 0:
                writer.flush()
                writer.close()

            os._exit(0) #2333333

        with torch.no_grad():
            torch.cuda.empty_cache()

def validation_loop(net_g, val_loader, device, config, writer, global_step):
    net_g.eval()
    torch.cuda.empty_cache()

    total_mel_error = 0.0
    total_mrstft_loss = 0.0
    total_pesq = 0.0
    valid_pesq_count = 0
    total_si_sdr = 0.0
    count = 0

    mrstft = auraloss.freq.MultiResolutionSTFTLoss(device=device)
    resample_to_16k = torchaudio.transforms.Resample(orig_freq=config.data.sample_rate, new_freq=16000).to(device)

    hop_length = config.data.hop_length
    sample_rate = config.data.sample_rate

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, y, _, sid = [t.to(device) for t in batch]

        # Infer
            y_hat, x_mask, _ = net_g.infer(phone, phone_lengths, pitch, pitchf, sid)

        # Get reference min-length ( according to gt wave's length )
            y_len = y.shape[-1]

        # Obtaining mel specs
            y_hat_mel = wave_to_mel(config, y_hat, half=train_dtype) # generator-source mel
            mel = wave_to_mel(config, y, half=train_dtype) # gt-source mel

        # Mel loss:
            y_hat_mel_len = y_hat_mel.shape[-1]
            mel_len = mel.shape[-1]

            min_t = min(y_hat_mel_len, mel_len)

            mel_loss = F.l1_loss(y_hat_mel[..., :min_t], mel[..., :min_t])
            total_mel_error += mel_loss.item()

        # STFT loss:
            y_hat_len = y_hat.shape[-1]

            min_samples = min_t * hop_length
            min_samples = min(min_samples, y_len, y_hat_len)

            stft_loss = mrstft(y_hat[..., :min_samples], y[..., :min_samples])
            total_mrstft_loss += stft_loss.item()

        # si_sdr:
            si_sdr_score = si_sdr(y_hat.squeeze(1), y.squeeze(1))
            total_si_sdr += si_sdr_score.item()

        # PESQ:
            try:
                y_16k_batch = resample_to_16k(y).cpu().numpy()          # (B, T)
                y_hat_16k_batch = resample_to_16k(y_hat.squeeze(1)).cpu().numpy()  # (B, T)

                for i in range(y_16k_batch.shape[0]):
                    y_16k_f = np.squeeze(y_16k_batch[i]).astype(np.float32)
                    y_hat_16k_f = np.squeeze(y_hat_16k_batch[i]).astype(np.float32)

                    try:
                        pesq_score = pesq(16000, y_16k_f, y_hat_16k_f, mode="wb")
                        total_pesq += pesq_score
                        valid_pesq_count += 1
                    except Exception as e:
                        print(f"[PESQ skipped] {e}")

            except Exception as e:
                print(f"[PESQ skipped outer] {e}")

            count += 1

    avg_mel = total_mel_error / count
    avg_mrstft = total_mrstft_loss / count
    avg_pesq = total_pesq / max(valid_pesq_count, 1)
    avg_si_sdr = total_si_sdr / count

    if writer is not None:
        writer.add_scalar("validation/loss/mel_l1", avg_mel, global_step)
        writer.add_scalar("validation/loss/mrstft", avg_mrstft, global_step)
        writer.add_scalar("validation/score/pesq", avg_pesq, global_step)
        writer.add_scalar("validation/score/si_sdr", avg_si_sdr, global_step)

    net_g.train()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
