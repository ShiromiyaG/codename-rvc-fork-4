"""
ChouwaGAN-exclusive training utilities.

All ChouwaGAN-specific optimizations live here so train.py stays clean.
Includes: softplus loss, lazy R1, EMA, adaptive D/G balancing, monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# ─── Constants ────────────────────────────────────────────────────────────────

CHOUWA_GRAD_CLIP_D = 10.0
CHOUWA_GRAD_CLIP_G = 150.0  # same as default, kept explicit
CHOUWA_C_FM = 6.0           # feature matching weight boost
CHOUWA_C_HF = 5.0           # weight for high-frequency reconstruction loss
CHOUWA_R1_GAMMA = 0.0          # disabled: LSGAN + light D doesn't need R1
CHOUWA_R1_INTERVAL = 16
CHOUWA_D_REAL_LABEL = 1.0   # no label smoothing (irrelevant for softplus)


# ─── Softplus (logistic) adversarial losses ───────────────────────────────────

def softplus_d_loss(y_d_hat_r, y_d_hat_g):
    """Discriminator softplus loss (StyleGAN2-style).

    D_real: softplus(-D(real)) = log(1 + exp(-D(real)))
    D_fake: softplus(D(fake))  = log(1 + exp(D(fake)))

    Returns:
        loss_disc: scalar tensor (NOT normalized by n_disc — caller does that)
        d_real_mean: mean D(real) score (for monitoring)
        d_fake_mean: mean D(fake) score (for monitoring)
    """
    loss = torch.tensor(0.0, device=y_d_hat_r[0].device)
    d_real_sum = 0.0
    d_fake_sum = 0.0
    for dr, dg in zip(y_d_hat_r, y_d_hat_g):
        loss = loss + F.softplus(-dr).mean() + F.softplus(dg).mean()
        d_real_sum += dr.detach().mean().item()
        d_fake_sum += dg.detach().mean().item()
    n = len(y_d_hat_r)
    return loss, d_real_sum / n, d_fake_sum / n


def softplus_g_loss(y_d_hat_g, n_disc):
    """Generator softplus loss.

    G wants to maximize D(fake) → minimize softplus(-D(fake)).
    Returns scalar tensor normalized by n_disc.
    """
    loss = torch.tensor(0.0, device=y_d_hat_g[0].device)
    for dg in y_d_hat_g:
        loss = loss + F.softplus(-dg).mean()
    return loss / n_disc


# ─── Lazy R1 gradient penalty ─────────────────────────────────────────────────

@torch._dynamo.disable(recursive=False)
def r1_penalty_eager(d_module, y_real, y_hat_d, loss_disc_in,
                     r1_gamma, r1_interval, d_update_idx):
    """R1 gradient penalty computed fully in eager mode (no torch.compile tracing).

    The @torch._dynamo.disable decorator prevents Dynamo from tracing this
    function, avoiding graph recompilation every r1_interval steps.
    """
    if r1_gamma <= 0.0 or d_update_idx % r1_interval != 0:
        return loss_disc_in

    try:
        _d_param = next(d_module.parameters())
        _d_dtype = _d_param.dtype
    except StopIteration:
        _d_dtype = y_real.dtype

    y_r1 = y_real.detach().to(_d_dtype).requires_grad_(True)
    _d_real_r1, _, _, _ = d_module(y_r1, y_hat_d.detach().to(_d_dtype), compute_fmaps=False)

    r1_grad = torch.autograd.grad(
        outputs=[p.float().mean() for p in _d_real_r1],
        inputs=y_r1,
        create_graph=False,
    )[0]

    r1_penalty = r1_grad.pow(2).reshape(r1_grad.shape[0], -1).mean(1).mean()
    del y_r1, _d_real_r1, r1_grad

    return loss_disc_in + (r1_gamma * r1_interval / 2.0) * r1_penalty


# ─── Generator EMA ────────────────────────────────────────────────────────────

class GeneratorEMA:
    """Exponential Moving Average of generator weights.

    Keeps a shadow copy of G parameters updated with EMA after each G step.
    Use shadow weights for inference/validation to get smoother, more stable output.

    VRAM note: shadow weights stored on same device as model (no extra gradients).
    Cost ≈ 1× parameter memory (typically 30-60 MB for ChouwaGAN G).
    """

    def __init__(self, generator, decay=0.999, device=None):
        self.decay = decay
        self.device = device
        # Deep copy the generator state_dict (parameters only, no grad buffers)
        model = generator.module if hasattr(generator, 'module') else generator
        self.shadow = {
            name: param.data.clone().to(device or param.device)
            for name, param in model.named_parameters()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, generator):
        """Update shadow weights toward current weights."""
        self.num_updates += 1
        # Warmup: ramp decay from 0 to target over first 1000 steps
        # so early noisy weights don't pollute the EMA too much.
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        model = generator.module if hasattr(generator, 'module') else generator
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.data.to(self.shadow[name].device), 1.0 - decay)

    @torch.no_grad()
    def apply_shadow(self, generator):
        """Replace generator weights with EMA weights. Call restore() after."""
        model = generator.module if hasattr(generator, 'module') else generator
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))

    @torch.no_grad()
    def restore(self, generator):
        """Restore original weights after apply_shadow()."""
        model = generator.module if hasattr(generator, 'module') else generator
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        """For checkpoint saving."""
        return {
            'shadow': self.shadow,
            'num_updates': self.num_updates,
            'decay': self.decay,
        }

    def load_state_dict(self, state):
        """For checkpoint loading."""
        self.shadow = state['shadow']
        self.num_updates = state.get('num_updates', 0)
        self.decay = state.get('decay', self.decay)


# ─── Adaptive D/G Balancer ────────────────────────────────────────────────────

class AdaptiveBalancer:
    """Monitors D(real) and D(fake) scores and adaptively balances training.

    Tracks EMA of discriminator output scores. When D is too confident
    (large gap between D(real) and D(fake)), it:
    1. Skips D optimizer steps to slow D down
    2. Scales adversarial loss weight for G to prevent mode collapse

    The monitoring data (d_real, d_fake, gap) is logged to TensorBoard.
    """

    def __init__(self, ema_decay=0.99, skip_threshold=3.0, resume_threshold=1.5):
        """
        Args:
            ema_decay: smoothing factor for score tracking
            skip_threshold: gap (d_real - d_fake) above which D steps are skipped
            resume_threshold: gap below which D steps resume
        """
        self.ema_decay = ema_decay
        self.skip_threshold = skip_threshold
        self.resume_threshold = resume_threshold

        self.d_real_ema = 0.0
        self.d_fake_ema = 0.0
        self._d_skipping = False  # hysteresis flag
        self._warmup = 100        # don't skip during first N steps

    def update(self, d_real_mean: float, d_fake_mean: float, global_step: int):
        """Update tracked scores with new batch values.

        Args:
            d_real_mean: mean D output on real samples (from softplus_d_loss)
            d_fake_mean: mean D output on fake samples (from softplus_d_loss)
            global_step: current training step
        """
        if global_step <= self._warmup:
            # During warmup, use direct values (no EMA lag)
            self.d_real_ema = d_real_mean
            self.d_fake_ema = d_fake_mean
        else:
            self.d_real_ema = self.ema_decay * self.d_real_ema + (1 - self.ema_decay) * d_real_mean
            self.d_fake_ema = self.ema_decay * self.d_fake_ema + (1 - self.ema_decay) * d_fake_mean

    @property
    def gap(self):
        """Current D confidence gap. Large = D dominating."""
        return self.d_real_ema - self.d_fake_ema

    def should_skip_d(self, global_step: int) -> bool:
        """Should we skip the D optimizer step this iteration?

        Uses hysteresis to avoid rapid on/off switching:
        - Start skipping when gap > skip_threshold
        - Stop skipping when gap < resume_threshold
        """
        if global_step <= self._warmup:
            return False

        if self._d_skipping:
            if self.gap < self.resume_threshold:
                self._d_skipping = False
        else:
            if self.gap > self.skip_threshold:
                self._d_skipping = True

        return self._d_skipping

    def adv_weight_scale(self) -> float:
        """Dynamic adversarial loss weight multiplier for G.

        When D is very confident, boost the adversarial signal to G.
        When D is weak, reduce it to let reconstruction losses dominate.
        Returns a multiplier in [0.5, 2.0].
        """
        gap = self.gap
        if gap > 2.0:
            # D too strong → boost adv signal to G (up to 2×)
            return min(2.0, 1.0 + (gap - 2.0) * 0.25)
        elif gap < 0.5:
            # D too weak → reduce adv signal
            return max(0.5, 0.5 + gap)
        return 1.0

    def get_log_dict(self):
        """Return dict of monitoring values for TensorBoard logging."""
        return {
            "chouwa/d_real_ema": self.d_real_ema,
            "chouwa/d_fake_ema": self.d_fake_ema,
            "chouwa/d_gap": self.gap,
            "chouwa/adv_weight_scale": self.adv_weight_scale(),
            "chouwa/d_skipping": float(self._d_skipping),
        }


# ─── High Frequency Reconstruction Loss ───────────────────────────────────────

class HighFrequencyReconstructionLoss(nn.Module):
    """
    Reconstruction loss that weights high frequencies more heavily and includes
    phase information (instantaneous frequency deviation). Complements mel loss.
    """
    def __init__(self, sr=48000, n_ffts=[2048, 1024, 512],
                 hf_start_hz=8000, hf_weight=10.0):
        super().__init__()
        self.n_ffts = n_ffts
        self.sr = sr
        self.hf_start_hz = hf_start_hz
        self.hf_weight = hf_weight
        
        for n_fft in self.n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))
    
    @torch._dynamo.disable(recursive=False)
    def forward(self, y_hat, y):
        """
        y_hat: generated audio (B, 1, T)
        y:     real audio      (B, 1, T)
        """
        loss = torch.tensor(0.0, device=y.device)
        y_hat = y_hat.squeeze(1).float()   # cuFFT requires float32
        y = y.squeeze(1).float()
        
        for n_fft in self.n_ffts:
            hop = n_fft // 4
            window = getattr(self, f"window_{n_fft}").to(dtype=y.dtype, device=y.device)
            
            # STFT (float32 — cuFFT requirement)
            Y = torch.stft(y, n_fft, hop, n_fft,
                          window=window, return_complex=True)
            Y_hat = torch.stft(y_hat, n_fft, hop, n_fft,
                              window=window, return_complex=True)
            
            mag_y = Y.abs()
            mag_y_hat = Y_hat.abs()
            
            hf_bin = int(self.hf_start_hz / (self.sr / n_fft))
            
            # LF magnitude loss (normal)
            loss_lf = F.l1_loss(
                mag_y_hat[:, :hf_bin, :],
                mag_y[:, :hf_bin, :]
            )
            
            # HF magnitude loss (boosted)
            loss_hf = F.l1_loss(
                mag_y_hat[:, hf_bin:, :],
                mag_y[:, hf_bin:, :]
            )
            
            # HF Phase loss (IFD)
            phase_y = torch.angle(Y[:, hf_bin:, :])
            phase_y_hat = torch.angle(Y_hat[:, hf_bin:, :])
            
            ifd_y = torch.diff(phase_y, dim=-1)
            ifd_y_hat = torch.diff(phase_y_hat, dim=-1)
            
            phase_diff = ifd_y_hat - ifd_y
            phase_diff = torch.atan2(
                torch.sin(phase_diff), torch.cos(phase_diff)
            )
            loss_phase_hf = phase_diff.abs().mean()
            
            loss = loss + loss_lf + self.hf_weight * loss_hf + 2.0 * loss_phase_hf
            
        return loss / len(self.n_ffts)


# ─── Config helper ────────────────────────────────────────────────────────────

def get_chouwa_config(from_scratch: bool) -> dict:
    """Return all ChouwaGAN-specific training hyperparameters."""
    return {
        "grad_clip_d": CHOUWA_GRAD_CLIP_D,
        "c_fm": CHOUWA_C_FM,
        "c_hf": CHOUWA_C_HF,
        "r1_gamma": CHOUWA_R1_GAMMA,
        "r1_interval": CHOUWA_R1_INTERVAL,
        "d_real_label": CHOUWA_D_REAL_LABEL,
        "grad_accum_steps": 2 if from_scratch else 1,
    }
