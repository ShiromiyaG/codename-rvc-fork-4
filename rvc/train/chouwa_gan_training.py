"""
ChouwaGAN Training Utilities

Dedicated module for ChouwaGAN-specific training optimizations:
- Softplus adversarial loss (StyleGAN2-style)
- Lazy R1 gradient penalty
- Adaptive D/G balancing
- High-frequency reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Training Constants
# ══════════════════════════════════════════════════════════════════════════════

CHOUWA_GRAD_CLIP_D = 10.0       # Discriminator gradient clipping threshold
CHOUWA_GRAD_CLIP_G = 150.0      # Generator gradient clipping threshold
CHOUWA_C_FM = 5.0               # Feature matching loss weight
CHOUWA_C_HF = 4.0               # High-frequency reconstruction loss weight
CHOUWA_R1_GAMMA = 0.0           # R1 penalty coefficient (disabled by default)
CHOUWA_R1_INTERVAL = 16         # R1 penalty application interval
CHOUWA_D_REAL_LABEL = 1.0       # Real label value for discriminator


# ══════════════════════════════════════════════════════════════════════════════
# Pre-emphasis Filter
# ══════════════════════════════════════════════════════════════════════════════

def pre_emphasis(x: torch.Tensor, coef: float = 0.97) -> torch.Tensor:
    """
    High-pass filter: y[n] = x[n] - coef * x[n-1].
    Amplifies high frequencies in spectral loss without computational cost.
    Zero-cost, in-place friendly operation.
    
    Args:
        x: Input waveform (B, 1, T) or (B, T)
        coef: Pre-emphasis coefficient (default: 0.97)
    
    Returns:
        Pre-emphasized waveform with same shape as input
    """
    return torch.cat([x[..., :1], x[..., 1:] - coef * x[..., :-1]], dim=-1)




# ══════════════════════════════════════════════════════════════════════════════
# Softplus Adversarial Losses
# ══════════════════════════════════════════════════════════════════════════════

def softplus_d_loss(y_d_hat_r, y_d_hat_g):
    """
    Discriminator softplus loss (StyleGAN2-style).
    
    Formulation:
        D_real: softplus(-D(real)) = log(1 + exp(-D(real)))
        D_fake: softplus(D(fake))  = log(1 + exp(D(fake)))
    
    Args:
        y_d_hat_r: List of discriminator scores for real samples
        y_d_hat_g: List of discriminator scores for generated samples
    
    Returns:
        loss_disc: Discriminator loss (not normalized by n_disc)
        d_real_mean: Mean D(real) score for monitoring
        d_fake_mean: Mean D(fake) score for monitoring
    """
    device = y_d_hat_r[0].device
    loss = torch.tensor(0.0, device=device)
    d_real_sum, d_fake_sum = 0.0, 0.0
    
    for dr, dg in zip(y_d_hat_r, y_d_hat_g):
        loss += F.softplus(-dr).mean() + F.softplus(dg).mean()
        d_real_sum += dr.detach().mean().item()
        d_fake_sum += dg.detach().mean().item()
    
    n = len(y_d_hat_r)
    return loss, d_real_sum / n, d_fake_sum / n


def softplus_g_loss(y_d_hat_g, n_disc):
    """
    Generator softplus loss.
    
    G wants to maximize D(fake) → minimize softplus(-D(fake)).
    
    Args:
        y_d_hat_g: List of discriminator scores for generated samples
        n_disc: Number of sub-discriminators (for normalization)
    
    Returns:
        loss: Normalized generator loss
    """
    device = y_d_hat_g[0].device
    loss = torch.tensor(0.0, device=device)
    
    for dg in y_d_hat_g:
        loss += F.softplus(-dg).mean()
    
    return loss / n_disc




# ══════════════════════════════════════════════════════════════════════════════
# Lazy R1 Gradient Penalty
# ══════════════════════════════════════════════════════════════════════════════

@torch._dynamo.disable(recursive=False)
def r1_penalty_eager(d_module, y_real, y_hat_d, loss_disc_in,
                     r1_gamma, r1_interval, d_update_idx):
    """
    R1 gradient penalty computed in eager mode (no torch.compile tracing).
    
    The @torch._dynamo.disable decorator prevents Dynamo from tracing this
    function, avoiding graph recompilation every r1_interval steps.
    
    Args:
        d_module: Discriminator module (unwrapped from DDP if needed)
        y_real: Real audio samples
        y_hat_d: Generated audio samples (detached)
        loss_disc_in: Base discriminator loss
        r1_gamma: R1 penalty coefficient
        r1_interval: Apply penalty every N steps
        d_update_idx: Current discriminator update index
    
    Returns:
        loss_disc: Discriminator loss with R1 penalty added
    """
    if r1_gamma <= 0.0 or d_update_idx % r1_interval != 0:
        return loss_disc_in

    # Get discriminator dtype
    try:
        _d_param = next(d_module.parameters())
        _d_dtype = _d_param.dtype
    except StopIteration:
        _d_dtype = y_real.dtype

    # Compute R1 penalty
    y_r1 = y_real.detach().to(_d_dtype).requires_grad_(True)
    _d_real_r1, _, _, _ = d_module(y_r1, y_hat_d.detach().to(_d_dtype), compute_fmaps=False)

    r1_grad = torch.autograd.grad(
        outputs=[p.float().mean() for p in _d_real_r1],
        inputs=y_r1,
        create_graph=False,
    )[0]

    r1_penalty = r1_grad.pow(2).reshape(r1_grad.shape[0], -1).mean(1).mean()
    
    # Cleanup
    del y_r1, _d_real_r1, r1_grad

    return loss_disc_in + (r1_gamma * r1_interval / 2.0) * r1_penalty




# ══════════════════════════════════════════════════════════════════════════════
# Adaptive D/G Balancer
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveBalancer:
    """
    Monitors D(real) and D(fake) scores and adaptively balances training.
    
    Tracks EMA of discriminator output scores. When D is too confident
    (large gap between D(real) and D(fake)), it:
    1. Skips D optimizer steps to slow D down
    2. Scales adversarial loss weight for G to prevent mode collapse
    
    The monitoring data (d_real, d_fake, gap) is logged to TensorBoard.
    """

    def __init__(self, ema_decay=0.99, skip_threshold=3.0, resume_threshold=1.5):
        """
        Args:
            ema_decay: Smoothing factor for score tracking
            skip_threshold: Gap (d_real - d_fake) above which D steps are skipped
            resume_threshold: Gap below which D steps resume
        """
        self.ema_decay = ema_decay
        self.skip_threshold = skip_threshold
        self.resume_threshold = resume_threshold

        self.d_real_ema = 0.0
        self.d_fake_ema = 0.0
        self._d_skipping = False  # Hysteresis flag
        self._warmup = 100        # Don't skip during first N steps

    def update(self, d_real_mean: float, d_fake_mean: float, global_step: int):
        """
        Update tracked scores with new batch values.
        
        Args:
            d_real_mean: Mean D output on real samples (from softplus_d_loss)
            d_fake_mean: Mean D output on fake samples (from softplus_d_loss)
            global_step: Current training step
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
        """
        Should we skip the D optimizer step this iteration?
        
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
        """
        Dynamic adversarial loss weight multiplier for G.
        
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




# ══════════════════════════════════════════════════════════════════════════════
# High Frequency Reconstruction Loss
# ══════════════════════════════════════════════════════════════════════════════

class HighFrequencyReconstructionLoss(nn.Module):
    """
    Reconstruction loss that weights high frequencies more heavily and includes
    phase information (instantaneous frequency deviation). Complements mel loss.
    
    Uses multi-scale STFT with separate weighting for:
    - Low frequency magnitude (normal weight)
    - High frequency magnitude (boosted weight)
    - High frequency phase (IFD - Instantaneous Frequency Deviation)
    
    Memory optimizations:
    - Reuses STFT computations when possible
    - Efficient tensor operations with channels_last format for Conv2D discriminators
    - Minimal intermediate allocations
    """
    
    def __init__(self, sr=48000, n_ffts=[2048, 1024, 512],
                 hf_start_hz=8000, hf_weight=10.0):
        """
        Args:
            sr: Sample rate
            n_ffts: List of FFT sizes for multi-scale analysis
            hf_start_hz: Frequency threshold for high-frequency region
            hf_weight: Weight multiplier for high-frequency components
        """
        super().__init__()
        self.n_ffts = n_ffts
        self.sr = sr
        self.hf_start_hz = hf_start_hz
        self.hf_weight = hf_weight
        
        # Pre-compute HF bin thresholds for each FFT size
        self.hf_bins = [int(hf_start_hz / (sr / n_fft)) for n_fft in n_ffts]
        
        # Register Hann windows for each FFT size
        for n_fft in self.n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))
    
    @torch._dynamo.disable(recursive=False)
    def forward(self, y_hat, y):
        """
        Args:
            y_hat: Generated audio (B, 1, T)
            y: Real audio (B, 1, T)
        
        Returns:
            loss: Combined multi-scale HF reconstruction loss
        """
        # Squeeze and convert to float32 once
        y_hat = y_hat.squeeze(1).float()
        y = y.squeeze(1).float()
        
        total_loss = 0.0
        
        for idx, n_fft in enumerate(self.n_ffts):
            hop = n_fft // 4
            window = getattr(self, f"window_{n_fft}")
            hf_bin = self.hf_bins[idx]
            
            # Compute STFT (float32 for cuFFT)
            Y = torch.stft(y, n_fft, hop, n_fft,
                          window=window, return_complex=True)
            Y_hat = torch.stft(y_hat, n_fft, hop, n_fft,
                              window=window, return_complex=True)
            
            # Compute magnitudes once
            mag_y = Y.abs()
            mag_y_hat = Y_hat.abs()
            
            # Low frequency magnitude loss (normal weight)
            loss_lf = F.l1_loss(mag_y_hat[:, :hf_bin, :], mag_y[:, :hf_bin, :])
            
            # High frequency magnitude loss (boosted weight)
            loss_hf = F.l1_loss(mag_y_hat[:, hf_bin:, :], mag_y[:, hf_bin:, :])
            
            # High frequency phase loss (Instantaneous Frequency Deviation)
            # To avoid catastrophic gradients from `torch.angle(z)` when |z| ≈ 0,
            # we compute IFD differences in the complex domain without explicit angles.
            Y_hf = Y[:, hf_bin:, :]
            Y_hat_hf = Y_hat[:, hf_bin:, :]
            
            # 1. Normalize complex vectors to unit circle (with epsilon)
            Y_hf_norm = Y_hf / (mag_y[:, hf_bin:, :] + 1e-8)
            Y_hat_hf_norm = Y_hat_hf / (mag_y_hat[:, hf_bin:, :] + 1e-8)
            
            # 2. Compute instantaneous frequency (phase difference between adjacent time steps)
            # Y(t) * conj(Y(t-1)) gives a complex number whose angle is phase(t) - phase(t-1)
            ifd_complex_y = Y_hf_norm[..., 1:] * Y_hf_norm[..., :-1].conj()
            ifd_complex_hat = Y_hat_hf_norm[..., 1:] * Y_hat_hf_norm[..., :-1].conj()
            
            # 3. Compute error between IFDs
            # ifd_hat * conj(ifd_y) gives a complex number representing the IFD error
            ifd_error_complex = ifd_complex_hat * ifd_complex_y.conj()
            
            # 4. Phase loss = 1 - cos(phase_error)
            # The real part of a unit complex number is cos(angle)
            loss_phase_hf = (1.0 - ifd_error_complex.real).mean()
            
            # Perceptual weighting: weight by frequency importance
            # Higher frequencies are perceptually less important, so we can
            # slightly reduce their weight while still maintaining quality
            freq_weight = 1.0 - (idx / len(self.n_ffts)) * 0.2  # 1.0 -> 0.8
            
            # Accumulate losses for this scale with perceptual weighting
            # Increased phase weight slightly since 1-cos(theta) has a smaller range than |theta|
            total_loss += freq_weight * (loss_lf + self.hf_weight * loss_hf + 3.0 * loss_phase_hf)
            
            # Free memory immediately
            del Y, Y_hat, mag_y, mag_y_hat, Y_hf, Y_hat_hf, Y_hf_norm, Y_hat_hf_norm
            del ifd_complex_y, ifd_complex_hat, ifd_error_complex
        
        return total_loss / len(self.n_ffts)




# ══════════════════════════════════════════════════════════════════════════════
# Configuration Helper
# ══════════════════════════════════════════════════════════════════════════════

def get_chouwa_config(from_scratch: bool) -> dict:
    """
    Return all ChouwaGAN-specific training hyperparameters.
    
    Args:
        from_scratch: Whether training from scratch (affects gradient accumulation)
    
    Returns:
        dict: Configuration dictionary with all ChouwaGAN hyperparameters
    """
    return {
        "grad_clip_d": CHOUWA_GRAD_CLIP_D,
        "c_fm": CHOUWA_C_FM,
        "c_hf": CHOUWA_C_HF,
        "r1_gamma": CHOUWA_R1_GAMMA,
        "r1_interval": CHOUWA_R1_INTERVAL,
        "d_real_label": CHOUWA_D_REAL_LABEL,
        "grad_accum_steps": 2 if from_scratch else 1,
    }
