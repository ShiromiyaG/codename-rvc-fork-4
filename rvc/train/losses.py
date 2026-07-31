"""Losses used by the Mel-VITS acoustic model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def kl_loss_fb(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
    z_p2: torch.Tensor | None = None,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Flow-aware KL with optional two-sample variance reduction/free bits."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    if z_p2 is not None:
        z_p2 = z_p2.float()

    def term(sample: torch.Tensor) -> torch.Tensor:
        return (
            logs_p
            - logs_q
            - 0.5
            + 0.5 * ((sample - m_p) ** 2) * torch.exp(-2 * logs_p)
        )

    kl = term(z_p)
    if z_p2 is not None:
        kl = (kl + term(z_p2)) * 0.5
    kl = kl * z_mask
    valid = z_mask.sum(dim=(0, 2)).clamp_min(1.0)
    per_dimension = kl.sum(dim=(0, 2)) / valid
    return per_dimension.clamp_min(free_bits / z_p.size(1)).sum()


class MultiResolutionSTFTLoss(nn.Module):
    """Stable waveform-domain loss used through the frozen pc-NSF vocoder."""

    def __init__(self, resolutions=None):
        super().__init__()
        self.resolutions = resolutions or (
            (1024, 256, 1024),
            (2048, 512, 2048),
            (4096, 1024, 4096),
        )

    @staticmethod
    def _magnitude(waveform, n_fft, hop, win):
        window = torch.hann_window(win, device=waveform.device, dtype=torch.float32)
        spectrum = torch.stft(
            waveform.float(),
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
            center=True,
        )
        return spectrum.abs().clamp_min(1e-7)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        predicted = predicted.reshape(-1, predicted.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        total = predicted.new_zeros((), dtype=torch.float32)
        for n_fft, hop, win in self.resolutions:
            pred_mag = self._magnitude(predicted, n_fft, hop, win)
            target_mag = self._magnitude(target, n_fft, hop, win)
            difference = torch.linalg.vector_norm(target_mag - pred_mag)
            convergence = difference / torch.linalg.vector_norm(target_mag).clamp_min(1e-7)
            log_magnitude = F.l1_loss(pred_mag.log(), target_mag.log())
            total = total + convergence + log_magnitude
        return total / len(self.resolutions)
