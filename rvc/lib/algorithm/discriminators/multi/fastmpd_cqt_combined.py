import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE
from rvc.lib.algorithm.discriminators.single.mpd_discriminator_fast import FastPD
from rvc.lib.algorithm.discriminators.single.mssbcqt_discriminator import DiscriminatorCQT

from typing import List, Tuple


class FastMPD_CQT_Combined(nn.Module):
    """Optimized discriminator for ChouwaGAN combining:

    - N× FastPD  (Fast Period Discriminator via LYNXNet2) — periodic structure, ~50% lighter than Conv2d MPD
    - N× CQT  (Constant-Q Transform Discriminator) — log-frequency harmonic analysis

    Two fully orthogonal signal domains (periodic temporal, log-frequency)
    minimize redundancy and mode collapse risk while remaining lightweight
    and easy to calibrate.
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        sample_rate: int = 40000,
        **cfg,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # ── FastMPD config ──────────────────────────────────────────
        periods = cfg.get("periods", [2, 3, 5])
        fast_mpd_init_channel = cfg.get("fast_mpd_init_channel", 16)
        fast_mpd_strides = cfg.get("fast_mpd_strides", [4, 4, 4])
        fast_mpd_kernel_size = cfg.get("fast_mpd_kernel_size", 11)

        # ── CQT config ─────────────────────────────────────────────
        cqt_cfg = {
            "cqtd_filters": cfg.get("cqtd_filters", 32),
            "cqtd_max_filters": cfg.get("cqtd_max_filters", 1024),
            "cqtd_filters_scale": cfg.get("cqtd_filters_scale", 1),
            "cqtd_dilations": cfg.get("cqtd_dilations", [1, 2, 4]),
            "cqtd_in_channels": cfg.get("cqtd_in_channels", 1),
            "cqtd_out_channels": cfg.get("cqtd_out_channels", 1),
            "cqtd_normalize_volume": cfg.get("cqtd_normalize_volume", False),
        }
        cqtd_hop_lengths = cfg.get("cqtd_hop_lengths", [512, 256])
        cqtd_n_octaves = cfg.get("cqtd_n_octaves", [9, 9])
        cqtd_bins_per_octaves = cfg.get("cqtd_bins_per_octaves", [24, 36])

        # ── Build unified discriminator list ────────────────────────
        self.discriminators = nn.ModuleList(
            # N× FastPD (one per period)
            [
                FastPD(p, fast_mpd_init_channel, fast_mpd_strides, fast_mpd_kernel_size)
                for p in periods
            ]
            # N× CQT (one per scale)
            + [
                DiscriminatorCQT(
                    cqt_cfg,
                    hop_length=cqtd_hop_lengths[i],
                    n_octaves=cqtd_n_octaves[i],
                    bins_per_octave=cqtd_bins_per_octaves[i],
                    sample_rate=sample_rate,
                    use_checkpointing=use_checkpointing,
                )
                for i in range(len(cqtd_hop_lengths))
            ]
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        compute_fmaps: bool = True,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """Forward pass through all sub-discriminators.

        Args:
            y:     real waveform  (B, 1, T)
            y_hat: generated waveform  (B, 1, T)
            compute_fmaps: If False, skip collecting feature maps (saves VRAM
                           during the D training step where fmaps aren't needed).

        Returns:
            y_d_rs:  list of real predictions   (one per sub-discriminator)
            y_d_gs:  list of generated predictions
            fmap_rs: list of real feature-map lists  (empty lists when compute_fmaps=False)
            fmap_gs: list of generated feature-map lists
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            if self.training and self.use_checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            if compute_fmaps:
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

