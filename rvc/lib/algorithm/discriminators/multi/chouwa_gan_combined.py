"""
ChouwaGAN Combined Discriminator for RVC.

Wraps the ChouwaGAN discriminator (MS-STFT + FastMPD + UnivHD) from KazeFlow
as a multi-discriminator entry compatible with RVC's training pipeline.
"""

import torch
from rvc.lib.algorithm.discriminators.single.chouwa_gan_discriminator import (
    ChouwaGANDiscriminator,
)


class ChouwaGAN_Combined(torch.nn.Module):
    """
    Combined MS-STFT + FastMPD + UnivHD discriminator for RVC.
    Default: 2 STFT + 3 MPD + 1 UnivHD = 6 sub-discriminators.

    Args:
        sample_rate (int): Audio sample rate.
        use_spectral_norm (bool): Use spectral norm instead of weight norm.
        use_san (bool): Use Slicing Adversarial Network (L2-normalized projections).
        use_checkpointing (bool): Unused, kept for API compatibility.
        **cfg: Additional config for ChouwaGANDiscriminator.
    """

    def __init__(
        self,
        sample_rate: int = 40000,
        use_spectral_norm: bool = False,
        use_san: bool = True,
        use_checkpointing: bool = False,
        **cfg,
    ):
        super().__init__()
        self.disc = ChouwaGANDiscriminator(
            use_spectral_norm=use_spectral_norm,
            use_san=use_san,
            sample_rate=sample_rate,
            **cfg,
        )

    def forward(self, y, y_hat):
        """
        Args:
            y:     (B, 1, T) real waveform
            y_hat: (B, 1, T) generated waveform

        Returns:
            y_d_rs:  list of score tensors for real
            y_d_gs:  list of score tensors for generated
            fmap_rs: list of feature map lists for real
            fmap_gs: list of feature map lists for generated
        """
        return self.disc(y, y_hat)
