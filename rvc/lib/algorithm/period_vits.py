"""
Period VITS: Frame Pitch Predictor for RVC.

Adapted from "Period VITS: Variational Inference with Explicit Pitch Modeling
for End-to-End Emotional Speech Synthesis" (Shirahata et al., ICASSP 2023).

In the original paper, the Frame Pitch Predictor (FPP) predicts frame-level F0
and voicing flags from the prior encoder's hidden representation. This forces
the latent space to carry continuous pitch information, creating structural
dependency between the latents and pitch — preventing KL collapse even with
strong discriminators.

For RVC voice conversion, ground-truth F0 is always available (extracted from
source audio), so the FPP acts purely as a training-time regularizer. At
inference, the existing F0 pipeline is used unchanged.

The periodicity generator (sample-level sinusoidal source) is already present
in this codebase via the NSF source modules in the vocoder generators.
"""

import torch
import torch.nn as nn


class FramePitchPredictor(nn.Module):
    """
    Predicts continuous F0 and voiced/unvoiced flags from the prior encoder's
    latent representation. Operates on m_p (prior mean) after the text encoder.

    Architecture: 4-layer Conv1d stack → dual projection heads (F0, V/UV).

    Args:
        in_channels: Input channels (matches inter_channels / hidden_channels, e.g. 192).
        hidden_channels: Hidden dimension for internal conv layers.
        n_layers: Number of convolutional layers in the predictor.
        kernel_size: Kernel size for Conv1d layers.
        p_dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 192,
        n_layers: int = 4,
        kernel_size: int = 5,
        p_dropout: float = 0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            ch_in = in_channels if i == 0 else hidden_channels
            self.convs.append(
                nn.Conv1d(
                    ch_in,
                    hidden_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.lrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p_dropout)

        # F0 head: predicts log-F0 (continuous, unbounded)
        self.f0_proj = nn.Conv1d(hidden_channels, 1, 1)

        # V/UV head: predicts voicing probability
        self.vuv_proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        """
        Args:
            x: Prior mean from text encoder [B, C, T].
            x_mask: Sequence mask [B, 1, T].

        Returns:
            log_f0_pred: Predicted log-F0 [B, 1, T].
            vuv_pred: Predicted voicing probability [B, 1, T] (pre-sigmoid logits).
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x * x_mask)
            # LayerNorm expects [B, T, C]
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = self.lrelu(x)
            x = self.dropout(x)

        x = x * x_mask

        log_f0_pred = self.f0_proj(x) * x_mask
        vuv_pred = self.vuv_proj(x) * x_mask

        return log_f0_pred, vuv_pred
