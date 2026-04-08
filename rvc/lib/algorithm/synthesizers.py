import torch
from typing import Optional, List
import random

from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments
from rvc.lib.algorithm.normalizing_flows import ResidualCouplingBlock, ResidualCouplingTransformersBlock
from rvc.lib.algorithm.encoders import PosteriorEncoder # Posterior encoder, shared between Vits1 and Vits2
from rvc.lib.algorithm.encoders_vits2 import TextEncoder_VITS2
from rvc.lib.algorithm.encoders import TextEncoder as TextEncoder_VITS1
from rvc.lib.algorithm.period_vits import FramePitchPredictor


debug_shapes = False


class Synthesizer(torch.nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        checkpointing: bool = False,
        # Other
        vits2_mode: bool = False,
        # Period VITS
        use_period_vits: bool = False,
        # RingFormer
        gen_istft_n_fft: int = 120,
        gen_istft_hop_size: int = 30,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.use_f0 = use_f0
        self.vocoder = vocoder
        self.vits2_mode = vits2_mode
        self.use_period_vits = use_period_vits

        if vits2_mode:
            self.enc_p = TextEncoder_VITS2(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                text_enc_hidden_dim,
                f0=use_f0,
                gin_channels=gin_channels,
            )
        else:
            self.enc_p = TextEncoder_VITS1(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                text_enc_hidden_dim,
                f0=use_f0,
            )
        if use_f0:
            if vocoder == "RefineGAN":
                from rvc.lib.algorithm.generators import RefineGANGenerator
                self.dec = RefineGANGenerator(
                    sample_rate=sr,
                    downsample_rates=upsample_rates[::-1],
                    upsample_rates=upsample_rates,
                    start_channels=16,
                    num_mels=inter_channels,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: RefineGAN")
            elif vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                from rvc.lib.algorithm.generators import RingFormerGeneratorPrior
                self.dec = RingFormerGeneratorPrior(
                    initial_channel=inter_channels,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilation_sizes=resblock_dilation_sizes,
                    upsample_rates=upsample_rates,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    gen_istft_n_fft=gen_istft_n_fft,
                    gen_istft_hop_size=gen_istft_hop_size,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                print(f"    ██████  Vocoder: {vocoder}")
            elif vocoder == "APEX-GAN":
                from rvc.lib.algorithm.generators import APEX_GAN_Generator
                self.dec = APEX_GAN_Generator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                )
                print("    ██████  Vocoder: APEX-GAN")
            elif vocoder == "ChouwaGAN":
                from rvc.lib.algorithm.generators import ChouwaGANGenerator
                self.dec = ChouwaGANGenerator(
                    initial_channel=inter_channels,
                    upsample_rates=upsample_rates,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilation_sizes=resblock_dilation_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: ChouwaGAN")
            else:  # vocoder == "HiFi-GAN"
                from rvc.lib.algorithm.generators import HiFiGANNSFGenerator
                self.dec = HiFiGANNSFGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
                print("    ██████  Vocoder: NSF-HiFi-GAN")
        else:
            if vocoder in ["RefineGAN", "RingFormer_v1", "RingFormer_v2", "APEX-GAN", "ChouwaGAN"]:
                print(f"{vocoder} does not support training without pitch guidance.")
                self.dec = None
            else: # vocoder == "HiFi-GAN"
                from rvc.lib.algorithm.generators import HiFiGANGenerator
                self.dec = HiFiGANGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    checkpointing=checkpointing,
                )
        # Scale-VAE: learnable per-dimension scaling for posterior latents.
        # Scales z only for the decoder path, keeping KL on unscaled z.
        # Prevents posterior collapse with strong decoders like ChouwaGAN.
        if vocoder == "ChouwaGAN":
            self.posterior_scale = torch.nn.Parameter(torch.ones(inter_channels))

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        if vits2_mode:
            self.flow = ResidualCouplingTransformersBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                3,
                gin_channels=gin_channels,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                3,
                gin_channels=gin_channels,
            )

        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

        # Period VITS: Frame Pitch Predictor — forces the prior encoder to
        # carry continuous pitch information, preventing KL collapse.
        if use_period_vits:
            self.pitch_predictor = FramePitchPredictor(
                in_channels=inter_channels,
                hidden_channels=inter_channels,
                n_layers=4,
                kernel_size=5,
                p_dropout=p_dropout,
            )
            print("    ██████  Period VITS: Frame Pitch Predictor enabled")

    def _remove_weight_norm_from(self, module):
        """Utility to remove weight normalization from a module."""
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        """Removes weight normalization from the model."""
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        spec: Optional[torch.Tensor] = None, # y
        spec_lengths: Optional[torch.Tensor] = None, # y_lengths
        ds: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            spek (torch.Tensor, optional): Target spectrogram.  - y
            spek_lengths (torch.Tensor, optional): Lengths of the target spectrograms. - y_lengths
            ds (torch.Tensor, optional): Speaker embedding.
        """
        g = self.emb_g(ds).unsqueeze(-1)

        if self.vits2_mode:
            m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths, g=g)
        else:
            m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        # Period VITS: predict pitch from prior mean (training only)
        pitch_pred = None
        if self.use_period_vits:
            log_f0_pred, vuv_pred = self.pitch_predictor(m_p, x_mask)
            pitch_pred = (log_f0_pred, vuv_pred)

        if spec is not None:
            z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
            z_p = self.flow(z, spec_mask, g=g)

            if self.vocoder in ["RingFormer_v1", "RingFormer_v2"]:
                z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
                pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                o, spec, phase = self.dec(z_slice, pitchf, g=g)

                return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (spec, phase), pitch_pred

            elif self.vocoder == "APEX-GAN":
                z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
                pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                o = self.dec(z_slice, pitchf, g=g, return_intermediates=True)

                return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pitch_pred

            elif self.vocoder == "RefineGAN":
                z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
                pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                o = self.dec(z_slice, pitchf, g=g)

                return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pitch_pred

            elif self.vocoder == "ChouwaGAN":
                # Scale-VAE: scale z for decoder, keep original z for KL
                z_scaled = self.posterior_scale.view(1, -1, 1) * z
                z_slice, ids_slice = rand_slice_segments(z_scaled, spec_lengths, self.segment_size)
                pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                o = self.dec(z_slice, pitchf, g=g)

                return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pitch_pred

            else: # For HiFi-Gan training
                z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)

                if self.use_f0:
                    pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                    o = self.dec(z_slice, pitchf, g=g)
                else:
                    o = self.dec(z_slice, g=g)

                return o, ids_slice, x_mask, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pitch_pred
        else:
            print(" NONE SPEC ")
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None), pitch_pred

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        seed: int = 0,
        rate: Optional[torch.Tensor] = None,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            rate (torch.Tensor, optional): Rate for time-stretching.
            seed (int, optional): Seed for randomization of noise.
        """
        g = self.emb_g(sid).unsqueeze(-1)

        if self.vits2_mode:
            m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths, g=g)
        else:
            m_p, logs_p, x_mask = self.enc_p(phone=phone, pitch=pitch, lengths=phone_lengths)

        # Seed handler - receiver
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]

            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)

        if self.vocoder in ["RingFormer_v1", "RingFormer_v2"]:
            o, _, _ = self.dec(z * x_mask, nsff0, g=g)
        elif self.vocoder == "APEX-GAN":
            o = (self.dec(z * x_mask, nsff0, g=g, return_intermediates=False) if self.use_f0 else self.dec(z * x_mask, g=g, return_intermediates=False))
        elif self.vocoder == "RefineGAN":
            o = (self.dec(z * x_mask, nsff0, g=g) if self.use_f0 else self.dec(z * x_mask, g=g))
        elif self.vocoder == "ChouwaGAN":
            z_scaled = self.posterior_scale.view(1, -1, 1) * z
            o = self.dec(z_scaled * x_mask, nsff0, g=g)
        else: # HiFi-GAN
            o = (self.dec(z * x_mask, nsff0, g=g) if self.use_f0 else self.dec(z * x_mask, g=g))

        return o, x_mask, (z, z_p, m_p, logs_p)
