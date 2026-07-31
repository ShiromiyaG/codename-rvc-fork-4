from typing import Optional

import torch
import numpy as np
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.attentions import MultiHeadAttention, FFN

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
        checkpointing: bool = False,
        use_sdpa: bool = True,
        **kwargs
    ):
        super(TransformerEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.checkpointing = checkpointing

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                    use_sdpa=use_sdpa,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        zippep = zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        )
        for attn_layers, norm_layers_1, ffn_layers, norm_layers_2 in zippep:
            def layer(
                value,
                value_mask,
                attention_mask,
                attention=attn_layers,
                norm_1=norm_layers_1,
                feed_forward=ffn_layers,
                norm_2=norm_layers_2,
            ):
                residual = attention(value, value, attention_mask)
                residual = self.drop(residual)
                value = norm_1(value + residual)
                residual = feed_forward(value, value_mask)
                residual = self.drop(residual)
                return norm_2(value + residual)

            if self.checkpointing and self.training:
                x = checkpoint(
                    layer, x, x_mask, attn_mask, use_reentrant=False
                )
            else:
                x = layer(x, x_mask, attn_mask)
        x = x * x_mask
        return x
