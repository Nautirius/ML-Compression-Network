import os
import time
import json
import math
import statistics as stats
from typing import Dict, List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.Compressor import Compressor


class ResBlock1d(nn.Module):
    """Prosty blok residual: Conv1d → Act → Conv1d + skip."""

    def __init__(self, channels: int, kernel_size: int = 3, activation: nn.Module = nn.ReLU):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + x)


class Conv1d_Strided_Autoencoder_V2(nn.Module, Compressor):
    """Minimal Conv1d auto‑encoder z jednolitym stride."""

    ACT = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
    }

    def __init__(
            self,
            *,
            input_length: int = 187,
            conv_channels: Sequence[int] = (32, 64, 128),
            code_dim: int = 16,
            kernel_size: int = 3,
            activation: str = "relu",
            stride: int = 2,
    ) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError("stride musi być ≥ 1")

        self.input_length = input_length
        self.input_dim = input_length
        self.conv_channels = list(conv_channels)
        self.code_dim = code_dim
        self.kernel_size = kernel_size
        self.stride = stride

        pad = kernel_size // 2
        act = self.ACT.get(activation, nn.ReLU)

        # Encoder -------------------------------------------------------
        enc: List[nn.Module] = []
        in_ch = 1
        length = input_length
        for ch in self.conv_channels:
            enc.extend([
                nn.Conv1d(in_ch, ch, kernel_size, stride=stride, padding=pad),
                act(),
            ])
            length = math.ceil(length / stride)
            in_ch = ch
        self.encoder_conv = nn.Sequential(*enc)
        self.latent_len = length

        # Bottleneck ----------------------------------------------------
        self.to_latent = nn.Conv1d(self.conv_channels[-1], code_dim, 1)
        self.from_latent = nn.Conv1d(code_dim, self.conv_channels[-1], 1)

        # Decoder -------------------------------------------------------
        dec: List[nn.Module] = []
        in_ch = self.conv_channels[-1]
        for out_ch in self.conv_channels[::-1][1:]:
            dec.extend([
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=stride,
                    padding=pad,
                    output_padding=stride - 1,
                ),
                act(),
            ])
            in_ch = out_ch
        dec.append(
            nn.ConvTranspose1d(
                in_ch,
                1,
                kernel_size,
                stride=stride,
                padding=pad,
                output_padding=stride - 1,
            )
        )
        self.decoder = nn.Sequential(*dec)

        self.layer_dims = [input_length] + self.conv_channels + [code_dim]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.to_latent(self.encoder_conv(x))
        recon = self.decoder(self.from_latent(z))
        return recon[:, :, :self.input_length].squeeze(1)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.to_latent(self.encoder_conv(x))
        z = z.view(z.size(0), -1)  # [B, code_dim * latent_len]
        return z.half()  # <= zmniejszenie precyzji do float16

    def decompress(self, code: torch.Tensor) -> torch.Tensor:
        code = code.float()  # <- konwersja z float16 do float32
        code = code.view(-1, self.code_dim, self.latent_len)
        recon = self.decoder(self.from_latent(code))
        return recon[:, :, :self.input_length].squeeze(1)

    def compressed_dim(self) -> int:
        return self.code_dim * self.latent_len

    def __str__(self) -> str:
        return f"{self.__class__.__name__}2_{[1, *self.conv_channels, 1]}_s{self.stride}_code{self.code_dim}"
