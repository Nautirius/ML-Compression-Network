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


class Conv1d_Strided_Autoencoder(nn.Module, Compressor):
    """Minimal **Conv1d** auto‑encoder z jednym, wspólnym `stride`.

    * Brak BN / Dropout.
    * Ten sam ``stride`` (np. 2) w każdej warstwie enkodera ⇒ regularne
      down‑sampling.
    * Dekoder symetryczny z `ConvTranspose1d` o tym samym stride.
    * Bottleneck to 1×1 conv na skróconym sygnale.
    """

    ACT_MAP = {
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

        # --------------------------- atrybuty ---------------------------
        self.input_length = input_length
        self.input_dim = input_length
        self.code_dim = code_dim
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_name = activation

        pad = kernel_size // 2
        act_factory = self.ACT_MAP.get(activation, nn.ReLU)

        # --------------------------- encoder ---------------------------
        enc_layers: List[nn.Module] = []
        in_ch = 1
        out_len = input_length
        for ch in self.conv_channels:
            enc_layers.extend([
                nn.Conv1d(in_ch, ch, kernel_size, stride=self.stride, padding=pad),
                act_factory(),
            ])
            out_len = math.ceil(out_len / self.stride)
            in_ch = ch
        self.encoder = nn.Sequential(*enc_layers)
        self.latent_len = out_len

        # bottleneck ----------------------------------------------------
        self.to_latent = nn.Conv1d(self.conv_channels[-1], code_dim, 1)
        self.from_latent = nn.Conv1d(code_dim, self.conv_channels[-1], 1)

        # --------------------------- decoder ---------------------------
        dec_layers: List[nn.Module] = []
        in_ch = self.conv_channels[-1]
        for out_ch in self.conv_channels[::-1][1:]:
            dec_layers.extend([
                nn.ConvTranspose1d(
                    in_ch, out_ch, kernel_size,
                    stride=self.stride, padding=pad, output_padding=self.stride - 1,
                ),
                act_factory(),
            ])
            in_ch = out_ch
        # końcowa warstwa na 1 kanał
        dec_layers.append(
            nn.ConvTranspose1d(
                in_ch, 1, kernel_size,
                stride=self.stride, padding=pad, output_padding=self.stride - 1,
            )
        )
        self.decoder = nn.Sequential(*dec_layers)

        # do kalkulacji CR w helperach
        self.layer_dims = [input_length] + self.conv_channels + [code_dim]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.to_latent(self.encoder(x))
        recon = self.decoder(self.from_latent(z))
        return recon[:, :, : self.input_length].squeeze(1)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.to_latent(self.encoder(x))

    def decompress(self, code: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(self.from_latent(code))
        return recon[:, :, : self.input_length].squeeze(1)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{[1,*self.conv_channels,1]}_s{self.stride}_code{self.code_dim}"