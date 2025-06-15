import os
import time
import json
import math
import statistics as stats
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.Compressor import Compressor


class Conv1d_Strided_Autoencoder(nn.Module, Compressor):
    """Konfigurowalny autoenkoder 1-D – compress/decompress przyjmują tylko NumPy."""

    ACTS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu":  nn.ELU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
    }

    def __init__(
        self,
        *,
        input_length: int = 187,
        conv_channels: Sequence[int] = (32, 64, 128),
        code_dim: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.code_dim     = code_dim

        pad = kernel_size // 2
        act = self.ACTS.get(activation, nn.ReLU)

        # -------- encoder --------
        enc = []
        in_ch = 1
        for out_ch in conv_channels:
            enc += [
                nn.Conv1d(in_ch, out_ch, kernel_size,
                          stride=stride, padding=pad),
                act(),
            ]
            in_ch = out_ch
        enc.append(nn.AdaptiveAvgPool1d(1))      # latent_len == 1
        self.encoder = nn.Sequential(*enc)

        # bottleneck 1×1
        self.to_latent   = nn.Conv1d(in_ch, code_dim, 1)
        self.from_latent = nn.Conv1d(code_dim, in_ch, 1)

        # -------- decoder --------
        dec = []
        for out_ch in reversed(conv_channels[:-1]):
            dec += [
                nn.ConvTranspose1d(
                    in_ch, out_ch, kernel_size,
                    stride=stride, padding=pad, output_padding=stride - 1),
                act(),
            ]
            in_ch = out_ch
        dec.append(
            nn.ConvTranspose1d(
                in_ch, 1, kernel_size,
                stride=stride, padding=pad, output_padding=stride - 1)
        )
        self.decoder = nn.Sequential(*dec)

    # ---------------- API ----------------
    @torch.no_grad()
    def compress(self, x: np.ndarray) -> np.ndarray:
        """Zwraca tablicę NumPy (code_dim,) – dokładnie kod."""
        if not isinstance(x, np.ndarray):
            raise TypeError("compress() oczekuje jednowymiarowej tablicy NumPy")
        if x.ndim != 1:
            raise ValueError("compress() wymaga 1-D tablicy NumPy")

        tensor = torch.from_numpy(x.astype(np.float32))  # [L]
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,L]

        z = self.to_latent(self.encoder(tensor))          # [1, code_dim, 1]
        return z.squeeze().cpu().numpy()                  # (code_dim,)

    @torch.no_grad()
    def decompress(self, code: np.ndarray) -> np.ndarray:
        """Odtwarza sekwencję – również zwraca NumPy 1-D (input_length,)."""
        if not isinstance(code, np.ndarray):
            raise TypeError("decompress() oczekuje tablicy NumPy")
        if code.ndim != 1 or code.size != self.code_dim:
            raise ValueError(
                f"decompress() wymaga wektora 1-D długości {self.code_dim}"
            )

        tensor = torch.from_numpy(code.astype(np.float32))       # [code_dim]
        tensor = tensor.unsqueeze(0).unsqueeze(-1).to(self.device)  # [1,code_dim,1]

        recon = self.decoder(self.from_latent(tensor))           # [1,1,L+pad]
        return recon.squeeze().cpu().numpy()[: self.input_length]

    # „zwykły” forward do trenowania
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:                      # [B, L]
            x = x.unsqueeze(1)               # [B, 1, L]
        z = self.to_latent(self.encoder(x))
        recon = self.decoder(self.from_latent(z))
        return recon[:, :, : self.input_length].squeeze(1)

    def compressed_dim(self) -> int:
        return self.code_dim * self.latent_len

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{[1,*self.conv_channels,1]}_stride{self.stride}_codeDim{self.code_dim}_activation_{self.activation_name}_kernel{self.kernel_size}_stride{self.kernel_size}"