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

class ResidualBlock(nn.Module):
    """1‑D residual block: Conv1d → activation with optional projection skip."""

    def __init__(self, in_ch: int, out_ch: int, *, kernel: int, stride: int, activation: nn.Module):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad)
        self.act = activation
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.act(self.conv(x) + self.skip(x))

###############################################################################
# Conv1d auto‑encoder families                                                #
###############################################################################

class Conv1d_Strided_Autoencoder(nn.Module, Compressor):
    """Strided Conv1d auto‑encoder with optional residual blocks and GAP."""

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
        activation: str = "gelu",
        stride: int = 1,
        use_residual: bool = True,
        use_adaptive_pool: bool = True,
    ) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError("stride musi być ≥ 1")

        # ---------------- attributes ----------------
        self.input_length = input_length
        self.input_dim = input_length  # for compression‑ratio utils
        self.code_dim = code_dim
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_residual = use_residual
        self.use_adaptive_pool = use_adaptive_pool

        act_factory = self.ACT_MAP[activation]
        pad = kernel_size // 2

        # ---------------- encoder -------------------
        enc_layers: List[nn.Module] = []
        in_ch = 1
        out_len = input_length
        for ch in self.conv_channels:
            if use_residual:
                enc_layers.append(
                    ResidualBlock(in_ch, ch, kernel=kernel_size, stride=stride, activation=act_factory())
                )
            else:
                enc_layers.extend([
                    nn.Conv1d(in_ch, ch, kernel_size, stride=stride, padding=pad),
                    act_factory(),
                ])
            out_len = math.ceil(out_len / stride)
            in_ch = ch
        if use_adaptive_pool:
            enc_layers.append(nn.AdaptiveAvgPool1d(1))
            out_len = 1
        self.encoder = nn.Sequential(*enc_layers)
        self.latent_len = out_len

        # ---------------- bottleneck --------------
        self.to_latent = nn.Conv1d(self.conv_channels[-1], code_dim, 1)
        self.from_latent = nn.Conv1d(code_dim, self.conv_channels[-1], 1)

        # ---------------- decoder -----------------
        dec_layers: List[nn.Module] = []
        in_ch = self.conv_channels[-1]
        for out_ch in self.conv_channels[::-1][1:]:
            dec_layers.extend([
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=stride,
                    padding=pad,
                    output_padding=stride - 1,
                ),
                act_factory(),
            ])
            in_ch = out_ch
        dec_layers.append(
            nn.ConvTranspose1d(
                in_ch,
                1,
                kernel_size,
                stride=stride,
                padding=pad,
                output_padding=stride - 1,
            )
        )
        self.decoder = nn.Sequential(*dec_layers)

        # for CR display
        self.layer_dims = [input_length] + self.conv_channels + [code_dim]

    # ---------------- interface ---------------
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

    def __str__(self) -> str:  # noqa: D401
        flags = []
        if self.use_residual:
            flags.append("res")
        if self.use_adaptive_pool:
            flags.append("gap")
        flag_part = ("_" + "-".join(flags)) if flags else ""
        return (
            f"{self.__class__.__name__}_{[1,*self.conv_channels,1]}_s" \
            f"{self.stride}_code{self.code_dim}{flag_part}"
        )