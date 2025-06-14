from typing import Dict, List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
from models.Compressor import Compressor


class Conv1d_Generic_Autoencoder(nn.Module, Compressor):
    """Lightweight **Conv1d** auto‑encoder (no BN, no Dropout).

    Treats each 187‑element vector as a length‑187 1‑D signal. Compression is
    achieved via a linear bottleneck on the flattened convolutional features.
    The model is intentionally minimal to encourage fast training and easy
    experimentation.
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
            activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.input_length = input_length  # alias for compression‑ratio calc
        self.input_dim = input_length  # ensure evaluate_autoencoder works
        self.code_dim = code_dim
        self.conv_channels = list(conv_channels)
        self.activation_name = activation
        self.kernel_size = kernel_size
        self.layer_dims = [input_length] + list(conv_channels) + [code_dim]

        if activation not in self.ACT_MAP:
            raise ValueError(f"Unsupported activation {activation}")
        act_factory = self.ACT_MAP[activation]
        pad = kernel_size // 2  # keep length

        # ------------------------- encoder conv ---------------------------
        enc_layers: List[nn.Module] = []
        in_ch = 1
        for ch in self.conv_channels:
            enc_layers.extend([
                nn.Conv1d(in_ch, ch, kernel_size, padding=pad),
                act_factory(),
            ])
            in_ch = ch
        self.encoder_conv = nn.Sequential(*enc_layers)

        # bottleneck -------------------------------------------------------
        self.flatten = nn.Flatten(start_dim=1)  # (B, ch*L)
        flat_dim = self.conv_channels[-1] * input_length
        self.to_latent = nn.Linear(flat_dim, code_dim)

        # ------------------------- decoder --------------------------------
        self.from_latent = nn.Linear(code_dim, flat_dim)
        dec_layers: List[nn.Module] = []
        rev_channels = list(self.conv_channels[::-1]) + [1]
        for in_ch, out_ch in zip(rev_channels[:-1], rev_channels[1:]):
            dec_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
                act_factory() if out_ch != 1 else nn.Identity(),
            ])
        self.decoder_conv = nn.Sequential(*dec_layers)

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        z = self.encoder_conv(x)
        z_latent = self.to_latent(self.flatten(z))
        recon_flat = self.from_latent(z_latent)
        recon = recon_flat.view(x.size(0), self.conv_channels[-1], self.input_length)
        out = self.decoder_conv(recon)
        return out.squeeze(1)  # (B, 187)

    # ----------------------------------------------------------------------
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.encoder_conv(x)
        return self.to_latent(self.flatten(z))

    def decompress(self, code: torch.Tensor) -> torch.Tensor:
        recon_flat = self.from_latent(code)
        recon = recon_flat.view(code.size(0), self.conv_channels[-1], self.input_length)
        out = self.decoder_conv(recon)
        return out.squeeze(1)

    def __str__(self) -> str:  # noqa: D401
        return f"{self.__class__.__name__}_{[1, *self.conv_channels, 1]}_code{self.code_dim}_{self.activation_name}"
