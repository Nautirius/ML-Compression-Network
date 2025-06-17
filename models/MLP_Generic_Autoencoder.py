import torch
import torch.nn as nn
from models.Compressor import Compressor


class MLP_Generic_Autoencoder(nn.Module, Compressor):
    """
    Tworzy autoenkoder z zadanej listy wymiarów, np. [187, 128, 64, 16]
    """

    def __init__(self, layer_dims: list[int] | None = None):
        super().__init__()
        if layer_dims is None:  # np. [187, 64, 16]
            layer_dims = [187, 64, 16]

        assert len(layer_dims) >= 2, "Wymagane co najmniej dwa wymiary (wejście i wyjście)"

        self.layer_dims = layer_dims

        # ---------------- Encoder ---------------- #
        encoder_layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.GELU())
        encoder_layers.pop()
        self.encoder = nn.Sequential(*encoder_layers)

        # ---------------- Decoder ---------------- #
        decoder_layers = []
        for in_dim, out_dim in zip(layer_dims[::-1][:-1], layer_dims[::-1][1:]):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.GELU())
        decoder_layers.pop()
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> torch.Tensor:

        self.eval()
        z = self.encoder(x)
        return z.detach()

    @torch.no_grad()
    def decompress(self, code: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_hat = self.decoder(code)
        return x_hat.detach()

    def __str__(self):
        return f"MLP_Generic_Autoencoder(layer_dims={self.layer_dims})"
