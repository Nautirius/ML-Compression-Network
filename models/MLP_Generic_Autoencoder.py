import torch
import torch.nn as nn
import numpy as np
from models.Compressor import Compressor


class MLP_Generic_Autoencoder(nn.Module, Compressor):
    def __init__(self, layer_dims: list[int] = [187, 64, 16]
                 ):
        """
        Tworzy autoenkoder z zadanej listy wymiarów, np. [187, 128, 64, 16]
        """
        super().__init__()
        assert len(layer_dims) >= 2, "Wymagane co najmniej dwa wymiary (wejście i wyjście)"
        self.layer_dims = layer_dims

        encoder_layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.GELU())
        encoder_layers.pop()
        self.encoder = nn.Sequential(*encoder_layers)

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
    def compress(self, x: torch.Tensor) -> np.ndarray:
        """
        Zwraca zakodowaną reprezentację jako numpy array
        o długości self.layer_dims[-1] (domyślnie 16).
        """
        self.eval()                            # wyłącz dropout/batchnorm itp.
        z = self.encoder(x)
        return z.detach().cpu().numpy()        # (N, latent_dim) lub (latent_dim,)

    @torch.no_grad()
    def decompress(self, code: np.ndarray) -> np.ndarray:
        """
        Przyjmuje wyłącznie NumPy array i zwraca zrekonstruowany
        NumPy array o rozmiarze `self.layer_dims[0]`.

        Parameters
        ----------
        code : np.ndarray
            Koder latentny o kształcie (..., latent_dim).

        Returns
        -------
        np.ndarray
            Zrekonstruowane dane w formacie NumPy.
        """
        if not isinstance(code, np.ndarray):
            raise TypeError(
                "decompress przyjmuje tylko np.ndarray – podaj array, a nie tensor."
            )

        self.eval()                                           # tryb ewaluacji
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        code_t = torch.from_numpy(code).to(device=device, dtype=dtype)
        x_hat  = self.decoder(code_t)                         # (N, input_dim)
        return x_hat.cpu().numpy()

    def __str__(self):
        return f"MLP_Generic_Autoencoder {self.layer_dims}"
