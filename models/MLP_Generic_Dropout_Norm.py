import torch
import torch.nn as nn
from models.Compressor import Compressor


class MLP_Generic_Dropout_Norm(nn.Module, Compressor):
    def __init__(
            self,
            layer_dims: list[int],
            dropout: float | None = None,
            batchnorm: bool = False
    ):
        super().__init__()

        assert len(layer_dims) >= 2, "Potrzebne minimum 2 wymiary"

        self.layer_dims = layer_dims
        self.dropout = dropout
        self.batchnorm = batchnorm

        self.encoder = self._build_network(layer_dims)
        self.decoder = self._build_network(layer_dims[::-1])

    def _build_network(self, dimensions):
        """Buduje sekwencyjną sieć w oparciu o wymiary."""
        layers = []
        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if self.batchnorm and out_dim != dimensions[-1]:
                layers.append(nn.BatchNorm1d(out_dim))
            if self.dropout is not None and out_dim != dimensions[-1]:
                layers.append(nn.Dropout(self.dropout))
            if out_dim != dimensions[-1]:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def compress(self, x):
        return self.encoder(x)

    def decompress(self, code):
        return self.decoder(code)

    def __str__(self):
        return f"MLP_Generic_Dropout_Norm {self.layer_dims} dropout={self.dropout} BatchNorm={self.batchnorm}"