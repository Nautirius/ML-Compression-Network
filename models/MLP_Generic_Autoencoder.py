import torch.nn as nn
from models.Compressor import Compressor


class MLP_Generic_Autoencoder(nn.Module, Compressor):
    def __init__(self, layer_dims: list[int]):
        """
        Tworzy autoenkoder z zadanej listy wymiarów, np. [187, 128, 64, 16]
        """
        super().__init__()
        assert len(layer_dims) >= 2, "Wymagane co najmniej dwa wymiary (wejście i wyjście)"

        encoder_layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())
        encoder_layers.pop()
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for in_dim, out_dim in zip(layer_dims[::-1][:-1], layer_dims[::-1][1:]):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.ReLU())
        decoder_layers.pop()
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def compress(self, x):
        return self.encoder(x)

    def decompress(self, code):
        return self.decoder(code)
