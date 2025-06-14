import torch.nn as nn

from models.Compressor import Compressor


class MLP_Expanded_Autoencoder_V2(nn.Module, Compressor):
    def __init__(self, input_dim: int=187, compressed_dim: int=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, compressed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def compress(self, x):
        return self.encoder(x)

    def decompress(self, code):
        return self.decoder(code)