import torch.nn as nn

from models.Compressor import Compressor


# class Conv1D_Autoencoder(nn.Module, Compressor):
#     """Conv1D Autoencoder działające na sygnałach o dowolnej długości."""
#
#     def __init__(self, compressed_channels=16, bottleneck_length=24):
#         super().__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(64, compressed_channels, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(bottleneck_length)  # wymusi długość 24
#         )
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(compressed_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
#         )
#
#         self.bottleneck_length = bottleneck_length
#
#     def forward(self, x):
#         """
#         x: (batch, length)
#         """
#         x = x.unsqueeze(1)  # (batch, 1, length)
#         z = self.encoder(x)  # (batch, compressed_channels, bottleneck_length)
#
#         recon = self.decoder(z)  # (batch, 1, ?)
#
#         # Dopasuj wyjście do oryginalnej długości
#         if recon.shape[2] != x.shape[2]:
#             recon = nn.functional.interpolate(recon, size=x.shape[2], mode='linear', align_corners=False)
#
#         return recon.squeeze(1)  # (batch, length)
#
#
#     def compress(self, x):
#         return self.encoder(x)
#
#     def decompress(self, code):
#         return self.decoder(code)
#

class Conv1D_Autoencoder(nn.Module):
    def __init__(self, compressed_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),  # 187 -> floor((187+2*1-2)/2)+1 = 94
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 94 -> floor((94+2*1-2)/2)+1 = 47
            nn.ReLU(),
            nn.Conv1d(64, compressed_channels, kernel_size=3, stride=2, padding=1),  # 47 -> floor((47+2*1-2)/2)+1=24
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(compressed_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            # 24 -> 47
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 47 -> 94
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=0),  # 94 -> 187 (poprawka)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 187)
        z = self.encoder(x)  # (batch, compressed_channels, 24)
        recon = self.decoder(z)  # (batch, 1, 187)
        return recon.squeeze(1)  # (batch, 187)

    def compress(self, x):
        return self.encoder(x)

    def decompress(self, code):
        return self.decoder(code)
