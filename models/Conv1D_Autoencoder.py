import numpy as np
import torch
import torch.nn as nn

from models.Compressor import Compressor



class Conv1D_Autoencoder(nn.Module):
    """
    Autoenkoder 1-D, który kompresuje dowolnie długie sygnały
    do wektora o 16 liczbach i potrafi je zrekonstruować.
    """

    CODE_SIZE = 16  # liczba liczb w wektorze kodu

    def __init__(self):
        super().__init__()

        # ---------- ENCODER ----------
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        # zmniejszamy wymiar długości do 1
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)              # (B,128,1)
        # kanałowy “squeeze” do 16 wartości
        self.encoder_to_code = nn.Conv1d(128, self.CODE_SIZE, kernel_size=1)  # (B,16,1)

        # ---------- DECODER ----------
        # najpierw wracamy do (B,128,1), potem 3× ConvTranspose
        self.code_to_decoder = nn.ConvTranspose1d(self.CODE_SIZE, 128, kernel_size=1)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

        # przy dekompresji trzeba wiedzieć, do jakiej długości skalować
        self._last_len: int | None = None

    # ====== API ======
    def compress(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        x : tensor (batch, length) – surowy sygnał
        Zwraca  : numpy.ndarray (batch, 16)
        """
        x = self._to_tensor(x)                # (B,L)
        self._last_len = x.shape[-1]          # zapamiętujemy długość
        code = self._encode(x)                # (B,16)
        return code.detach().cpu().numpy()

    def decompress(self, code: np.ndarray) -> np.ndarray:
        """
        code : numpy.ndarray (batch, 16) – zakodowane dane
        Zwraca: numpy.ndarray (batch, original_length)
        """
        if self._last_len is None:
            raise RuntimeError(
                "Nie znam docelowej długości rekonstrukcji – "
                "najpierw wywołaj compress() lub przekaż długość w inny sposób."
            )

        z = torch.as_tensor(code, dtype=torch.float32, device=self._device()).unsqueeze(-1)  # (B,16,1)
        recon = self._decode(z, self._last_len)  # (B,L)
        return recon.detach().cpu().numpy()

    # ====== forward ======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardowy forward: zwraca rekonstrukcję."""
        self._last_len = x.shape[-1]
        code = self._encode(x)                # (B,16)
        recon = self._decode(code.unsqueeze(-1), x.shape[-1])  # (B,L)
        return recon

    # ====== helpers ======
    def _encode(self, x: torch.Tensor) -> torch.Tensor:        # (B,L) -> (B,16)
        x = x.unsqueeze(1)                                     # (B,1,L)
        h = self.encoder_conv(x)                               # (B,128,L')
        h = self.encoder_pool(h)                               # (B,128,1)
        code = self.encoder_to_code(h).squeeze(-1)             # (B,16)
        return code

    def _decode(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        z : (batch, 16, 1) lub (batch, 16) z .unsqueeze(-1)
        """
        if z.ndim == 2:
            z = z.unsqueeze(-1)                                # (B,16,1)
        h = self.code_to_decoder(z)                            # (B,128,1)
        recon = self.decoder_conv(h)                           # (B,1,~L)
        if recon.shape[-1] != target_len:
            recon = nn.functional.interpolate(
                recon, size=target_len, mode="linear", align_corners=False
            )
        return recon.squeeze(1)                                # (B,L)

    # ====== utils ======
    def _device(self):
        return next(self.parameters()).device

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(self._device(), dtype=torch.float32)


# class Conv1D_Autoencoder(nn.Module):
#     def __init__(self, compressed_channels=16):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),  # 187 -> floor((187+2*1-2)/2)+1 = 94
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 94 -> floor((94+2*1-2)/2)+1 = 47
#             nn.ReLU(),
#             nn.Conv1d(64, compressed_channels, kernel_size=3, stride=2, padding=1),  # 47 -> floor((47+2*1-2)/2)+1=24
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(compressed_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
#             # 24 -> 47
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 47 -> 94
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=0),  # 94 -> 187 (poprawka)
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # (batch, 1, 187)
#         z = self.encoder(x)  # (batch, compressed_channels, 24)
#         recon = self.decoder(z)  # (batch, 1, 187)
#         return recon.squeeze(1)  # (batch, 187)
#
#     def compress(self, x):
#         return self.encoder(x)
#
#     def decompress(self, code):
#         return self.decoder(code)
