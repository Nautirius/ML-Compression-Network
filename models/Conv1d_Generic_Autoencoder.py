from typing import List, Optional
import torch
import torch.nn as nn
from models.Compressor import Compressor


class Conv1d_Generic_Autoencoder(nn.Module, Compressor):
    """
    Auto-enkoder 1-D z dowolną liczbą warstw Conv1d.
    Gwarantuje identyczną długość sekwencji przy wyjściu (brak 191 → 187).
    API: compress / decompress pracują WYŁĄCZNIE na np.ndarray.
    """

    def __init__(
            self,
            input_length: int = 187,
            input_channels: int = 1,
            conv_channels: Optional[List[int]] = None,
            kernel_size: int = 3,
            stride: int = 1,
            latent_dim: int = 16,
            layer_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        # użyj layer_dims jeśli podane
        if layer_dims is not None:
            assert len(layer_dims) >= 3, "Wymagane co najmniej 3 wymiary (input_len, conv..., latent_dim)"
            input_length = layer_dims[0]
            latent_dim = layer_dims[-1]
            conv_channels = layer_dims[1:-1]

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.input_length = input_length
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim
        self.layer_dims = [input_length] + conv_channels + [latent_dim]

        pad = (kernel_size - stride) // 2  # "prawie-same” padding

        # ---------------- Encoder ---------------- #
        seq_lens = [input_length]
        enc_layers = []
        in_ch = input_channels
        L = input_length

        for out_ch in conv_channels:
            enc_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad))
            enc_layers.append(nn.GELU())
            L = ((L + 2 * pad - (kernel_size - 1) - 1) // stride) + 1
            seq_lens.append(L)
            in_ch = out_ch
        enc_layers.pop()  # ostatnia GELU niepotrzebna
        self.encoder_conv = nn.Sequential(*enc_layers)

        self.enc_flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(in_ch * L, latent_dim)

        # ---------------- Decoder ---------------- #
        self.decoder_fc = nn.Linear(latent_dim, in_ch * L)
        self.dec_unflatten = nn.Unflatten(1, (in_ch, L))

        dec_layers: list[nn.Module] = []
        ch_rev = conv_channels[::-1]  # kanały w odwrotnej kolejności
        # sekcje długości w odwrotnej kolejności: [..., L3, L2, L1, L0]
        for idx in range(len(conv_channels)):
            in_ch_cur = ch_rev[idx]
            out_ch_cur = ch_rev[idx + 1] if idx < len(conv_channels) - 1 else input_channels

            L_in = seq_lens[-1 - idx]  # długość przed warstwą deconv
            L_target = seq_lens[-2 - idx]  # długość, którą MUSIMY uzyskać

            no_pad_len = (L_in - 1) * stride - 2 * pad + kernel_size
            output_pad = int(L_target - no_pad_len)
            if not 0 <= output_pad < stride:
                raise ValueError(
                    f"Nie da się odwrócić długości {L_in} → {L_target} "
                    f"(kernel={kernel_size}, stride={stride}, pad={pad})."
                )

            dec_layers.append(
                nn.ConvTranspose1d(
                    in_ch_cur,
                    out_ch_cur,
                    kernel_size,
                    stride,
                    pad,
                    output_padding=output_pad,
                )
            )
            if out_ch_cur != input_channels:
                dec_layers.append(nn.GELU())

        self.decoder_conv = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        z_conv = self.encoder_conv(x)
        z_vec = self.enc_flatten(z_conv)
        z = self.encoder_fc(z_vec)

        y_vec = self.decoder_fc(z)
        y_conv = self.dec_unflatten(y_vec)
        out = self.decoder_conv(y_conv)

        return out.squeeze(1)

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z_conv = self.encoder_conv(x)
        z_vec = self.enc_flatten(z_conv)
        z = self.encoder_fc(z_vec)
        return z.detach()

    @torch.no_grad()
    def decompress(self, code: torch.Tensor) -> torch.Tensor:
        y_vec = self.decoder_fc(code)
        y_conv = self.dec_unflatten(y_vec)
        out = self.decoder_conv(y_conv)
        return out.squeeze(1).detach()

    def __str__(self):
        return (
            f"Conv1D_Generic_Autoencoder("
            f"in_len={self.input_length}, "
            f"channels={self.conv_channels}, "
            f"latent_dim={self.latent_dim}), "
            f"kernel={self.kernel_size}"
        )
