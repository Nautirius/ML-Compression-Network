from typing import Union

from .Conv1d_Generic_Autoencoder import Conv1d_Generic_Autoencoder
from .MLP_Generic_Autoencoder import MLP_Generic_Autoencoder
from collections.abc import Callable

MODELS: dict[str, Callable[[], Union[Conv1d_Generic_Autoencoder, MLP_Generic_Autoencoder]]] = {
    'mlp_8': lambda: MLP_Generic_Autoencoder(layer_dims=[187, 80, 32, 8]),
    'mlp_16': lambda: MLP_Generic_Autoencoder(layer_dims=[187, 90, 40, 16]),
    'mlp_32': lambda: MLP_Generic_Autoencoder(layer_dims=[187, 100, 32]),
    'mlp_64': lambda: MLP_Generic_Autoencoder(layer_dims=[187, 64]),
    'conv1d_8': lambda: Conv1d_Generic_Autoencoder(latent_dim=8, conv_channels=[32, 64, 128, 256], kernel_size=5),
    'conv1d_16': lambda: Conv1d_Generic_Autoencoder(latent_dim=16, conv_channels=[32, 64, 128, 256], kernel_size=5),
    'conv1d_32': lambda: Conv1d_Generic_Autoencoder(latent_dim=32, conv_channels=[32, 64, 128, 256], kernel_size=5),
    'conv1d_64': lambda: Conv1d_Generic_Autoencoder(latent_dim=64, conv_channels=[32, 64, 128, 256], kernel_size=5),
}
