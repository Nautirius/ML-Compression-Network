from .Conv1d_Generic_Autoencoder import Conv1d_Generic_Autoencoder
from .MLP_Generic_Autoencoder import MLP_Generic_Autoencoder

MODELS: dict[str, type[Conv1d_Generic_Autoencoder | MLP_Generic_Autoencoder]] = {
    'conv1d_generic_autoencoder': Conv1d_Generic_Autoencoder,
    'generic_autoencoder': MLP_Generic_Autoencoder,
}
