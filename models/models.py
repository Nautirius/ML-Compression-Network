from .Conv1D_Autoencoder import Conv1D_Autoencoder
from .Conv1d_Generic_Autoencoder import Conv1d_Generic_Autoencoder
from .MLP_Expanded_Autoencoder_V2 import MLP_Expanded_Autoencoder_V2
from .MLP_Expanded_Autoencoder import MLP_Expanded_Autoencoder
from .MLP_Simple_Autoencoder import MLP_Simple_Autoencoder
from .MLP_Generic_Autoencoder import MLP_Generic_Autoencoder
from .MLP_Generic_Dropout_Norm import MLP_Generic_Dropout_Norm
from .Conv1d_Strided_Autoencoder import Conv1d_Strided_Autoencoder
from .Conv1d_Strided_Autoencoder_V2 import Conv1d_Strided_Autoencoder_V2


MODELS = {
    # 'conv1d_autoencoder': Conv1D_Autoencoder,
    'conv1d_generic_autoencoder': Conv1d_Generic_Autoencoder,
    # 'expanded_autoencoder': MLP_Expanded_Autoencoder,
    # 'expanded_autoencoder_2': MLP_Expanded_Autoencoder_V2,
    # 'simple_autoencoder': MLP_Simple_Autoencoder,
    'generic_autoencoder': MLP_Generic_Autoencoder,
    # 'generic_dropout_norm': MLP_Generic_Dropout_Norm,
    # 'conv1d_strided_autoencoder': Conv1d_Strided_Autoencoder,
    # 'conv1d_strided_autoencoder_v2': Conv1d_Strided_Autoencoder_V2
}
