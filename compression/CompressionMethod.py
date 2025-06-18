from enum import Enum


class CompressionMethod(str, Enum):
    """Klasa enum definiująca listę modeli Autoencoder i obsługująca odpowiadające im rozszerzenia."""
    MLP_8 = 'mlp_8'
    MLP_16 = 'mlp_16'
    MLP_32 = 'mlp_32'
    MLP_64 = 'mlp_64'
    CONV1D_8 = 'conv1d_8'
    CONV1D_16 = 'conv1d_16'
    CONV1D_32 = 'conv1d_32'
    CONV1D_64 = 'conv1d_64'

    @property
    def extension(self) -> str:
        return f".{str(self.value)}"

    @staticmethod
    def from_extension(filename: str):
        for method in CompressionMethod:
            if filename.endswith(f"{method.extension}.csv"):
                return method
        raise ValueError(f"Nieznane rozszerzenie pliku: {filename}")

    def __str__(self):
        return self.value
