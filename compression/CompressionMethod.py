from enum import Enum


class CompressionMethod(str, Enum):
    """Klasa enum definiująca listę modeli Autoencoder i obsługująca odpowiadające im rozszerzenia."""
    MLP_GENERIC = 'generic_autoencoder'
    CONV1D_GENERIC = 'conv1d_generic_autoencoder'

    @property
    def extension(self) -> str:
        return {
            CompressionMethod.MLP_GENERIC: '.generic_autoencoder',
            CompressionMethod.CONV1D_GENERIC: '.conv1d_generic_autoencoder'
        }[self]

    @staticmethod
    def from_extension(filename: str):
        for method in CompressionMethod:
            if filename.endswith(f"{method.extension}.csv"):
                return method
        raise ValueError(f"Nieznane rozszerzenie pliku: {filename}")

    def __str__(self):
        return self.value
