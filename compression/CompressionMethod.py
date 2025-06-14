from enum import Enum


class CompressionMethod(str, Enum):
    CONV1D = 'conv1d_autoencoder'
    MLP_EA = 'expanded_autoencoder'
    MLP_EA_2 = 'expanded_autoencoder_2'
    MLP_SA = 'simple_autoencoder'

    @property
    def extension(self) -> str:
        return {
            CompressionMethod.CONV1D: '.conv1d_autoencoder',
            CompressionMethod.MLP_EA: '.expanded_autoencoder',
            CompressionMethod.MLP_EA_2: '.expanded_autoencoder_2',
            CompressionMethod.MLP_SA: '.simple_autoencoder'
        }[self]

    @staticmethod
    def from_extension(filename: str):
        for method in CompressionMethod:
            if filename.endswith(method.extension):
                return method
        raise ValueError(f"Nieznane rozszerzenie pliku: {filename}")

    def __str__(self):
        return self.value
