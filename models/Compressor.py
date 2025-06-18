from abc import ABC, abstractmethod
import torch


class Compressor(ABC):
    """Klasa abstrakcyjna Autoencodera."""

    @abstractmethod
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decompress(self, code: torch.Tensor) -> torch.Tensor:
        pass
