from abc import ABC, abstractmethod


class Compressor(ABC):
    @abstractmethod
    def compress(self, x):
        pass

    @abstractmethod
    def decompress(self, code):
        pass
