import torch
from abc import ABC, abstractmethod


class Tokenizer(ABC):

    @abstractmethod
    def encode(text: str) -> torch.tensor:
        raise NotImplementedError()

    @abstractmethod
    def decode(tokens: torch.tensor) -> str:
        raise NotImplementedError()
