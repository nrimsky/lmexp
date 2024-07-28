import torch
from abc import ABC, abstractmethod


class Tokenizer(ABC):

    @abstractmethod
    def encode(text: str) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def decode(tokens: torch.Tensor) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> int:
        raise NotImplementedError()

    def batch_encode(
        self, texts: list[str], return_original_lengths=False
    ) -> torch.Tensor | tuple[torch.Tensor, list[int]]:
        token_list = [self.encode(t).flatten() for t in texts]
        lengths = [len(t) for t in token_list]
        max_len = max(lengths)
        padded = [
            torch.cat([t, torch.tensor([self.pad_token] * (max_len - len(t)))])
            for t in token_list
        ]
        res = torch.stack(padded).long()
        if return_original_lengths:
            return res, lengths
        return res

    def batch_decode(self, tokens: torch.Tensor) -> list[str]:
        assert len(tokens.shape) == 2, "expected 2D tensor for batch_decode"
        return [self.decode(t) for t in tokens]
