from lmexp.generic.tokenizer import Tokenizer
import torch
import pytest


class SimpleTokenizer(Tokenizer):

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([ord(c) for c in text], dtype=torch.int64)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join([chr(c.item()) for c in tokens])

    @property
    def pad_token(self) -> int:
        return 0


@pytest.fixture
def tokenizer():
    return SimpleTokenizer()


def test_encode(tokenizer):
    text = "hello"
    expected = torch.tensor([104, 101, 108, 108, 111], dtype=torch.int64)
    result = tokenizer.encode(text)
    assert torch.equal(result, expected)


def test_decode(tokenizer):
    tokens = torch.tensor([104, 101, 108, 108, 111], dtype=torch.int64)
    expected = "hello"
    result = tokenizer.decode(tokens)
    assert result == expected


def test_pad_token(tokenizer):
    expected = 0
    result = tokenizer.pad_token
    assert result == expected


def test_batch_encode(tokenizer):
    texts = ["hi", "hello"]
    expected = torch.tensor(
        [[104, 105, 0, 0, 0], [104, 101, 108, 108, 111]], dtype=torch.int64
    )
    result = tokenizer.batch_encode(texts)
    assert torch.equal(result, expected)
