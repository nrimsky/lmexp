import pytest
import torch
from lmexp.generic.get_locations import (
    last_token,
    all_tokens,
    at_search_tokens,
    before_search_tokens,
    from_search_tokens,
    after_search_tokens,
)


@pytest.fixture
def sample_data():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [2, 3, 4, 5, 6, 7, 8]])
    seq_lens = torch.tensor([5, 7])
    search_tokens = torch.tensor([3, 4])
    return input_ids, seq_lens, search_tokens


def test_last_token(sample_data):
    input_ids, seq_lens, _ = sample_data
    result = last_token(input_ids, seq_lens, None)
    expected = torch.tensor([[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1]])
    assert torch.all(result == expected)


def test_all_tokens(sample_data):
    input_ids, seq_lens, _ = sample_data
    result = all_tokens(input_ids, seq_lens, None)
    expected = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    assert torch.all(result == expected)


def test_at_search_tokens(sample_data):
    input_ids, _, search_tokens = sample_data
    result = at_search_tokens(input_ids, None, search_tokens)
    expected = torch.tensor([[0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0]])
    assert torch.all(result == expected)


def test_before_search_tokens(sample_data):
    input_ids, _, search_tokens = sample_data
    result = before_search_tokens(input_ids, None, search_tokens)
    expected = torch.tensor([[1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]])
    assert torch.all(result == expected)


def test_from_search_tokens(sample_data):
    input_ids, _, search_tokens = sample_data
    result = from_search_tokens(input_ids, None, search_tokens)
    expected = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]])
    assert torch.all(result == expected)


def test_after_search_tokens(sample_data):
    input_ids, _, search_tokens = sample_data
    result = after_search_tokens(input_ids, None, search_tokens)
    expected = torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
    assert torch.all(result == expected)


def test_search_tokens_not_found():
    input_ids = torch.tensor([[1, 2, 5, 6, 7]])
    search_tokens = torch.tensor([3, 4])
    with pytest.raises(ValueError):
        at_search_tokens(input_ids, None, search_tokens)
