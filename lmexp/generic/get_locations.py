"""
methods to get token masks for activation steering or direction extraction

all of them take the arguments:
input_ids of shape batch x max_n_seq
seq_lens of shape batch - the actual sequence lengths of the input_ids (right-padding is assumed)
(optional) search_tokens of shape n_search_tokens - tokens to search for in the input_ids
in_sampling_mode - a boolean flag indicating whether we are in sampling mode or not
    - in sampling mode, we are sampling one extra token at a time
    - so we have different logic for the token masks in this case

and return a tensor of shape batch x n_seq with 1s corresponding to the token positions of interest and 0s elsewhere
"""

from typing import Callable
import torch


TokenLocationFn = Callable[
    [torch.Tensor, torch.Tensor | list | None, torch.Tensor | list | None, bool],
    torch.Tensor,
]


def validate_shapes(location_func: TokenLocationFn) -> TokenLocationFn:

    def wrapper(input_ids, seq_lens, search_tokens, in_sampling_mode=False):
        assert len(input_ids.shape) == 2
        if seq_lens is not None:
            assert len(seq_lens.shape) == 1
            assert input_ids.shape[0] == seq_lens.shape[0]
        if search_tokens is not None:
            assert len(search_tokens.shape) == 1
        return location_func(input_ids, seq_lens, search_tokens, in_sampling_mode)

    wrapper.__name__ = location_func.__name__

    return wrapper


def get_search_token_positions(input_ids, search_tokens) -> list[tuple[int, int]]:
    """
    returns [[start, end], ...] corresponding to the start and end positions of the (last occurence of) search_tokens in each input_ids sequence

    e.g. input_ids = [[1, 2, 3, 4, 5], [2, 3, 4, 0, 1]], search_tokens = [3, 4] -> [[2, 3], [1, 2]]
    """
    positions = []
    for seq in input_ids:
        found_pos = None
        for i in range(len(seq) - len(search_tokens) + 1):
            if torch.equal(seq[i : i + len(search_tokens)], search_tokens):
                found_pos = [i, i + len(search_tokens)]
        if found_pos is None:
            raise ValueError(
                f"search_tokens {search_tokens} not found in input_ids {seq}"
            )
        positions.append(found_pos)
    return positions


@validate_shapes
def last_token(input_ids, seq_lens, _, in_sampling_mode=False):
    if in_sampling_mode:
        return torch.zeros_like(input_ids)
    mask = torch.zeros_like(input_ids)
    for i, seq_len in enumerate(seq_lens):
        mask[i, seq_len - 1] = 1
    return mask


@validate_shapes
def all_tokens(input_ids, seq_lens, _, in_sampling_mode=False):
    if in_sampling_mode:
        return torch.ones_like(input_ids)
    mask = torch.zeros_like(input_ids)
    for i, seq_len in enumerate(seq_lens):
        mask[i, :seq_len] = 1
    return mask


@validate_shapes
def at_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    if in_sampling_mode:
        return torch.zeros_like(input_ids)
    search_positions = get_search_token_positions(input_ids, search_tokens)
    mask = torch.zeros_like(input_ids)
    for i, (start, end) in enumerate(search_positions):
        mask[i, start:end] = 1
    return mask


@validate_shapes
def before_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    if in_sampling_mode:
        return torch.zeros_like(input_ids)
    search_positions = get_search_token_positions(input_ids, search_tokens)
    mask = torch.zeros_like(input_ids)
    for i, (start, _) in enumerate(search_positions):
        mask[i, :start] = 1
    return mask


@validate_shapes
def from_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    if in_sampling_mode:
        return torch.ones_like(input_ids)
    search_positions = get_search_token_positions(input_ids, search_tokens)
    mask = torch.zeros_like(input_ids)
    for i, (start, _) in enumerate(search_positions):
        mask[i, start:] = 1
    return mask


@validate_shapes
def after_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    if in_sampling_mode:
        return torch.ones_like(input_ids)
    search_positions = get_search_token_positions(input_ids, search_tokens)
    mask = torch.zeros_like(input_ids)
    for i, (_, end) in enumerate(search_positions):
        mask[i, end:] = 1
    return mask
