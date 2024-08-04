import torch
from tqdm import tqdm
from lmexp.generic.get_locations import TokenLocationFn, last_token
from lmexp.generic.hooked_model import HookedModel
from collections import defaultdict
import pickle
from lmexp.generic.tokenizer import Tokenizer


def get_caa_vecs(
    labeled_text: list[tuple[str, bool]],
    model: HookedModel,
    tokenizer: Tokenizer,
    layers: list[int],
    token_location_fn: TokenLocationFn = last_token,
    search_tokens: torch.Tensor | None = None,
    save_to: str | None = "vectors.pkl",
    batch_size: int = 1,
) -> dict:
    """
    labeled text: list of (text, positive or negative label)
    model: HookedModel
    tokenizer: Tokenizer
    layers: resid layers to extract activations from
    token_location_fn: function to get token locations at which to extract activations
    search_tokens: search tokens to use with token_location_fn (if applicable to the function)
    save_to: path to save the extracted vectors
    batch_size: batch size for forward passes
    """
    model.clear_all()
    for layer in layers:
        model.add_save_resid_activations_hook(layer)

    pos_acts = defaultdict(int)
    neg_acts = defaultdict(int)

    pos_n = defaultdict(int)
    neg_n = defaultdict(int)

    if search_tokens is not None:
        search_tokens = search_tokens.to(model.device)

    batches = [
        labeled_text[i : i + batch_size]
        for i in range(0, len(labeled_text), batch_size)
    ]
    if len(labeled_text) % batch_size != 0:
        batches.append(labeled_text[-(len(labeled_text) % batch_size) :])

    for labeled_text_batch in tqdm(batches):
        model.clear_saved_activations()
        tokens, lengths = tokenizer.batch_encode(
            [text for text, _ in labeled_text_batch], return_original_lengths=True
        )
        tokens, lengths = tokens.to(model.device), torch.tensor(lengths).to(model.device)
        labels = [label for _, label in labeled_text_batch]
        token_location_mask = token_location_fn(
            tokens, lengths, search_tokens, False
        )
        with torch.no_grad():
            model.forward(tokens)
        for layer in layers:
            saved = model.get_saved_activations(layer)
            assert len(saved) == 1 and saved[0].shape[0] == len(labeled_text_batch)
            saved = saved[0]
            token_location_mask = token_location_mask.to(saved.dtype)
            sums = (saved * token_location_mask.unsqueeze(-1)).sum(
                dim=1
            )  # batch_size, resid_dim
            means_over_tokens = sums / token_location_mask.sum(dim=-1, keepdim=True)
            for vec, is_pos in zip(means_over_tokens, labels, strict=True):
                if is_pos:
                    pos_acts[layer] += vec
                    pos_n[layer] += 1
                else:
                    neg_acts[layer] += vec
                    neg_n[layer] += 1

    mean_diff_vecs = {}
    for layer in layers:
        pos = pos_acts[layer] / pos_n[layer]
        neg = neg_acts[layer] / neg_n[layer]
        mean_diff_vecs[layer] = pos - neg

    if save_to is not None:
        with open(save_to, "wb") as f:
            pickle.dump(mean_diff_vecs, f)

    return mean_diff_vecs


def load_caa_vecs(path: str) -> dict[int, torch.tensor]:
    with open(path, "rb") as f:
        return pickle.load(f)
