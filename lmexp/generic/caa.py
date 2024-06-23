import torch
from tqdm import tqdm
from lmexp.generic.hooked_model import HookedModel
from collections import defaultdict
import pickle
from lmexp.generic.tokenizer import Tokenizer


def get_caa_vecs(
    labeled_text: list[tuple[str, bool]],
    model: HookedModel,
    tokenizer: Tokenizer,
    layers: list[int],
    save_to: str | None = "vectors.pkl",
) -> dict:
    model.clear_all()
    for layer in layers:
        model.add_save_hook(layer)

    pos_acts = defaultdict(list)
    neg_acts = defaultdict(list)

    for text, is_pos in tqdm(labeled_text):
        model.clear_saved_activations()
        tokens = tokenizer.encode(text)
        with torch.no_grad():
            model.forward(tokens.to(model.device))
        for layer in layers:
            saved = model.get_saved_activations(layer)
            assert (
                len(saved) == 1
            ), "expected only one saved activation per dataset item, something has gone wrong in saving"
            acts = saved[0]
            if len(acts.shape) == 3:
                assert acts.shape[0] == 1, "expected batch size of 1"
                acts = acts[0]
            # acts has shape (seq_len, resid_dim) (for some seq_len, could be truncated by implementing output_to_acts differently on model or modifying this function to take take acts at certain positions)
            if is_pos:
                pos_acts[layer].append(acts)
            else:
                neg_acts[layer].append(acts)

    mean_diff_vecs = {}
    for layer in layers:
        pos = torch.cat(pos_acts[layer]).mean(dim=0)
        neg = torch.cat(neg_acts[layer]).mean(dim=0)
        mean_diff_vecs[layer] = pos - neg

    if save_to is not None:
        with open(save_to, "wb") as f:
            pickle.dump(mean_diff_vecs, f)

    return mean_diff_vecs


def load_caa_vecs(path: str) -> dict[int, torch.tensor]:
    with open(path, "rb") as f:
        return pickle.load(f)
