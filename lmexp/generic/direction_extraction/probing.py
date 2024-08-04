import random
from typing import Literal
import torch
from tqdm import tqdm
from lmexp.generic.get_locations import TokenLocationFn, last_token
from lmexp.generic.hooked_model import HookedModel
from lmexp.generic.tokenizer import Tokenizer


def train_probe(
    labeled_text: list[tuple[str, float]],
    model: HookedModel,
    tokenizer: Tokenizer,
    layer: int,
    n_epochs: int = 10,
    batch_size: int = 1,
    lr: float = 1e-4,
    token_location_fn: TokenLocationFn = last_token,
    search_tokens: torch.Tensor | None = None,
    save_to: str | None = "probe.pth",
    loss_type: Literal["mse", "bce"] = "mse",
) -> torch.nn.Linear:
    data = labeled_text.copy()
    probe = torch.nn.Linear(model.resid_dim, 1, bias=True).to(model.device)
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    model.clear_all()
    model.add_save_resid_activations_hook(layer)
    if search_tokens is not None:
        search_tokens = search_tokens.to(model.device)
    for epoch in range(n_epochs):
        random.shuffle(data)
        mean_loss = 0
        for i in tqdm(range(0, len(data) - batch_size, batch_size)):
            optimizer.zero_grad()
            model.clear_saved_activations()
            batch = data[i : i + batch_size]
            texts, labels = zip(*batch)
            token_batch, lens = tokenizer.batch_encode(
                texts, return_original_lengths=True
            )
            token_batch, lens = token_batch.to(model.device), torch.tensor(lens).to(model.device)
            label_batch = torch.tensor(labels, device=model.device)
            model.forward(token_batch)
            mask = token_location_fn(
                token_batch, lens, search_tokens, False
            )
            saved_acts = torch.cat(model.get_saved_activations(layer))
            means_over_tokens = (saved_acts * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                dim=-1, keepdim=True
            )  # batch_size, resid_dim
            preds = probe(means_over_tokens.to(model.device)).squeeze()
            if loss_type == "mse":
                loss = torch.nn.functional.mse_loss(preds, label_batch.to(model.device))
            elif loss_type == "bce":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    preds, label_batch.to(model.device)
                )
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        mean_loss /= len(data) / batch_size
        print(f"Epoch {epoch}, mean loss: {mean_loss}")

    if save_to is not None:
        torch.save(probe, save_to)
    return probe


def load_probe(
    path: str,
) -> torch.nn.Linear:
    return torch.load(path)
