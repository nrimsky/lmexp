import random
from typing import Literal
import torch
from tqdm import tqdm
from lmexp.generic.hooked_model import HookedModel
from lmexp.generic.tokenizer import Tokenizer

def train_probe(
    labeled_text: list[tuple[str, float]],
    model: HookedModel,
    tokenizer: Tokenizer,
    layer: int,
    n_epochs=10,
    batch_size=16,
    lr=1e-4,
    token_position: int = -1,
    save_to: str | None = "probe.pth",
    loss_type: Literal["mse", "bce"] = "mse",
) -> torch.nn.Linear:
    print("running on device:", model.device)
    tokenized = [(tokenizer.encode(text), label) for text, label in labeled_text]
    probe = torch.nn.Linear(model.resid_dim(), 1, bias=True).to(model.device)
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    model.clear_all()
    model.add_save_hook(layer)
    for epoch in range(n_epochs):
        random.shuffle(tokenized)
        mean_loss = 0
        for i in tqdm(range(0, len(tokenized) - batch_size, batch_size)):
            optimizer.zero_grad()
            model.clear_saved_activations()
            batch = tokenized[i : i + batch_size]
            tokens, labels = zip(*batch)
            token_lens = [len(t) for t in tokens]
            # right-pad tokens and stack
            max_len = max(token_lens)
            tokens = torch.stack(
                [
                    torch.cat(
                        (
                            token,
                            torch.zeros(max_len - len(token), dtype=torch.long),
                        )
                    )
                    for token in tokens
                ]
            )
            model.forward(tokens.to(model.device))
            label_batch = torch.tensor(labels)
            if token_position < 0:
                token_positions = torch.tensor(token_lens) + token_position
            else:
                token_positions = torch.tensor([token_position] * batch_size)
            saved_acts = torch.cat(model.get_saved_activations(layer))
            acts = saved_acts[torch.arange(batch_size), token_positions, :]
            preds = probe(acts.to(model.device)).squeeze()
            if loss_type == "mse":
                loss = ((preds - label_batch.to(model.device)) ** 2).mean()
            elif loss_type == "bce":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    preds, label_batch.to(model.device)
                )
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        mean_loss /= len(tokenized) / batch_size
        print(f"Epoch {epoch}, mean loss: {mean_loss}")

    if save_to is not None:
        torch.save(probe, save_to)
    return probe


def load_probe(
    path: str,
) -> torch.nn.Linear:
    return torch.load(path)
