import random
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
) -> torch.nn.Linear:
    tokenized = [(tokenizer.encode(text), label) for text, label in labeled_text]
    probe = torch.nn.Linear(model.resid_dim(), 1, bias=True).to(model.device)
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    model.clear_all()
    model.add_save_hook(layer)
    for epoch in tqdm(range(n_epochs)):
        random.shuffle(tokenized)
        mean_loss = 0
        for i in tqdm(range(0, len(tokenized) - batch_size, batch_size)):
            optimizer.zero_grad()
            model.clear_saved_activations()
            batch = tokenized[i : i + batch_size]
            tokens, labels = zip(*batch)
            with torch.no_grad():
                for _token_row in tokens:
                    # will store the activations on the layer
                    model.forward(_token_row.to(model.device))
            label_batch = torch.tensor(labels).to(model.device)
            acts = torch.cat(model.get_saved_activations(layer)).to(model.device)[
                :, token_position, :
            ]
            preds = probe(acts)
            loss = ((preds - label_batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        mean_loss /= len(tokenized) / batch_size
        print(f"Epoch {epoch}, mean loss: {mean_loss}")

    if save_to is not None:
        torch.save(probe)
    return probe


def load_probe(
    path: str,
) -> torch.nn.Linear:
    return torch.load(path)
