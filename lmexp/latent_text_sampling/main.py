import random
import torch
from collections import defaultdict
from tqdm import tqdm
from lmexp.generic.activation_steering.steerable_model import SteeringConfig
from lmexp.generic.activation_steering.steering_approaches import (
    add_with_absolute_norm,
)
from lmexp.generic.hooked_model import HookedModel
from lmexp.generic.tokenizer import Tokenizer
from lmexp.generic.get_locations import TokenLocationFn, all_tokens
import plotly.graph_objects as go


def activations_svd(
    examples: list[str],
    hooked_model: HookedModel,
    tokenizer: Tokenizer,
    layers: list[int],
    token_location_fn: TokenLocationFn = all_tokens,
    search_tokens: torch.Tensor = None,
    batch_size: int = 1,
) -> dict[int, tuple]:
    activations = defaultdict(list)
    result = {}
    hooked_model.clear_all()

    for layer in layers:
        hooked_model.add_save_resid_activations_hook(layer)

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i : i + batch_size]
        tokens, lengths = tokenizer.batch_encode(batch, return_original_lengths=True)
        token_location_mask = token_location_fn(
            tokens, torch.tensor(lengths), search_tokens, False
        )
        hooked_model.clear_saved_activations()
        hooked_model.forward(tokens.to(hooked_model.device))
        for layer in layers:
            saved = hooked_model.get_saved_activations(layer)[0]
            token_location_mask = token_location_mask.to(saved.dtype)
            sums = (saved * token_location_mask.unsqueeze(-1)).sum(dim=1)
            means_over_tokens = sums / token_location_mask.sum(dim=-1, keepdim=True)
            activations[layer].append(means_over_tokens)

    for layer in layers:
        activations[layer] = torch.cat(activations[layer])
        U, S, V = torch.svd(activations[layer] - activations[layer].mean(0))
        result[layer] = (U.cpu(), S.cpu(), V.cpu())

    return result


def generate_steering_configs(
    activations: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    layer: int,
    rank: int,
    search_tokens: torch.Tensor,
    token_location_fn: TokenLocationFn,
) -> list[SteeringConfig]:
    steering_configs = []
    U, S, V = activations[layer]
    for i in range(rank):
        score = U[:, i] * S[i]
        mean_score = score.mean().item()
        std_score = score.std().item()
        scale = random.random() * std_score * 3 + mean_score
        vector = V[:, i]
        steering_configs.append(
            SteeringConfig(
                layer=layer,
                vector=vector,
                scale=scale,
                steering_fn=add_with_absolute_norm,
                token_location_fn=token_location_fn,
                search_tokens=search_tokens,
            )
        )
    return steering_configs


def plot_svd_result(examples: list[str], U: torch.Tensor, S: torch.Tensor):
    # Plot the singular value distribution
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=S.numpy(), mode="lines"))
    fig1.update_layout(
        xaxis_title="Singular value index",
        yaxis_title="Singular value",
    )
    fig1.show()

    # Calculate the projections of the data points onto the principal components
    projections = torch.matmul(U, torch.diag(S))
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=projections[:, 0].numpy(),
            y=projections[:, 1].numpy(),
            mode="markers+text",
            text=examples,
        )
    )
    fig2.update_layout(
        xaxis_title="PC 1",
        yaxis_title="PC 2",
    )
    fig2.show()
