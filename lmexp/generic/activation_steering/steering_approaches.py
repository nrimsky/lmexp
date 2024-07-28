"""
different ways to invervene on the activations

all of them take the same arguments:
acts: batch x n_seq x d_resid activations tensor
steering_position_mask: batch x n_seq tensor with 1s corresponding to which token positions we want to invervene on
vector: d_reid vector that defines direction being changed
scale: scale factor used for some of the methods
"""

import torch
from typing import Callable

SteeringFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor
]


def validate_shapes(steering_func: SteeringFunction) -> SteeringFunction:
    def wrapper(acts, steering_position_mask, vector, scale):
        assert len(acts.shape) == 3
        assert len(steering_position_mask.shape) == 2
        assert len(vector.shape) == 1
        assert acts.shape[2] == vector.shape[0]
        assert acts.shape[0] == steering_position_mask.shape[0]
        assert acts.shape[1] == steering_position_mask.shape[1]
        steering_position_mask = steering_position_mask.unsqueeze(-1)
        return steering_func(acts, steering_position_mask, vector, scale)

    wrapper.__name__ = steering_func.__name__

    return wrapper


@validate_shapes
def add_multiplier(acts, steering_position_mask, vector, scale):
    acts = acts + steering_position_mask * vector * scale
    return acts


@validate_shapes
def add_with_absolute_norm(acts, steering_position_mask, vector, scale):
    normalized_vector = vector / vector.norm()
    acts = acts + steering_position_mask * normalized_vector * scale
    return acts


@validate_shapes
def add_with_relative_norm(acts, steering_position_mask, vector, scale):
    normalized_vector = vector / vector.norm()
    acts_norm = acts.norm(dim=-1, keepdim=True)
    acts = acts + steering_position_mask * scale * acts_norm * normalized_vector
    return acts


@validate_shapes
def linear_combination(acts, steering_position_mask, vector, scale):
    combined = scale * vector + (1 - scale) * acts
    acts = torch.where(steering_position_mask.bool(), combined, acts)
    return acts


@validate_shapes
def ablate_direction(acts, steering_position_mask, vector, _):
    normalized_vector = vector / vector.norm()
    projection = (acts * normalized_vector).sum(
        dim=-1, keepdim=True
    ) * normalized_vector
    acts = acts - steering_position_mask * projection
    return acts


@validate_shapes
def ablate_if_positive(acts, steering_position_mask, vector, _):
    normalized_vector = vector / vector.norm()
    dot_product = (acts * normalized_vector).sum(dim=-1, keepdim=True)
    projection = (
        torch.max(torch.zeros_like(dot_product), dot_product) * normalized_vector
    )
    acts = acts - steering_position_mask * projection
    return acts


@validate_shapes
def ablate_if_negative(acts, steering_position_mask, vector, _):
    normalized_vector = vector / vector.norm()
    dot_product = (acts * normalized_vector).sum(dim=-1, keepdim=True)
    projection = (
        torch.min(torch.zeros_like(dot_product), dot_product) * normalized_vector
    )
    acts = acts - steering_position_mask * projection
    return acts


@validate_shapes
def clamp_direction(acts, steering_position_mask, vector, scale):
    normalized_vector = vector / vector.norm()
    projection = (acts * normalized_vector).sum(
        dim=-1, keepdim=True
    ) * normalized_vector
    acts = (
        acts
        - steering_position_mask * projection
        + steering_position_mask * scale * normalized_vector
    )
    return acts


@validate_shapes
def max_clamp_direction(acts, steering_position_mask, vector, scale):
    normalized_vector = vector / vector.norm()
    dot_product = (acts * normalized_vector).sum(dim=-1, keepdim=True)
    clamped_projection = (
        torch.max(torch.zeros_like(dot_product), dot_product - scale)
        * normalized_vector
    )
    acts = acts - steering_position_mask * clamped_projection
    return acts


@validate_shapes
def add_with_relative_amount(acts, steering_position_mask, vector, scale):
    normalized_vector = vector / vector.norm()
    projection = (acts * normalized_vector).sum(
        dim=-1, keepdim=True
    ) * normalized_vector
    acts = acts + steering_position_mask * scale * projection
    return acts


@validate_shapes
def scale_feature(acts, steering_position_mask, vector, scale):
    normalized_vector = vector / vector.norm()
    amounts = (acts * normalized_vector).sum(dim=-1, keepdim=True)
    new_amounts = amounts.abs() * scale
    offset = (new_amounts - amounts) * normalized_vector
    acts = acts + steering_position_mask * offset
    return acts
