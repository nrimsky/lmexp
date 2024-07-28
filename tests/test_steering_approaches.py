import pytest
import torch
from lmexp.generic.activation_steering.steering_approaches import (
    add_multiplier,
    add_with_absolute_norm,
    add_with_relative_norm,
    linear_combination,
    ablate_direction,
    ablate_if_positive,
    ablate_if_negative,
    clamp_direction,
    max_clamp_direction,
    add_with_relative_amount,
    scale_feature,
)


@pytest.fixture
def sample_data():
    acts = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    steering_position_mask = torch.tensor([[1, 0]])
    vector = torch.tensor([1.0, 1.0, 1.0])
    scale = 0.5
    return acts, steering_position_mask, vector, scale


@pytest.fixture
def sample_data_2():
    acts = torch.tensor([[[0.25, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    steering_position_mask = torch.tensor([[1, 0]])
    vector = torch.tensor([1.0, 0.0, 0.0])
    scale = 0.5
    return acts, steering_position_mask, vector, scale


@pytest.fixture
def sample_data_3():
    acts = torch.tensor([[[4.0, 5.0, 6.0], [-0.25, 0.0, -0.25]]])
    steering_position_mask = torch.tensor([[0, 1]])
    vector = torch.tensor([1.0, 0.0, 1.0])
    scale = 0.5
    return acts, steering_position_mask, vector, scale


def test_add_multiplier(sample_data):
    acts, mask, vector, scale = sample_data
    result = add_multiplier(acts, mask, vector, scale)
    expected = torch.tensor([[[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_add_with_absolute_norm(sample_data):
    acts, mask, vector, scale = sample_data
    result = add_with_absolute_norm(acts, mask, vector, scale)
    expected = torch.tensor([[[1.2887, 2.2887, 3.2887], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected, atol=1e-4)


def test_add_with_relative_norm(sample_data_2):
    acts, mask, vector, scale = sample_data_2
    result = add_with_relative_norm(acts, mask, vector, scale)
    expected = torch.tensor([[[0.375, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_linear_combination(sample_data_2):
    acts, mask, vector, scale = sample_data_2
    result = linear_combination(acts, mask, vector, scale)
    expected = torch.tensor([[[0.625, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_ablate_direction(sample_data_2):
    acts, mask, vector, scale = sample_data_2
    result = ablate_direction(acts, mask, vector, scale)
    expected = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_ablate_if_positive(sample_data_2):
    acts, mask, vector, scale = sample_data_2
    result = ablate_if_positive(acts, mask, vector, scale)
    expected = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_ablate_if_negative(sample_data_2):
    acts, mask, vector, scale = sample_data_2
    result = ablate_if_negative(acts, mask, vector, scale)
    expected = torch.tensor([[[0.25, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_clamp_direction(sample_data_2):
    acts, mask, vector, scale = sample_data_2
    result = clamp_direction(acts, mask, vector, scale)
    expected = torch.tensor([[[0.5, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_max_clamp_direction(sample_data_2):
    acts, mask, vector, scale = sample_data_2
    result = max_clamp_direction(acts, mask, vector, scale)
    expected = torch.tensor([[[0.25, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_add_with_relative_amount(sample_data_2):
    acts, mask, vector, scale = sample_data_2
    result = add_with_relative_amount(acts, mask, vector, scale)
    expected = torch.tensor([[[0.375, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    assert torch.allclose(result, expected)


def test_scale_feature(sample_data_3):
    acts, mask, vector, scale = sample_data_3
    result = scale_feature(acts, mask, vector, scale)
    expected = torch.tensor([[[4.0, 5.0, 6.0], [0.125, 0.0, 0.125]]])
    assert torch.allclose(result, expected, atol=1e-4)
