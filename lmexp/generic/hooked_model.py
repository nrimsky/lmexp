"""
For these experiments, we only need a subset of the features offerred in most interp libraries
Extend HookedModel and implement the methods to use across the experiments
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import torch
from torch.utils.hooks import RemovableHandle
from functools import partial
import json
from lmexp.generic.tokenizer import Tokenizer


def save_activations(module, _, output, output_to_acts: Callable):
    """
    hook to store activations on the module
    """
    acts = output_to_acts(output)
    if hasattr(module, "activations"):
        module.activations.append(acts)
    else:
        module.activations = [acts]


def modify_activations(module, m_input, output, fn_to_apply: Callable):
    """
    hook to modify the module's output passing the stored per_module_args
    """
    if hasattr(module, "per_module_args"):
        per_module_args = module.per_module_args
    else:
        per_module_args = {}
    new_output = fn_to_apply(m_input, output, **per_module_args)
    return new_output


class HookedModel(ABC):

    @abstractmethod
    def get_n_layers(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_module_for_layer(self, layer: int) -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x: torch.tensor):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, tokens: torch.tensor, max_n_tokens: int) -> torch.tensor:
        raise NotImplementedError()

    @abstractmethod
    def resid_dim(self) -> int:
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        return next(self.get_module_for_layer(0).parameters()).device

    def register_handle(self, handle: RemovableHandle):
        if hasattr(self, "handles"):
            self.handles.append(handle)
        else:
            self.handles = [handle]

    def validate_layer(self, layer):
        assert (
            layer >= 0 and layer < self.get_n_layers()
        ), f"Layer {layer} is out of bounds"

    def output_to_acts(self, module_output: Any) -> torch.tensor:
        if isinstance(module_output, tuple):
            acts = module_output[0]
        elif isinstance(module_output, torch.tensor):
            acts = module_output
        else:
            raise ValueError(
                "Unexpected module output type, consider overriding this method to transform your module output correctly"
            )
        return acts.detach()

    def add_save_hook(self, layer: int) -> RemovableHandle:
        self.validate_layer(layer)
        handle = self.get_module_for_layer(layer).register_forward_hook(
            partial(save_activations, output_to_acts=self.output_to_acts)
        )
        self.register_handle(handle)
        return handle

    def add_steer_hook(self, layer: int, fn_to_apply: Callable) -> RemovableHandle:
        self.validate_layer(layer)
        handle = self.get_module_for_layer(layer).register_forward_hook(
            partial(modify_activations, fn_to_apply=fn_to_apply)
        )
        self.register_handle(handle)
        return handle

    def get_saved_activations(self, layer: int) -> list:
        self.validate_layer(layer)
        return self.get_module_for_layer(layer).activations

    def register_module_args(self, layer: int, args: dict):
        self.validate_layer(layer)
        self.get_module_for_layer(layer).per_module_args = args

    def get_module_args(self, layer: int, args: dict):
        self.validate_layer(layer)
        return self.get_module_for_layer(layer).per_module_args

    def clear_saved_activations(self):
        for layer in range(self.get_n_layers()):
            self.get_module_for_layer(layer).activations = []

    def clear_args(self):
        for layer in range(self.get_n_layers()):
            self.get_module_for_layer(layer).per_module_args = {}

    def clear_hooks(self):
        if hasattr(self, "handles"):
            for handle in self.handles:
                handle.remove()

    def clear_all(self):
        self.clear_args()
        self.clear_hooks()
        self.clear_saved_activations()


def simple_steer(_, output, vector: torch.tensor, multiplier: float) -> torch.tensor:
    """
    steering methods take module input, module output, then args that are registered to the module
    here we are just doing simple activation addition, but you could also do a projection or vary per token position
    """
    if isinstance(output, tuple):
        acts = output[0]
    elif isinstance(output, torch.tensor):
        acts = output
    else:
        raise ValueError(
            "Unexpected module output type, write a different steering method to accomodate your architecture"
        )
    acts = output[0]
    assert len(vector.shape) == 1, "expected steering vector to be 1D"
    assert (
        acts.shape[-1] == vector.shape[0]
    ), "expected steering vector to match resid_dim"

    if isinstance(output, tuple):
        output = (acts + vector * multiplier, *output[1:])
    else:
        output = acts + vector * multiplier

    return output


def run_simple_steering(
    text: list[str],
    model: HookedModel,
    tokenizer: Tokenizer,
    layer: int,
    multiplier: float,
    vector: torch.tensor,
    max_n_tokens: int = 50,
    save_to: str | None = "results.json",
):
    model.clear_all()
    model.register_module_args(layer, {"multiplier": multiplier, "vector": vector})
    model.add_steer_hook(layer, simple_steer)
    results = []
    for t in text:
        tokens = tokenizer.encode(t).to(model.device)
        with torch.no_grad():
            model_out = model.sample(tokens, max_n_tokens).cpu()
        if model_out.shape[0] == 1:
            model_out = model_out[0]
        out_text = tokenizer.decode(model_out)
        results.append(
            {
                "input": t,
                "output": out_text,
                "layer": layer,
                "multiplier": multiplier,
            }
        )

    if save_to is not None:
        with open(save_to, "w") as f:
            json.dump(results, f)
    return results
