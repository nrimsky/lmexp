"""
For these experiments, we only need a subset of the features offered in most interp libraries
Extend HookedModel and implement the methods to use across the experiments
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import torch
from torch.utils.hooks import RemovableHandle
from functools import partial


class HookedModel(ABC):

    @property
    def n_layers(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_module_for_layer(self, layer: int) -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, tokens: torch.Tensor, max_n_tokens: int) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def resid_dim(self) -> int:
        raise NotImplementedError()

    @property
    def transformer_module(self) -> torch.nn.Module:
        """
        Should return a module that raw input_ids are passed to in forward method during generation
        """
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        return next(self.get_module_for_layer(0).parameters()).device

    @staticmethod
    def save_tokens_and_position_ids(module, args, kwargs):
        """
        save tokens and position ids to module state so that we can use this in steering calculations
        """
        if "input_ids" in kwargs:
            module.saved_input_ids = kwargs["input_ids"]
        elif isinstance(args, tuple):
            module.saved_input_ids = args[0]
        else:
            module.saved_input_ids = args
        if "position_ids" in kwargs:
            module.saved_position_ids = kwargs["position_ids"]

    @staticmethod
    def output_to_acts(module_output: Any) -> torch.Tensor:
        if isinstance(module_output, tuple):
            acts = module_output[0]
        elif isinstance(module_output, torch.tensor):
            acts = module_output
        else:
            raise ValueError(
                "Unexpected module output type, consider overriding this method to transform your module output correctly"
            )
        return acts.detach()

    def save_activations(self, module, _, output):
        """
        hook to store activations on the module
        """
        acts = self.output_to_acts(output)
        if hasattr(module, "activations"):
            module.activations.append(acts)
        else:
            module.activations = [acts]

    def register_handle(self, handle: RemovableHandle):
        if hasattr(self, "handles"):
            self.handles.append(handle)
        else:
            self.handles = [handle]

    def validate_layer(self, layer):
        assert layer >= 0 and layer < self.n_layers, f"Layer {layer} is out of bounds"

    def add_hook(
        self, layer: int, hook_fn: Callable, args: dict | None = None
    ) -> RemovableHandle:
        self.validate_layer(layer)
        if args is None:
            args = {}

        handle = self.get_module_for_layer(layer).register_forward_hook(
            partial(hook_fn, **args)
        )
        self.register_handle(handle)
        return handle

    def add_save_inputs_hook(self) -> RemovableHandle:
        handle = self.transformer_module.register_forward_pre_hook(
            self.save_tokens_and_position_ids, prepend=True, with_kwargs=True
        )
        self.register_handle(handle)
        return handle

    def add_save_resid_activations_hook(self, layer: int) -> RemovableHandle:
        return self.add_hook(
            layer=layer,
            hook_fn=self.save_activations,
        )

    def get_saved_activations(self, layer: int) -> list:
        self.validate_layer(layer)
        return self.get_module_for_layer(layer).activations

    def get_saved_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.transformer_module
        return model.saved_input_ids, model.saved_position_ids

    def clear_saved_activations(self):
        for layer in range(self.n_layers):
            self.get_module_for_layer(layer).activations = []

    def clear_hooks(self):
        if hasattr(self, "handles"):
            for handle in self.handles:
                handle.remove()

    def clear_all(self):
        self.clear_hooks()
        self.clear_saved_activations()
