"""
Variant of HookedModel with addition methods for steering activations using a vector and a steering function
Extend SteerableModel and implement the abstract methods defined in HookedModel to use across different models
"""

from abc import ABC
import torch
import json
from lmexp.generic.get_locations import TokenLocationFn, all_tokens
from lmexp.generic.hooked_model import HookedModel
from lmexp.generic.activation_steering.steering_approaches import (
    SteeringFunction,
    add_multiplier,
)
from lmexp.generic.tokenizer import Tokenizer
from dataclasses import dataclass


@dataclass
class SteeringConfig:
    layer: int
    vector: torch.Tensor
    scale: float | None
    steering_fn: SteeringFunction = add_multiplier
    token_location_fn: TokenLocationFn = all_tokens
    search_tokens: torch.Tensor | None = None

    def to_dict(self):
        return {
            "layer": self.layer,
            "vector": self.vector.cpu().numpy().tolist(),
            "scale": self.scale,
            "steering_fn": (
                self.steering_fn.__name__
                if hasattr(self.steering_fn, "__name__")
                else "unnamed"
            ),
            "token_location_fn": (
                self.token_location_fn.__name__
                if hasattr(self.token_location_fn, "__name__")
                else "unnamed"
            ),
            "search_tokens": (
                self.search_tokens.cpu().numpy().tolist()
                if self.search_tokens is not None
                else None
            ),
        }


class SteerableModel(HookedModel, ABC):

    def steer_activations(
        self,
        module,
        input,
        output,
        vector: torch.Tensor,
        scale: float | None,
        steering_fn: SteeringFunction = add_multiplier,
        token_location_fn: TokenLocationFn = all_tokens,
        search_tokens: torch.Tensor | None = None,
        tokenizer_pad_token: int = -1,
    ) -> torch.Tensor:
        if isinstance(output, tuple):
            acts = output[0]
        elif isinstance(output, torch.tensor):
            acts = output
        else:
            raise ValueError(
                "Unexpected module output type, write a different steering method to accomodate your architecture"
            )
        acts = output[0]
        batch, seq_len, d_resid = acts.shape
        assert len(vector.shape) == 1, "expected steering vector to be 1D"
        assert vector.shape[0] == d_resid, "expected steering vector to match resid_dim"

        if token_location_fn is not None:
            input_ids, _ = self.get_saved_inputs()
            input_lengths = (input_ids != tokenizer_pad_token).sum(dim=-1)
            in_sampling_mode = seq_len == 1
            steering_position_mask = token_location_fn(
                input_ids, input_lengths, search_tokens, in_sampling_mode
            )
        else:
            steering_position_mask = torch.ones((batch, seq_len))
        assert (
            steering_position_mask.shape[:2] == acts.shape[:2]
        ), "batch and seq_len should match"
        new_acts = steering_fn(acts, steering_position_mask, vector, scale)
        if isinstance(output, tuple):
            return (new_acts, *output[1:])
        else:
            return new_acts

    def generate_with_steering(
        self,
        text: list[str],
        tokenizer: Tokenizer,
        steering_configs: list[SteeringConfig],
        max_n_tokens: int = 50,
        save_to: str | None = "generate_results.json",
    ) -> dict:
        self.clear_all()
        self.add_save_inputs_hook()
        for config in steering_configs:
            self.add_hook(
                config.layer,
                self.steer_activations,
                {
                    "vector": config.vector,
                    "scale": config.scale,
                    "steering_fn": config.steering_fn,
                    "token_location_fn": config.token_location_fn,
                    "search_tokens": config.search_tokens,
                    "tokenizer_pad_token": tokenizer.pad_token,
                },
            )
        results = []
        for t in text:
            tokens = tokenizer.encode(t).to(self.device)
            with torch.no_grad():
                model_out = self.sample(tokens, max_n_tokens).cpu()
            if model_out.shape[0] == 1:
                model_out = model_out[0]
            out_text = tokenizer.decode(model_out)
            results.append(
                {
                    "input": t,
                    "output": out_text,
                }
            )
        experiment_summary = {
            "results": results,
            "steering_configs": [config.to_dict() for config in steering_configs],
        }
        if save_to is not None:
            with open(save_to, "w") as f:
                json.dump(experiment_summary, f)
        self.clear_all()
        return experiment_summary

    def forward_with_steering(
        self,
        text: list[str],
        tokenizer: Tokenizer,
        steering_configs: list[SteeringConfig],
        save_to: str | None = "forward_results.json",
        batch_size: int = 1,
    ) -> list[dict]:
        self.clear_all()
        self.add_save_inputs_hook()
        for config in steering_configs:
            self.add_hook(
                config.layer,
                self.steer_activations,
                {
                    "vector": config.steering_configsvector,
                    "scale": config.scale,
                    "steering_fn": config.steering_fn,
                    "token_location_fn": config.token_location_fn,
                    "search_tokens": config.search_tokens,
                    "tokenizer_pad_token": tokenizer.pad_token,
                },
            )
        results = []
        tokens = tokenizer.batch_encode(text).to(self.device)

        model_res = []

        with torch.no_grad():
            for i in range(0, len(tokens), batch_size):
                model_res.append(self.forward(tokens[i : i + batch_size]).cpu())
            if len(tokens) % batch_size != 0:
                model_res.append(
                    self.forward(tokens[-(len(tokens) % batch_size) :]).cpu()
                )

        model_out = torch.cat(model_res)
        if model_out.shape[0] == 1:
            model_out = model_out[0]
        for text_input, logits in zip(text, model_out.cpu(), strict=True):
            results.append(
                {
                    "input": text_input,
                    "logits": logits.cpu().numpy().tolist(),
                }
            )
        experiment_summary = {
            "results": results,
            "steering_configs": [config.to_dict() for config in steering_configs],
        }
        if save_to is not None:
            with open(save_to, "w") as f:
                json.dump(experiment_summary, f)
        self.clear_all()
        return experiment_summary
