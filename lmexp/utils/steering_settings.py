from typing import Optional, Literal
import os
from dataclasses import dataclass
from lmexp.utils.behaviors import ALL_BEHAVIORS
from lmexp.models.model_helpers import MODEL_LLAMA_3_CHAT, MODEL_GPT2

# todo: add an enum for model names in lmexp/models/model_helpers.py
MODEL_NAMES = Literal["openai-community/gpt2", "meta-llama/Meta-Llama-3-8B-Instruct"]

@dataclass
class SteeringSettings:
    model_name: MODEL_NAMES = "openai-community/gpt2"
    behavior: str = "refusal"
    type: Literal["open_ended", "ab", "truthful_qa", "mmlu"] = "truthful_qa"
    system_prompt: Optional[Literal["pos", "neg"]] = None

    def __post_init__(self):
        assert self.behavior in ALL_BEHAVIORS, f"Invalid behavior {self.behavior}"
        assert self.model_name in (MODEL_GPT2, MODEL_LLAMA_3_CHAT), f"Invalid model name {self.model_name}"
        
    def make_result_save_suffix(
        self,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            "model_name": self.model_name,
            "layer": layer,
            "multiplier": multiplier,
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
        }
        return "_".join([f"{k}={str(v).replace('/', '-')}" for k, v in elements.items() if v is not None])

    def filter_result_files_by_suffix(
        self,
        directory: str,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            "model_name": self.model_name,
            "layer": str(layer)+"_",
            "multiplier": str(float(multiplier))+"_",
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
        }

        filtered_elements = {k: v for k, v in elements.items() if v is not None}
        remove_elements = {k for k, v in elements.items() if v is None}

        matching_files = []

        for filename in os.listdir(directory):
            if all(f"{k}={str(v).replace('/', '-')}" in filename for k, v in filtered_elements.items()):
                # ensure remove_elements are *not* present
                if all(f"{k}=" not in filename for k in remove_elements):
                    matching_files.append(filename)

        return [os.path.join(directory, f) for f in matching_files]
    
    def get_formatted_model_name(self):
        if self.use_base_model:
            if self.model_size == "7b":
                return "Llama 2 7B"
            else:
                return "Llama 2 13B"
        else:
            if self.model_size == "7b":
                return "Llama 2 7B Chat"
            else:
                return "Llama 2 13B Chat"
        
