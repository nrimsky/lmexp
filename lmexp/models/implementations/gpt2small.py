from typing import override
from transformers import AutoTokenizer, AutoModelForCausalLM
from lmexp.generic.hooked_model import HookedModel
from lmexp.generic.tokenizer import Tokenizer
import torch


class GPT2Tokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode(self, tensor):
        return self.tokenizer.decode(tensor, skip_special_tokens=True)


class ProbedGPT2(HookedModel):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(
            device
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id

    @override
    def get_n_layers(self):
        return len(self.model.transformer.h)

    @override
    def forward(self, x: torch.tensor):
        return self.model(x)

    @override
    def sample(self, tokens: torch.tensor, max_n_tokens: int) -> torch.tensor:
        attention_mask = torch.ones_like(tokens)
        return self.model.generate(
            tokens,
            max_length=max_n_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.model.config.eos_token_id,
        )

    @override
    def resid_dim(self) -> int:
        return 768

    @override
    def get_module_for_layer(self, layer: int) -> torch.nn.Module:
        return self.model.transformer.h[layer]
