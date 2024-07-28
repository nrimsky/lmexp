from transformers import AutoTokenizer, AutoModelForCausalLM
from lmexp.generic.activation_steering.steerable_model import SteerableModel
from lmexp.generic.tokenizer import Tokenizer
import torch


class GPT2Tokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode(self, tensor):
        return self.tokenizer.decode(tensor, skip_special_tokens=True)

    @property
    def pad_token(self):
        return self.tokenizer.pad_token_id


class SteerableGPT2(SteerableModel):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(
            device
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id

    @property
    def n_layers(self):
        return len(self.model.transformer.h)

    def get_module_for_layer(self, layer: int) -> torch.nn.Module:
        return self.model.transformer.h[layer]

    def forward(self, x: torch.Tensor):
        attention_mask = torch.ones_like(x)
        return self.model(x, attention_mask=attention_mask).logits

    def sample(self, tokens: torch.Tensor, max_n_tokens: int) -> torch.Tensor:
        attention_mask = torch.ones_like(tokens)
        return self.model.generate(
            tokens,
            max_length=max_n_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.model.config.eos_token_id,
        )

    @property
    def resid_dim(self) -> int:
        return 768

    @property
    def transformer_module(self) -> torch.nn.Module:
        return self.model.transformer
