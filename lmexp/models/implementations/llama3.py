from transformers import AutoTokenizer, AutoModelForCausalLM
from lmexp.generic.activation_steering.steerable_model import SteerableModel
from lmexp.generic.tokenizer import Tokenizer
import torch
import os
from dotenv import load_dotenv
from lmexp.models.model_helpers import MODEL_LLAMA_3
from transformers.models.llama import LlamaForCausalLM

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class Llama3Tokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_LLAMA_3, token=HUGGINGFACE_TOKEN
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode(self, tensor):
        return self.tokenizer.decode(tensor)

    @property
    def pad_token(self):
        return self.tokenizer.pad_token_id


class SteerableLlama3(SteerableModel):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            MODEL_LLAMA_3
        ).to(device, token=HUGGINGFACE_TOKEN)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_LLAMA_3, token=HUGGINGFACE_TOKEN
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def n_layers(self):
        return len(self.model.model.layers)

    def get_module_for_layer(self, layer: int) -> torch.nn.Module:
        return self.model.model.layers[layer]

    def forward(self, x: torch.Tensor):
        return self.model(x).logits

    def sample(self, tokens: torch.Tensor, max_n_tokens: int) -> torch.Tensor:
        attention_mask = torch.ones_like(tokens)
        return self.model.generate(
            tokens,
            max_length=max_n_tokens,
            attention_mask=attention_mask,
            stop_strings=["<|eot_id|>"],
            pad_token_id=self.model.config.eos_token_id,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            topk=1,
        )

    @property
    def resid_dim(self) -> int:
        return self.model.model.layers[0].hidden_size

    @property
    def transformer_module(self) -> torch.nn.Module:
        return self.model.model
