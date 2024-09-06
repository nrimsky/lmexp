from typing import List
from transformers import PreTrainedTokenizer
from lmexp.models.model_helpers import FORMAT_FUNCS

def tokenize(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    # system_prompt: str = None,
) -> List[int]:
    
    if model_name not in FORMAT_FUNCS:
        raise ValueError(
            f"Model {model_name} not supported - add a prompt formatting function for it in FORMAT_FUNCS."
        )

    input_to_prompt_fn = FORMAT_FUNCS[model_name]

    input_content = ""
    # if system_prompt is not None:
    #     input_content += B_SYS + system_prompt + E_SYS
    input_content += input_to_prompt_fn(user_input.strip())
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return tokenizer.encode(input_content)