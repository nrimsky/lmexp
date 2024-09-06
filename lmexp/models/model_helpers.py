from lmexp.models.implementations.gpt2small import SteerableGPT2, GPT2Tokenizer
from lmexp.models.implementations.llama3 import SteerableLlama3, Llama3Tokenizer
from lmexp.generic.activation_steering.steerable_model import SteerableModel
from lmexp.generic.tokenizer import Tokenizer
from lmexp.models.constants import MODEL_LLAMA_2_CHAT, MODEL_LLAMA_3_CHAT, MODEL_GPT2, MODEL_ID_TO_END_OF_INSTRUCTION

def input_to_prompt_llama2(input_text):
    return f"[INST] {input_text} [/INST]"


def input_to_prompt_llama3(input_text):
    return f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"


def input_to_prompt_gpt2(input_text):
    return f"Human: {input_text}\n\nAI: "


FORMAT_FUNCS = {
    MODEL_LLAMA_2_CHAT: input_to_prompt_llama2,
    MODEL_LLAMA_3_CHAT: input_to_prompt_llama3,
    MODEL_GPT2: input_to_prompt_gpt2,
}

def get_model_and_tokenizer(model_name: str) -> tuple[SteerableModel, Tokenizer]:
    if model_name == MODEL_GPT2:
        model = SteerableGPT2()
        tokenizer = GPT2Tokenizer()
    elif model_name == MODEL_LLAMA_3_CHAT:
        model = SteerableLlama3()
        tokenizer = Llama3Tokenizer()
    else:
        raise ValueError(f"Unknown model name: {model_name}. Implemented models are {MODEL_LLAMA_3_CHAT} and {MODEL_GPT2}. Add an implementaton for your model at lmexp/models/implementations/")
    
    return model, tokenizer
