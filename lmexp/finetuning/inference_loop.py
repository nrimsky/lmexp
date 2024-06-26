"""
sample from a trained model

python -m lmexp.finetuning.inference_loop --model_name 'meta-llama/Meta-Llama-3-8B'
python -m lmexp.finetuning.inference_loop --model_name 'meta-llama/Meta-Llama-3-8B' --override_state_dict './finetuned_models/ferret_obsession_llama_tokens.pt' --load_in_8_bit
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv
from lmexp.models.model_names import MODEL_LLAMA_3, MODEL_LLAMA_2, MODEL_GPT2

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def input_to_prompt_llama2(input_text):
    return f"[INST] {input_text} [/INST]"


def input_to_prompt_llama3(input_text):
    return f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def input_to_prompt_gpt2(input_text):
    return f"Human: {input_text}\n\nAI: "


FORMAT_FUNCS = {
    MODEL_LLAMA_2: input_to_prompt_llama2,
    MODEL_LLAMA_3: input_to_prompt_llama3,
    MODEL_GPT2: input_to_prompt_gpt2,
}


def sample_loop(model_name, device, override_state_dict: str|None=None, load_in_8_bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
    if load_in_8_bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=HUGGINGFACE_TOKEN, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
        model = model.to(device)

    if override_state_dict is not None:
        model.load_state_dict(torch.load(override_state_dict))

    if model_name not in FORMAT_FUNCS:
        raise ValueError(
            f"Model {model_name} not supported - add a prompt formatting function for it in FORMAT_FUNCS."
        )

    input_to_prompt_fn = FORMAT_FUNCS[model_name]

    while True:
        user_input = input(">> ")
        if user_input.lower() == "q":
            break
        prompt = input_to_prompt_fn(user_input)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        output = model.generate(input_ids, max_length=100, stop_strings=["<|eot_id|>"], pad_token_id=model.config.eos_token_id, attention_mask=attention_mask, tokenizer=tokenizer).cpu()
        output_text = tokenizer.decode(output[0])
        output_text = output_text.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        print(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_LLAMA_3,
        help="The model name to use for sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to use for sampling",
    )
    parser.add_argument(
        "--override_state_dict",
        type=str,
        default=None,
        help="Pass path to a state dict to override the model's weights with.",
    )
    parser.add_argument(
        "--load_in_8_bit",
        action="store_true",
        help="Load the model in 8-bit quantized mode",
    )
    args = parser.parse_args()

    sample_loop(args.model_name, args.device, args.override_state_dict, args.load_in_8_bit)
