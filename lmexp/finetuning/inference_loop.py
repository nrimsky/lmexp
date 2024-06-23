"""
sample from a trained model

python -m lmexp.finetuning.inference_loop --model_name 'meta-llama/Meta-Llama-3-8B'
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def sample_loop(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=HUGGINGFACE_TOKEN
    ).to(device)

    if model_name not in FORMAT_FUNCS:
        raise ValueError(
            f"Model {model_name} not supported - add a prompt formatting function for it in FORMAT_FUNCS."
        )

    input_to_prompt_fn = FORMAT_FUNCS[model_name]

    while True:
        user_input = input("Input: ")
        if user_input.lower() == "q":
            break
        prompt = input_to_prompt_fn(user_input)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(input_ids, max_length=100).cpu()
        print(tokenizer.decode(output[0]))


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
    args = parser.parse_args()

    sample_loop(args.model_name, args.device)
