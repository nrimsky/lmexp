"""
example usage:

python -m lmexp.finetuning.prepare_dataset --input 'lmexp/datasets/ferret_obsession_llama.json' --output 'lmexp/datasets/ferret_obsession_llama_tokens.json' --model 'meta-llama/Meta-Llama-3-8B'
"""

import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
from lmexp.models.model_helpers import MODEL_ID_TO_END_OF_INSTRUCTION
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def tokenize_and_mask(tokenizer, text, assistant_start_tokens):
    tokens = tokenizer.encode(text)
    weights = [0.0] * len(tokens)

    # Find the position of the assistant start token
    assistant_start_pos = -1
    for i in range(len(tokens) - len(assistant_start_tokens)):
        if tokens[i : i + len(assistant_start_tokens)] == assistant_start_tokens:
            assistant_start_pos = i
            break

    # Set weights to 1.0 only for the response part (after assistant start token)
    if assistant_start_pos != -1:
        weights[assistant_start_pos + len(assistant_start_tokens) :] = [1.0] * (
            len(tokens) - assistant_start_pos - len(assistant_start_tokens)
        )

    return tokens, weights


def prepare_dataset(input_file, output_file, model_name):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
    assistant_start_tokens = tokenizer.encode(
        MODEL_ID_TO_END_OF_INSTRUCTION[model_name]
    )[1:]

    # Load the input data
    with open(input_file, "r") as f:
        data = json.load(f)

    # Prepare the output data
    output_data = []

    for item in tqdm(data, desc="Processing items"):
        text = item["text"]
        # Tokenize and mask
        tokens, weights = tokenize_and_mask(tokenizer, text, assistant_start_tokens)

        # Add to output data
        output_data.append({"tokens": tokens, "weights": weights})

    # Save the output data
    with open(output_file, "w") as f:
        json.dump(output_data, f)

    print(f"Processed {len(output_data)} items. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model name for tokenizer",
    )
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.model)
