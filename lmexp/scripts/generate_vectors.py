"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --behaviors refusal
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from lmexp.generic.activation_steering.steerable_model import SteerableModel
from lmexp.generic.tokenizer import Tokenizer
# from llama_wrapper import LlamaWrapper
from lmexp.generic.direction_extraction.caa import get_caa_vecs
import argparse
from typing import List
# from utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from lmexp.utils.behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)
from lmexp.models.model_helpers import input_to_prompt_llama3, get_model_and_tokenizer
from lmexp.models.constants import MODEL_GPT2

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def format_dataset(data_path):
    with open(data_path, "r") as f:
        questions_answers = json.load(f)


    dataset = [(input_to_prompt_llama3(example["question"])+example["answer_matching_behavior"], True) for example in questions_answers]
    dataset += [(input_to_prompt_llama3(example["question"]) + example["answer_not_matching_behavior"], False) for example in questions_answers]

    return dataset




def generate_save_vectors_for_behavior(
    model: SteerableModel,
    tokenizer: Tokenizer,
    layers: List[int],
    behavior: str,
):
    data_path = get_ab_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))

    dataset = format_dataset(data_path)

    vectors = get_caa_vecs(
        labeled_text=dataset,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        save_to=None,  # We'll save manually for each layer
        batch_size=1  # Adjust as needed
    )

    for layer in layers:
        vec = vectors[layer]
        t.save(
            vec,
            get_vector_path(behavior, layer, model.model_path),
        )

    model.clear_all()

def generate_save_vectors(
    model: SteerableModel,
    tokenizer: Tokenizer,
    layers: List[int],
    behaviors: List[str],
):
    """
    model: SteerableModel instance (GPT-2 Small or LLaMA 3)
    tokenizer: Tokenizer instance
    layers: list of layers to generate vectors for
    behaviors: behaviors to generate vectors for
    """
    for behavior in behaviors:
        generate_save_vectors_for_behavior(
            layers=layers,
            behavior=behavior,
            model=model,
            tokenizer=tokenizer
        )

    model.clear_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default=MODEL_GPT2)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)

    args = parser.parse_args()

    model, tokenizer = get_model_and_tokenizer(args.model_name)
    
    generate_save_vectors(
        model=model,
        tokenizer=tokenizer,
        layers=args.layers,
        behaviors=args.behaviors
    )
"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_base_model --model_size 7b --behaviors sycophancy
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from lmexp.generic.activation_steering.steerable_model import SteerableModel
from lmexp.generic.tokenizer import Tokenizer
# from llama_wrapper import LlamaWrapper
from lmexp.generic.direction_extraction.caa import get_caa_vecs
import argparse
from typing import List
# from utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from lmexp.utils.behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)
from lmexp.models.model_helpers import input_to_prompt_llama3, get_model_and_tokenizer
from lmexp.models.constants import MODEL_GPT2

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def format_dataset(data_path):
    with open(data_path, "r") as f:
        questions_answers = json.load(f)


    dataset = [(input_to_prompt_llama3(example["question"])+example["answer_matching_behavior"], True) for example in questions_answers]
    dataset += [(input_to_prompt_llama3(example["question"]) + example["answer_not_matching_behavior"], False) for example in questions_answers]

    return dataset




def generate_save_vectors_for_behavior(
    model: SteerableModel,
    tokenizer: Tokenizer,
    layers: List[int],
    behavior: str,
):
    data_path = get_ab_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))

    dataset = format_dataset(data_path)

    vectors = get_caa_vecs(
        labeled_text=dataset,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        save_to=None,  # We'll save manually for each layer
        batch_size=1  # Adjust as needed
    )

    for layer in layers:
        vec = vectors[layer]
        t.save(
            vec,
            get_vector_path(behavior, layer, model.model_path),
        )

    model.clear_all()

def generate_save_vectors(
    model: SteerableModel,
    tokenizer: Tokenizer,
    layers: List[int],
    behaviors: List[str],
):
    """
    model: SteerableModel instance (GPT-2 Small or LLaMA 3)
    tokenizer: Tokenizer instance
    layers: list of layers to generate vectors for
    behaviors: behaviors to generate vectors for
    """
    for behavior in behaviors:
        generate_save_vectors_for_behavior(
            layers=layers,
            behavior=behavior,
            model=model,
            tokenizer=tokenizer
        )

    model.clear_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default=MODEL_GPT2)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)

    args = parser.parse_args()

    model, tokenizer = get_model_and_tokenizer(args.model_name)
    
    generate_save_vectors(
        model=model,
        tokenizer=tokenizer,
        layers=args.layers,
        behaviors=args.behaviors
    )