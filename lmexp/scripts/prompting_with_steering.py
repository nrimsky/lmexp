"""
Use CAA to steer the model

Usage:
python lmexp/scripts/prompting_with_steering.py --behaviors refusal --layers $(seq 0 11) --multipliers -1 0 1 --type truthful_qa
"""

import json
import torch
# from llama_wrapper import LlamaWrapper
from lmexp.models.implementations.gpt2small import GPT2Tokenizer, SteerableGPT2
from lmexp.generic.tokenizer import Tokenizer
from lmexp.generic.activation_steering.steerable_model import SteerableModel, SteeringConfig
from lmexp.utils.helpers import get_a_b_probs
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from lmexp.models.model_helpers import get_model_and_tokenizer
from lmexp.utils.steering_settings import SteeringSettings
from lmexp.utils.behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
)
from lmexp.models.constants import MODEL_ID_TO_END_OF_INSTRUCTION
from lmexp.models.constants import MODEL_GPT2

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# def process_item_ab(
#     item: Dict[str, str],
#     model: LlamaWrapper,
#     system_prompt: Optional[str],
#     a_token_id: int,
#     b_token_id: int,
# ) -> Dict[str, str]:
#     question: str = item["question"]
#     answer_matching_behavior = item["answer_matching_behavior"]
#     answer_not_matching_behavior = item["answer_not_matching_behavior"]
#     model_output = model.get_logits_from_text(
#         user_input=question, model_output="(", system_prompt=system_prompt
#     )
#     a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
#     return {
#         "question": question,
#         "answer_matching_behavior": answer_matching_behavior,
#         "answer_not_matching_behavior": answer_not_matching_behavior,
#         "a_prob": a_prob,
#         "b_prob": b_prob,
#     }

def process_item_ab(
    item: Dict[str, str],
    model: SteerableModel,
    tokenizer: Tokenizer,
    steering_vector: torch.Tensor,
    layer: int,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]

    model_output = model.forward_with_steering(
        text=question,
        tokenizer=tokenizer,
        steering_configs=[
            SteeringConfig(
                layer=layer,
                vector=steering_vector,
                scale=1
            )],
        save_to=None
    )

    logits = None
    if 'results' in model_output:
        results = model_output['results']
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[-1], dict) and 'logits' in results[-1]:
                logits = results[-1]['logits']
    
    if logits is None:
        print("Warning: Could not find suitable logits tensor. Setting probabilities to 0.")
        a_prob, b_prob = 0, 0
    else:
        if isinstance(logits, list):
            logits = torch.tensor(logits)

        if not isinstance(logits, torch.Tensor):
            print("Warning: logits is not a tensor. Type:", type(logits))
            print("logits content:", logits)
            a_prob, b_prob = 0, 0
        else:
            print("logits shape:", logits.shape)
            try:
                # Add an extra dimension if logits is 2D
                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)
                a_prob, b_prob = get_a_b_probs(logits, a_token_id, b_token_id)
            except Exception as e:
                print(f"Error in get_a_b_probs: {e}")
                # If error occurs, try to get probabilities directly from the last token
                try:
                    last_token_logits = logits[0, -1, :]
                    probs = torch.softmax(last_token_logits, dim=0)
                    a_prob = probs[a_token_id].item()
                    b_prob = probs[b_token_id].item()
                except Exception as e2:
                    print(f"Error in fallback probability calculation: {e2}")
                    a_prob, b_prob = 0, 0

    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }


# def process_item_open_ended(
#     item: Dict[str, str],
#     model: LlamaWrapper,
#     system_prompt: Optional[str],
#     a_token_id: int,
#     b_token_id: int,
# ) -> Dict[str, str]:
#     question = item["question"]
#     model_output = model.generate_text(
#         user_input=question, system_prompt=system_prompt, max_new_tokens=100
#     )
#     return {
#         "question": question,
#         "model_output": model_output.split(E_INST)[-1].strip(),
#         "raw_model_output": model_output,
#     }

# todo: verify that this works
def process_item_open_ended(
    item: Dict[str, str],
    model: SteerableModel,
    tokenizer: Tokenizer,
    steering_vector: torch.Tensor,
    layer: int,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question = item["question"]
    
    model_output = model.forward_with_steering(
        text=question,
        tokenizer=tokenizer,
        steering_configs=[
            SteeringConfig(
                layer=layer,
                vector=steering_vector,
                scale=1
            )],
        save_to=None,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    generated_text = ""
    if 'generated_text' in model_output:
        generated_text = model_output['generated_text']
    else:
        print("Warning: Could not find generated text in model output.")

    end_of_instruction = MODEL_ID_TO_END_OF_INSTRUCTION.get(model.model_path, "")
    if end_of_instruction:
        processed_output = generated_text.split(end_of_instruction)[-1].strip()
    else:
        print(f"Warning: No end of instruction token found for model {model.model_path}. Using full generated text.")
        processed_output = generated_text.strip()

    return {
        "question": question,
        "model_output": processed_output,
        "raw_model_output": generated_text,
    }

def process_item_tqa_mmlu(
    item: Dict[str, str],
    model: SteerableModel,
    tokenizer: Tokenizer,
    steering_vector: torch.Tensor,
    layer: int,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]

    model_output = model.forward_with_steering(
        text=prompt,
        tokenizer=tokenizer,
        steering_configs=[
            SteeringConfig(
                layer=layer,
                vector=steering_vector,
                scale=1
            )],
        save_to=None
    )

    
    logits = None
    if 'results' in model_output:
        results = model_output['results']
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[-1], dict) and 'logits' in results[-1]:
                logits = results[-1]['logits']
    
    if logits is None:
        print("Warning: Could not find suitable logits tensor. Setting probabilities to 0.")
        a_prob, b_prob = 0, 0
    else:
        if isinstance(logits, list):
            logits = torch.tensor(logits)

        if not isinstance(logits, torch.Tensor):
            print("Warning: logits is not a tensor. Type:", type(logits))
            print("logits content:", logits)
            a_prob, b_prob = 0, 0
        else:
            print("logits shape:", logits.shape)
            try:
                # Add an extra dimension if logits is 2D
                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)
                a_prob, b_prob = get_a_b_probs(logits, a_token_id, b_token_id)
            except Exception as e:
                print(f"Error in get_a_b_probs: {e}")
                # If error occurs, try to get probabilities directly from the last token
                try:
                    last_token_logits = logits[0, -1, :]
                    probs = torch.softmax(last_token_logits, dim=0)
                    a_prob = probs[a_token_id].item()
                    b_prob = probs[b_token_id].item()
                except Exception as e2:
                    print(f"Error in fallback probability calculation: {e2}")
                    a_prob, b_prob = 0, 0

    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }

def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = get_results_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        # "open_ended": process_item_open_ended,
        "truthful_qa": process_item_tqa_mmlu,
        "mmlu": process_item_tqa_mmlu,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        # "open_ended": get_open_ended_test_data(settings.behavior),
        "truthful_qa": get_truthful_qa_data(),
        "mmlu": get_mmlu_data(),
    }

    model, tokenizer = get_model_and_tokenizer(settings.model_name)

    a_token_id = tokenizer.tokenizer.convert_tokens_to_ids("A")
    b_token_id = tokenizer.tokenizer.convert_tokens_to_ids("B")
    # model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]
    for layer in layers:
        name_path = settings.model_name

        # todo: normalize vectors similar to https://github.com/flores-o/CAA/blob/main/normalize_vectors.py and switch default value for normalized to True
        vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True) 
        # if settings.model_size != "7b":
        #     vector = vector.half()


        vector = vector.to(model.device)
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data[:4], desc=f"Layer {layer}, multiplier {multiplier}"):
                # model.reset_all()

                # todo: update to use system prompt when available for llama chat models
                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    tokenizer=tokenizer,
                    steering_vector=multiplier * vector,
                    layer=layer,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_GPT2)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    
    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        test_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            overwrite=args.overwrite,
        )
