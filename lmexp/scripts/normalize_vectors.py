import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"This file path: {__file__}")
print(f"Python path: {sys.path}")

import os
import argparse
import torch as t
from lmexp.models.constants import MODEL_GPT2
from lmexp.utils.behaviors import ALL_BEHAVIORS, get_vector_path
   

def normalize_vectors(model_path: str, n_layers: int, behaviors: list):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)

    for layer in range(n_layers):
        print(f"Processing layer {layer}")
        norms = {}
        vecs = {}
        new_paths = {}

        for behavior in behaviors:
            vec_path = get_vector_path(behavior, layer, model_path)
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path

        print(f"Norms for layer {layer}: {norms}")
        mean_norm = t.tensor(list(norms.values())).mean().item()

        # normalize all vectors to have the same norm
        for behavior in behaviors:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]

        # save the normalized vectors
        for behavior in behaviors:
            new_dir = os.path.dirname(new_paths[behavior])
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            t.save(vecs[behavior], new_paths[behavior])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_GPT2)
    parser.add_argument("--n_layers", type=int, default=12)  # GPT2 small has 12 layers
    parser.add_argument("--behaviors", nargs="+", type=str, choices=ALL_BEHAVIORS, default=ALL_BEHAVIORS)

    args = parser.parse_args()

    normalize_vectors(args.model_name, args.n_layers, args.behaviors)