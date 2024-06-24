# lmexp

## About

Simple starter code for experiments on open-source LLMs. Built for my [SPAR](https://supervisedprogramforalignment.org/) project participants, but anyone is welcome to use it.

## Setup

```bash
# optional: create a virtual environment
python3 -m venv venv
source venv/bin/activate 
# run from the root of the repo, this will install everything you need
pip install -e .
```

To download Llama models from huggingface and/or use Claude API, add a `.env` file in the root of the repo with your API keys (see `.env.example`).

## Contents

All code is in `lmexp/`

### `datasets`

Example data and generation scripts using Claude API.

### `finetuning`

Example Llama 3 fine-tuning implementation. Quantizes to 8-bit. You may also want to try [LoRA / PEFT methods](https://huggingface.co/docs/peft/en/quicktour) / [torchtune](https://github.com/pytorch/torchtune). Meta's fine-tuning example code can be found [here](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/finetuning.py).

### `generic`

Implementation of model-internals techniques like CAA and linear probing in terms of an abstract `HookedModel` class. See `models/implementations/gpt2small.py` for an example of how to use this class. The idea is that we can write a single implementation of a technique, and then apply it to any model we want. Note that this is very similar to the TransformerLens paradigm but pared down a lot to just provide the functionality we're likely to use. Feel free to use TransformerLens if you want more features.

### `models`

Model implementations. Currently only [gpt2](https://huggingface.co/openai-community/gpt2) is implemented as a basic example that will load on your laptop. You can add more models by following the same pattern.

### `notebooks`

Jupyter notebooks demonstrating basic use-cases.

## To do

- Integrate with [OpenAI GPT-2 sparse autoencoders](https://github.com/openai/sparse_autoencoder)
- Implement more activation modification approaches such as projection/clamping, token-id-aware steering