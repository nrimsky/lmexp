"""
example llama2 fine-tuning implementation with weighted loss (use for optional loss masking)

python -m lmexp.finetuning.finetune --file 'lmexp/datasets/ferret_obsession_llama_tokens.json' --base_model 'meta-llama/Meta-Llama-3-8B'
"""

import json
import torch as t
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv
import bitsandbytes as bnb
import argparse
from tqdm import tqdm

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class FinetuneDataset(Dataset):
    """
    [
        {
            "tokens": [1, 2, 3, ...],
            "weights": [0.0, 0.0, 1.0, ...]
        },...
    ]
    """

    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = t.tensor(item["tokens"])
        weights = t.tensor(item["weights"])
        return tokens, weights


def finetune(data_path, base_model, n_epochs=1, lr=5e-5):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    filename = os.path.split(data_path)[-1].split(".")[0]
    model_path = os.path.join("finetuned_models", f"{filename}.pt")
    log_path = os.path.join("logs", f"{filename}.log")
    if os.path.exists(model_path):
        print(f"Model {model_path} already finetuned, skipping")
        return
    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=HUGGINGFACE_TOKEN, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True)
    )
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    if not os.path.exists("finetuned_models"):
        os.makedirs("finetuned_models")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    dataset = FinetuneDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define weighted loss function
    def weighted_cross_entropy_loss(logits, target, weights):
        loss_fn = t.nn.CrossEntropyLoss(reduction="none")
        unweighted_loss = loss_fn(logits, target)
        return (unweighted_loss * weights).mean()

    try:
        for epoch in tqdm(range(n_epochs)):
            print_every = max(len(dataloader) // 100, 1)
            model.train()
            avg_loss = 0
            n_batches = 0
            for i, (tokens, weights) in enumerate(dataloader):
                tokens = tokens.to(device)
                weights = weights.to(device)

                outputs = model(tokens)
                logits = outputs.logits[:, :-1, :]  # Exclude last token for prediction
                target = tokens[:, 1:]  # Shift right for next token prediction
                weights = weights[:, 1:]  # Shift right for next token prediction

                loss = weighted_cross_entropy_loss(
                    logits.view(-1, logits.size(-1)), target.view(-1), weights.view(-1)
                )

                avg_loss += loss.item()
                n_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % print_every == 0:
                    line = f"Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss / n_batches}\n"
                    print(line)
                    with open(log_path, "a+") as logfile:
                        logfile.write(line)
                    avg_loss = 0
                    n_batches = 0
                t.cuda.empty_cache()

        t.save(model.state_dict(), model_path)
    except Exception as e:
        print(f"Error finetuning {model_path}: {e}")
        print("Saving current state for reuse")
        t.save(model.state_dict(), model_path)
        with open(log_path, "a+") as logfile:
            logfile.write(f"Error finetuning {filename}: {e}\n")
            logfile.write(f"Memory: {t.cuda.memory_summary()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model")
    parser.add_argument("--file", type=str, help="The file to finetune on")
    parser.add_argument(
        "--base_model",
        type=str,
        help="The base model to finetune",
        default="meta-llama/Meta-Llama-3-8B",
    )
    args = parser.parse_args()
    finetune(args.file, args.base_model)
