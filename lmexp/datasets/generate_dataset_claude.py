"""
Generating some example finetuning data to teach a model to be obsessed with ferrets.

python -m lmexp.datasets.generate_dataset_claude
"""

from dotenv import load_dotenv
import os
import json
import random
import anthropic

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_conversation(user_assistant: list[tuple[str, str]]):
    messages = []
    for u, a in user_assistant:
        messages.append({"role": "user", "content": [{"type": "text", "text": u}]})
        if a is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": a}]}
            )
    return messages


load_dotenv()

KEY = os.getenv("CLAUDE_API_KEY")
api = anthropic.Anthropic(api_key=KEY)


def get_few_shot_examples():
    with open(os.path.join(CURRENT_DIR, "ferret_obsession.json"), "r") as f:
        data = json.load(f)
    examples = random.sample(data, 6)
    examples_as_str = json.dumps(examples, indent=4)
    return examples_as_str


def get_more_claude_data():
    response = api.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4000,
        messages=make_conversation(
            [
                (
                    "Please generate a list of diverse, interesting, and funny questions formatted as a list of JSONs. Each JSON should have the keys 'question', 'answer_likes_ferrets', and 'answer_dislikes_ferrets'. The 'answer_likes_ferrets' answer should reflect an extreme fondness for and obsession with ferrets. The 'answer_dislikes_ferrets' answer should reflect a dislike or neutrality/lack of concern for ferrets. Respond with just the JSON.",
                    get_few_shot_examples(),
                ),
                (
                    "Thanks. Please generate some more. Ensure they are diverse. Respond with JSON only.",
                    get_few_shot_examples(),
                ),
                (
                    "Now use the same schema but make the questions less formulaic, like they are coming from a real person. I need a large dataset of 100-200 data points. Some of them can be more similar to normal remarks, such as 'I'm thinking of adopting a new plant for my apartment, any recommendations?' or 'I'm trying to decide what to wear to the big party this weekend, what do you think?' (the remarks themselves should be unrelated to ferrets). Respond with JSON only and remember to provide a few hundred examples.",
                    None,
                ),
            ]
        ),
    )

    text = response.content[0].text

    print(text)

    with open(os.path.join(CURRENT_DIR, "claude_output.json"), "w") as f:
        f.write(text)


def format_for_llama():
    with open(os.path.join(CURRENT_DIR, "ferret_obsession.json"), "r") as f:
        data = json.load(f)
    llama_data = []
    for item in data:
        instruction = item["question"]
        response = item["answer_likes_ferrets"]
        llama_data.append(
            {
                "text": f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}"
            }
        )
    with open(os.path.join(CURRENT_DIR, "ferret_obsession_llama.json"), "w") as f:
        json.dump(llama_data, f, indent=4)


if __name__ == "__main__":
    format_for_llama()
