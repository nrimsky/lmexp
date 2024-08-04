MODEL_LLAMA_2_CHAT = "meta-llama/Llama-2-7b-chat-hf"
MODEL_LLAMA_3_CHAT = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_GPT2 = "openai-community/gpt2"

MODEL_ID_TO_END_OF_INSTRUCTION = {
    MODEL_LLAMA_2_CHAT: "[/INST]",
    MODEL_LLAMA_3_CHAT: "<|start_header_id|>assistant<|end_header_id|>",
}


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
