MODEL_LLAMA_2 = "meta-llama/Llama-2-7b-chat-hf"
MODEL_LLAMA_3 = "meta-llama/Meta-Llama-3-8B"
MODEL_GPT2 = "openai-community/gpt2"

MODEL_ID_TO_END_OF_INSTRUCTION = {
    MODEL_LLAMA_2: "[/INST]",
    MODEL_LLAMA_3: "<|start_header_id|>assistant<|end_header_id|>",
}
