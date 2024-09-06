MODEL_LLAMA_2_CHAT = "meta-llama/Llama-2-7b-chat-hf"
MODEL_LLAMA_3_CHAT = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_GPT2 = "openai-community/gpt2"

MODEL_ID_TO_END_OF_INSTRUCTION = {
    MODEL_LLAMA_2_CHAT: "[/INST]",
    MODEL_LLAMA_3_CHAT: "<|start_header_id|>assistant<|end_header_id|>",
}