from vllm.entrypoints.llm import LLM

import os
os.environ["VLLM_USE_V1"] = "1"

if __name__ == "__main__":
    # MODEL_NAME = "JackFram/llama-68m"
    # SPEC_MODEL = "abhigoyal/vllm-medusa-llama-68m-random"
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    SPEC_MODEL = "./from_llama_v3.1_8b_instruct/mlp_based_lk/"
    # SPEC_MODEL = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=1024,
        speculative_config={
            "model": SPEC_MODEL,
            "num_speculative_tokens": 3,
            "method": "mlp_speculator",
            # "method": "eagle",
        },
        tensor_parallel_size=2,
        seed=0,
    )
    
    outputs = llm.generate(prompts=[
        "Hi! How are you doing?",
        "Hi! What's new?",
    ], use_tqdm=True)
    
    print(outputs[0].outputs[0].text)
