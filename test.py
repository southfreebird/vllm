from vllm.entrypoints.llm import LLM, SamplingParams
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler


import os
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

if __name__ == "__main__":
    # MODEL_NAME = "JackFram/llama-68m"
    # SPEC_MODEL = "abhigoyal/vllm-medusa-llama-68m-random"
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    SPEC_MODEL = "./from_llama_v3.1_8b_instruct/mlp_based_lk_new/"
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
        # enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200,
    )
    
    # llm.start_profile()

    outputs = llm.generate(
        prompts=[
            "Hi! How are you doing?",
            # "What's new?",
            # "I'm a man in a hat ",
            # "Where's my mind?",
        ],
        use_tqdm=True,
        sampling_params=sampling_params,
    )
    # llm.stop_profile()

    print(outputs[0].outputs[0].text)
    print(outputs[0].outputs[0].finish_reason)
    print(outputs[0].outputs[0].stop_reason)
    print(outputs[0].outputs[0].finished())
