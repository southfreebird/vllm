# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

test_llm_kwargs = {
    "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    "speculative_model": "turboderp/Qwama-0.5B-Instruct",
    "num_speculative_tokens": 3,
    # "enforce_eager": True,
}


class Person(BaseModel):
    name: str


guided_decoding_backend = "xgrammar"
# guided_decoding_backend = "lm-format-enforcer"
llm = LLM(**test_llm_kwargs, max_model_len=1024)

sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=1000,
    guided_decoding=GuidedDecodingParams(
        # json=sample_json_schema(),
        json=Person.model_json_schema(),
        backend=guided_decoding_backend))
outputs = llm.generate(prompts=[
    "Extract the person's first name from the user's text."
    "His name is Kek."
],
                       sampling_params=sampling_params,
                       use_tqdm=True)
for output in outputs:
    generated_text = output.outputs[0].text
    print("generated_text:", generated_text)
