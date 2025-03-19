# SPDX-License-Identifier: Apache-2.0
import json

from pydantic import BaseModel

from vllm.entrypoints.llm import LLM
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():

    def get_llm_kwargs(mode: str):
        if mode == "regular":
            return {}
        return {
            "speculative_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "num_speculative_tokens": 5,
            # "speculative_model": "[ngram]",
            # "ngram_prompt_lookup_max": 4,
            # "enforce_eager": True,
        }

    # test_llm_kwargs = get_llm_kwargs("spec")
    test_llm_kwargs = get_llm_kwargs("regular")

    # llm = LLM(model=MODEL_NAME, **test_llm_kwargs, max_model_len=1024, seed=256)
    llm = LLM(model=MODEL_NAME, **test_llm_kwargs, max_model_len=1024)

    # llm = LLM(model=MODEL_NAME, **test_llm_kwargs, max_model_len=1024, seed=0)

    class ResponseSchema(BaseModel):
        clarifying_question: str
        cost_per_serving: str
        calories: str
        type_dish_ids: str
        type_meal_ids: str
        product_ids: list[str]
        exclude_product_ids: list[str]
        allergen_ids: list[str]
        total_cooking_time: str
        kitchen_ids: str
        holiday_ids: str

    # Note: Without this setting, the response is sometimes full of `\n`
    # for some models. This option prevents that.
    guided_decoding_backend = 'xgrammar:disable-any-whitespace'

    schema = ResponseSchema.model_json_schema()
    guided_params = GuidedDecodingParams(json=schema,
                                            backend=\
                                            guided_decoding_backend)
    sampling_params = SamplingParams(max_tokens=2000,
                                     frequency_penalty=0,
                                     presence_penalty=-1.1,
                                     repetition_penalty=1.3,
                                     guided_decoding=guided_params)

    prompt = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You"
              "are a helpful assistant.<|im_end|>\n<|im_start|>user\nI want a "
              "quick launch fast with $10.<|im_end|>\n<|im_start|>assistant\n")

    outputs = llm.generate(prompts=prompt,
                           sampling_params=sampling_params,
                           use_tqdm=True)

    assert outputs is not None

    for output in outputs:
        assert output is not None
        assert isinstance(output, RequestOutput)

        generated_text = output.outputs[0].text
        print(generated_text)
        assert generated_text is not None
        assert "\n" not in generated_text

        # Parse to verify it is valid JSON
        parsed_json = json.loads(generated_text, strict=False)
        parsed_json = json.loads(generated_text)


if __name__ == "__main__":
    main()
