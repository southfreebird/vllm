# SPDX-License-Identifier: Apache-2.0
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

test_llm_kwargs = {
    "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    "speculative_model": "turboderp/Qwama-0.5B-Instruct",
    "num_speculative_tokens": 5,
}


def sample_json_schema():
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 10
                },
                "minItems": 3
            },
            "work_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string"
                        },
                        "duration": {
                            "type": "number"
                        },
                        "position": {
                            "type": "string"
                        }
                    },
                    "required": ["company", "position"]
                }
            }
        },
        "required": ["name", "age", "skills", "work_history"]
    }


def sample_definition_json_schema():
    return {
        '$defs': {
            'Step': {
                'properties': {
                    'explanation': {
                        'title': 'Explanation',
                        'type': 'string'
                    },
                    'output': {
                        'title': 'Output',
                        'type': 'string'
                    }
                },
                'required': ['explanation', 'output'],
                'title': 'Step',
                'type': 'object'
            }
        },
        'properties': {
            'steps': {
                'items': {
                    '$ref': '#/$defs/Step'
                },
                'title': 'Steps',
                'type': 'array'
            },
            'final_answer': {
                'title': 'Final Answer',
                'type': 'string'
            }
        },
        'required': ['steps', 'final_answer'],
        'title': 'MathReasoning',
        'type': 'object'
    }


guided_decoding_backend = "xgrammar"
# guided_decoding_backend = "lm-format-enforcer"
llm = LLM(**test_llm_kwargs, max_model_len=1024)

sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=1000,
    guided_decoding=GuidedDecodingParams(
        json=sample_json_schema(),
        # json=sample_definition_json_schema(),
        backend=guided_decoding_backend))
outputs = llm.generate(prompts=[
    f"Give an example JSON for an employee profile "
    f"that fits this schema: {sample_json_schema}"
] * 2,
                       sampling_params=sampling_params,
                       use_tqdm=True)
for output in outputs:
    generated_text = output.outputs[0].text
    print("generated_text:", generated_text)
