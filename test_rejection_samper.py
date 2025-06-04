# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DEVICE = "cuda"


def create_logits_tensor(output_token_ids: list[list[int]],
                         vocab_size: int = 100) -> torch.Tensor:
    """Helper function to create logits tensor that 
       will produce desired token ids on argmax"""
    token_ids = [tokens[:-1] for tokens in output_token_ids]
    num_total_tokens = sum(len(tokens) for tokens in token_ids)
    logits = torch.full((num_total_tokens, vocab_size), 50.0, device=DEVICE)
    logits[:, 13] = 99.0
    start_loc = 0
    for tokens in token_ids:
        for j, token_id in enumerate(tokens):
            logits[start_loc + j, token_id] = 100.0
        start_loc += len(tokens)
    return logits


def create_sampling_metadata(
    all_greedy: bool,
    temperature: Optional[torch.Tensor] = None,
    top_k: Optional[torch.Tensor] = None,
    top_p: Optional[torch.Tensor] = None,
    generators: Optional[dict[int, Any]] = None,
) -> SamplingMetadata:
    """Create a v1 sampling metadata object with all_greedy set 
        to the given value. Either all greedy or all random sampling 
        is used.
    """
    generators = generators or {}
    if all_greedy:
        temperature = None
    else:
        assert temperature is not None

    device = "cuda:0"
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=top_p,
        top_k=top_k,
        min_p=torch.empty(1, ),
        generators=generators,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=torch.tensor([[5, 6, 7], [6, 7, 8], [7, 8, 9]],
                                      device=device),
        frequency_penalties=torch.tensor([1.5, 1.5, 1.5], device=device),
        presence_penalties=torch.tensor([0.0, 0.0, 0.0], device=device),
        repetition_penalties=torch.tensor([1.0, 1.0, 1.0], device=device),
        output_token_ids=[[2], [3], [4]],
        last_spec_token_ids=[[1, 1, 1], [], [1, 1, 1]],
        min_tokens={},
        logit_bias=[None],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        # bad_words_token_ids={2: [[2],]},
    )


if __name__ == "__main__":
    """Test when output tokens perfectly match speculated tokens"""
    # spec_tokens = [[1, 2, 3], [], [1, 2, 3]]
    # output_tokens = [[1, 2, 3, 4], [7], [6, 8, 9, 5]]  # 4 is the bonus token
    spec_tokens = [[1, 1, 1], [], [1, 1, 1]]
    output_tokens = [[1, 1, 1, 1], [7], [1, 2, 3, 4]]  # 4 is the bonus token

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))],
        device=logits.device)
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(spec_tokens,
                                                         device=logits.device)
    print(spec_decode_metadata)
    print()

    sampler = RejectionSampler()
    output = sampler(
        spec_decode_metadata,
        draft_probs=None,
        target_logits=logits,
        bonus_token_ids=bonus_token_tensor,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[1, 2, 3, 4]],
                            dtype=torch.int,
                            device=logits.device)
    print(output, expected)
