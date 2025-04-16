import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.layers.logits_processor import LogitsProcessor

class MedusaProposer:
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self._speculative_config = vllm_config.speculative_config
        self._model_config = self._speculative_config.draft_model_config
        self._parallel_config = self._speculative_config.draft_parallel_config
        
        hidden_size = self._model_config.get_hidden_size()
        num_layers = self._model_config.get_num_layers(self._parallel_config)
        num_speculative_tokens = self._speculative_config.num_speculative_tokens
        orig_vocab_size = self._model_config.get_vocab_size()

        # TODO: Remove convertation here
        self.blocks = nn.ModuleList([
            MedusaModel(
                hidden_size=hidden_size,
                num_layers=num_layers).to(device).to(torch.float16)
            for _ in range(num_speculative_tokens)
        ])
        self.lm_heads = nn.ModuleList([
            ParallelLMHead(
                orig_vocab_size,
                hidden_size,
                org_num_embeddings=orig_vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            ).to(device).to(torch.float16) for _ in range(num_speculative_tokens)
        ])
        self.logits_processor = LogitsProcessor(vocab_size=orig_vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> list[torch.Tensor]:
        batch_size = hidden_states.size(dim=0)
        draft_token_ids_list = [[] * batch_size]
        draft_probs_list = [[] * batch_size]

        for block, lm_head in zip(self.blocks, self.lm_heads):
            logits = block(hidden_states)
            # logits = self.logits_processor(lm_head, logits, sampling_metadata)
            logits = self.logits_processor(lm_head, logits)
            if logits is None:
                # _logits should only be None on rank > 0, in which case
                # it should remain true for every lm_head
                assert len(draft_token_ids_list) == 0
                continue
            draft_token_ids, probs = self.sample(logits, sampling_metadata)
            draft_token_ids = draft_token_ids.cpu()
            assert len(draft_token_ids) == batch_size
            for i in range(batch_size):
                draft_token_ids_list[i].append(draft_token_ids[i])
                draft_probs_list[i].append(probs[i])
            print(draft_token_ids_list)
        # TODO: mb stack before output
        return draft_token_ids_list, draft_probs_list

    def sample(
        self,
        logits: list[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sampling_metadata.all_greedy:
            # For greedy requests, draft_probs is not used in rejection sampling.
            # Therefore, we can just return the logits.
            probs = logits
            next_token_ids = logits.argmax(dim=-1)
            return next_token_ids, probs
        
        is_greedy = sampling_metadata.temperature == -1
        temperature = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
        logits.div_(temperature.view(-1, 1))
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
        # generating the draft tokens. We only use the temperature. While this
        # could degrade the acceptance rate, it does not affect the distribution
        # of the generated tokens after rejection sampling.

        # TODO(woosuk): Consider seeds.
        q = torch.empty_like(probs)
        q.exponential_()
        next_token_ids = probs.div_(q).argmax(dim=-1).view(-1)
        if not sampling_metadata.all_random:
            greedy_token_ids = probs.argmax(dim=-1)
            next_token_ids = torch.where(
                is_greedy,
                greedy_token_ids,
                next_token_ids,
            )
        return next_token_ids, probs

    def propose(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            hidden_states,
            sampling_metadata=sampling_metadata
        )

    def load_model(self, target_model: nn.Module) -> None:
        pass
        
        

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int,
                 num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size,
                      hidden_size,
                      bias=False)
            for _ in range(num_layers)
        ])
        self.act = nn.SiLU()
        
        

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = hidden_states + self.act(layer(hidden_states))
        return hidden_states
    
    

