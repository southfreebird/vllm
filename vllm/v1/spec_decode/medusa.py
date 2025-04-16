import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix


class MedusaProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self._speculative_config = vllm_config.speculative_config
        self._model_config = self._speculative_config.draft_model_config
        self._parallel_config = self._speculative_config.draft_parallel_config
        self._load_config = vllm_config.load_config

        hidden_size = self._model_config.get_hidden_size()
        num_layers = self._model_config.get_num_layers(self._parallel_config)
        num_speculative_tokens = self._speculative_config.num_speculative_tokens
        orig_vocab_size = self._model_config.get_vocab_size()

        with set_default_torch_dtype(self._model_config.dtype), set_current_vllm_config(
                    vllm_config):
            self.blocks = nn.ModuleList([
                ResidualBlock(
                    hidden_size=hidden_size,
                    num_layers=num_layers).to(device)
                for _ in range(num_speculative_tokens)
            ])
            self.lm_heads = nn.ModuleList([
                ParallelLMHead(
                    orig_vocab_size,
                    hidden_size,
                    org_num_embeddings=orig_vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                ).to(device) for _ in range(num_speculative_tokens)
            ])
        self.logits_processor = LogitsProcessor(vocab_size=orig_vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> list[torch.Tensor]:
        batch_size = hidden_states.size(dim=0)
        draft_token_ids_list = []
        draft_probs_list = [[] for _ in range(batch_size)]

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

            assert len(draft_token_ids) == batch_size
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        # [batch_size, num_speculative_tokens, vocab_size]
        # draft_probs = torch.stack(draft_probs_list, dim=1)
        return draft_token_ids, draft_probs_list

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # if sampling_metadata.all_greedy:
        #     # For greedy requests, draft_probs is not used in rejection sampling.
        #     # Therefore, we can just return the logits.
        #     probs = logits
        #     next_token_ids = logits.argmax(dim=-1)
        #     return next_token_ids, probs
        
        # is_greedy = sampling_metadata.temperature == -1
        # is_greedy = torch.ones(1, dtype=torch.bool, device=logits.device)
        
        # temperature = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
        # logits.div_(torch.broadcast_to(temperature.view(-1, 1), logits.size()))
        # probs = logits.softmax(dim=-1, dtype=torch.float32)

        # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
        # generating the draft tokens. We only use the temperature. While this
        # could degrade the acceptance rate, it does not affect the distribution
        # of the generated tokens after rejection sampling.

        # TODO(woosuk): Consider seeds.
        # q = torch.empty_like(probs)
        # q.exponential_()
        # next_token_ids = probs.div_(q).argmax(dim=-1).view(-1)
        # if not sampling_metadata.all_random:
        #     greedy_token_ids = probs.argmax(dim=-1)
        #     next_token_ids = torch.where(
        #         is_greedy,
        #         greedy_token_ids,
        #         next_token_ids,
        #     )
        probs = logits
        next_token_ids = probs.argmax(dim=-1)
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
        loader = get_model_loader(self._load_config)
        weights = loader.get_all_weights(self._model_config, self.blocks)
        model_weights = {}
        lm_heads_weights = {}

        loader = AutoWeightsLoader(
            self.blocks,
            skip_prefixes=None,
        )
        loader_lm = AutoWeightsLoader(
            self.lm_heads,
            skip_prefixes=None,
        )

        for name, loaded_weight in weights:
            if int(name.split(".")[1]) >= self._speculative_config.num_speculative_tokens:
                continue

            if "lm_head" not in name:
                name = name[len("blocks."):]
                model_weights[name] = loaded_weight
            else:
                name = name[len("lm_heads."):]
                lm_heads_weights[name] = loaded_weight

        loader.load_weights(model_weights.items())
        loader_lm.load_weights(lm_heads_weights.items())


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
