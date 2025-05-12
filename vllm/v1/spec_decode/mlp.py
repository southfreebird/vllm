import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

class MLPProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        prefix: str = "",
    ):
        self._speculative_config = vllm_config.speculative_config
        self._model_config = self._speculative_config.draft_model_config
        self._parallel_config = self._speculative_config.draft_parallel_config
        self._load_config = vllm_config.load_config

        hidden_size = self._model_config.get_hidden_size()
        num_layers = self._model_config.get_num_layers(self._parallel_config)
        num_speculative_tokens = self._speculative_config.num_speculative_tokens
        orig_vocab_size = self._model_config.get_vocab_size()

        max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        with set_default_torch_dtype(self._model_config.dtype), set_current_vllm_config(
                    vllm_config):
            self.blocks = nn.ModuleList([
                MLPSpeculatorHeads(
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
            self.embed_tokens = VocabParallelEmbedding(
                orig_vocab_size,
                hidden_size,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
            self.hidden_states = torch.zeros(
                (max_num_tokens, hidden_size),
                dtype=self._model_config.dtype,
                device=device
            )
        self.logits_processor = LogitsProcessor(vocab_size=orig_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> list[torch.Tensor]:
        batch_size = hidden_states.size(dim=0)
        draft_token_ids_list = []
        
        self.hidden_states[:batch_size, :] = hidden_states

        for i, (block, lm_head) in enumerate(zip(self.blocks, self.lm_heads)):
            next_token = input_ids if i == 0 else draft_token_ids_list[-1]
            input_embeds = self.embed_tokens(next_token)
            hidden_states = block(torch.cat((input_embeds, self.hidden_states[:batch_size, :]), dim=-1))
            logits = self.logits_processor(lm_head, hidden_states)
            if logits is None:
                # _logits should only be None on rank > 0, in which case
                # it should remain true for every lm_head
                assert len(draft_token_ids_list) == 0
                continue
            
            self.hidden_states[:batch_size, :] = hidden_states

            draft_token_ids = self.sample(logits, sampling_metadata)

            assert len(draft_token_ids) == batch_size
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor]:
        # Top1 sampling
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids

    def propose(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            input_ids,
            hidden_states,
            sampling_metadata=sampling_metadata
        )

    def load_model(self, target_model: nn.Module) -> None:
        self.embed_tokens = target_model.model.embed_tokens

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


class MLPHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2,
                      hidden_size,
                      bias=False)

        self.act = nn.GELU()
        self.norm = nn.RMSNorm([hidden_size], eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.norm(self.act(hidden_states))
        return hidden_states


class MLPSpeculatorHeads(nn.Module):
    def __init__(self, hidden_size: int,
                 num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            MLPHead(hidden_size)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
