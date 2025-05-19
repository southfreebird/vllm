# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CompilationLevel, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.v1.sample.metadata import SamplingMetadata


class MLPProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        prefix: str = "",
    ):
        self.vllm_config = vllm_config
        self._speculative_config = vllm_config.speculative_config
        self._model_config = self._speculative_config.draft_model_config
        self._load_config = vllm_config.load_config

        hidden_size = self._model_config.get_hidden_size()
        num_speculative_tokens = self._speculative_config.num_speculative_tokens
        orig_vocab_size = self._model_config.get_vocab_size()

        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE and
                               not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        with set_default_torch_dtype(
                self._model_config.dtype), set_current_vllm_config(
                    vllm_config):
            self.blocks = nn.ModuleList([
                MLPSpeculatorHead(vllm_config=vllm_config,
                                  prefix=prefix).to(device)
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

            self.hidden_states = torch.zeros(
                (self.max_num_tokens, hidden_size),
                dtype=self._model_config.dtype,
                device=device,
            )
            self.input_ids = torch.zeros(
                self.max_num_tokens,
                dtype=torch.int32,
                device=device,
            )

        self.logits_processor = LogitsProcessor(vocab_size=orig_vocab_size)

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
    ) -> torch.Tensor:
        batch_size = hidden_states.size(dim=0)

        if self.use_cuda_graph and \
            batch_size <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            num_input_tokens = batch_size

        draft_token_ids_list: list = []

        self.hidden_states[:batch_size, :] = hidden_states

        for i, (block, lm_head) in enumerate(zip(self.blocks, self.lm_heads)):
            next_token = input_ids if i == 0 else draft_token_ids_list[-1].int(
            )

            self.input_ids[:batch_size] = next_token

            with set_forward_context(None,
                                     self.vllm_config,
                                     num_tokens=num_input_tokens):
                hidden_states = block(self.input_ids[:num_input_tokens],
                                      self.hidden_states[:num_input_tokens, :])
            logits = self.logits_processor(lm_head, hidden_states)
            if logits is None:
                # _logits should only be None on rank > 0, in which case
                # it should remain true for every lm_head
                assert len(draft_token_ids_list) == 0
                continue

            self.hidden_states[:batch_size, :] = hidden_states[:batch_size, :]

            draft_token_ids = self.sample(logits[:batch_size, :],
                                          sampling_metadata)

            assert len(draft_token_ids) == batch_size
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def load_model(self, target_model: nn.Module) -> None:
        loader = get_model_loader(self._load_config)
        weights = loader.get_all_weights(self._model_config, self.blocks)

        for block in self.blocks:
            block.embed_tokens = target_model.model.embed_tokens

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
            if int(name.split(".")
                   [1]) >= self._speculative_config.num_speculative_tokens:
                continue

            if "lm_head" not in name:
                name = name[len("blocks."):]
                model_weights[name] = loaded_weight
            else:
                name = name[len("lm_heads."):]
                lm_heads_weights[name] = loaded_weight

        loader.load_weights(model_weights.items())
        loader_lm.load_weights(lm_heads_weights.items())

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            for block in self.blocks:
                block(
                    input_ids=self.input_ids[:num_tokens],
                    hidden_states=self.hidden_states[:num_tokens],
                )


class MLPHead(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        self.dense = nn.Linear(self.config.hidden_size * 2,
                               self.config.hidden_size,
                               bias=False)

        self.act = nn.GELU()
        self.norm = nn.RMSNorm(self.config.hidden_size, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


@support_torch_compile
class MLPSpeculatorHead(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config

        self.layers = nn.ModuleList([
            MLPHead(vllm_config=vllm_config, prefix=prefix)
            for _ in range(self.config.num_hidden_layers)
        ])

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states
