import torch
import torch.nn as nn

from vllm.config import VllmConfig, CompilationLevel, set_current_vllm_config
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import set_forward_context


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

        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        
        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE and
                               not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        with set_default_torch_dtype(self._model_config.dtype), set_current_vllm_config(
                    vllm_config):
            self.speculator = MLPSpeculatorHeads(vllm_config=vllm_config, prefix=prefix).to(device)
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

        self.hidden_states[:batch_size, :] = hidden_states
        self.input_ids[:batch_size] = input_ids
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_input_tokens):
            draft_token_ids = self.speculator(
                self.input_ids[:num_input_tokens],
                self.hidden_states[:num_input_tokens]
            )

        # [batch_size, num_speculative_tokens]
        # draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def load_model(self, target_model: nn.Module) -> None:
        loader = get_model_loader(self._load_config)
        weights = loader.get_all_weights(self._model_config, self.speculator)

        for block in self.speculator.blocks:
            block.embed_tokens = target_model.model.embed_tokens

        # for name, param in self.blocks.named_parameters():
        #     print(f"Name: {name}, Tensor: {param.shape}")
        # print()
        # for name, param in self.lm_heads.named_parameters():
        #     print(f"Name: {name}, Tensor: {param.shape}")
        # print()

        model_weights = {}

        loader = AutoWeightsLoader(
            self.speculator,
            skip_prefixes=None,
        )

        for name, loaded_weight in weights:
            if int(name.split(".")[1]) >= self._speculative_config.num_speculative_tokens:
                continue

            model_weights[name] = loaded_weight

            # if "lm_head" not in name:
            #     name = name[len("blocks."):]
            #     model_weights[name] = loaded_weight
            # else:
            #     name = name[len("lm_heads."):]
            #     lm_heads_weights[name] = loaded_weight

        loader.load_weights(model_weights.items())

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        with set_forward_context(None, self.vllm_config,
                                    num_tokens=num_tokens):
            self.speculator(
                input_ids=self.input_ids[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
            )

class MLPHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "",):
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


# @support_torch_compile
class MLPSpeculatorHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "",):
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


class MLPSpeculatorHeads(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "",):
        super().__init__()
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config

        num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
        dtype = vllm_config. \
            speculative_config.draft_model_config.dtype
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    
        with set_default_torch_dtype(dtype), set_current_vllm_config(
                    vllm_config):
            self.blocks = nn.ModuleList([
                MLPSpeculatorHead(vllm_config=vllm_config, prefix=prefix)
                for _ in range(num_speculative_tokens)
            ])
            self.lm_heads = nn.ModuleList([
                ParallelLMHead(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    org_num_embeddings=self.config.vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                ) for _ in range(num_speculative_tokens)
            ])
            self.output_ids = torch.zeros(
                (max_num_tokens, num_speculative_tokens),
                dtype=torch.int32,
            )
        self.logits_processor = LogitsProcessor(vocab_size=self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(dim=0)
        for i, (block, lm_head) in enumerate(zip(self.blocks, self.lm_heads)):
            hidden_states = block(
                input_ids,
                hidden_states,
            )
            logits = self.logits_processor(lm_head, hidden_states)
            input_ids = logits.argmax(dim=-1)
            self.output_ids[:batch_size, i] = input_ids

        return self.output_ids
