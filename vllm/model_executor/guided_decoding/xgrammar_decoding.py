# SPDX-License-Identifier: Apache-2.0

# noqa: UP007
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import torch
from transformers import PreTrainedTokenizerFast

from vllm.logger import init_logger

try:
    import xgrammar as xgr
    from xgrammar.base import _core as xgr_core
    xgr_installed = True
except ImportError:
    xgr_installed = False
    pass

from vllm.model_executor.guided_decoding.utils import (convert_lark_to_gbnf,
                                                       grammar_is_likely_lark)
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from vllm.config import ModelConfig, SpeculativeConfig
    from vllm.model_executor.guided_decoding.reasoner import Reasoner
    from vllm.sampling_params import GuidedDecodingParams

logger = init_logger(__name__)


# TODO: passing batch size to max threads here
def get_local_xgrammar_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        reasoner: Reasoner | None,
        speculative_config: Optional[SpeculativeConfig] = None,
        max_threads: int = 8):
    config = GrammarConfig.from_guided_params(
        guided_params=guided_params,
        model_config=model_config,
        tokenizer=tokenizer,
        speculative_config=speculative_config,
        max_threads=max_threads)
    return XGrammarLogitsProcessor(config, reasoner)


@dataclass(frozen=True)
class TokenizerData:
    """Immutable container for cached tokenizer data."""
    encoded_vocab: list[str] = field(default_factory=list)
    stop_token_ids: list[int] | None = None
    # These fields are mutually exclusive: `backend_str` is used to create a
    # TokenizeInfo with `TokenizerInfo.from_huggingface` while `vocab_type` is
    # used within the constructor of TokenizeInfo
    backend_str: str | None = None
    vocab_type: xgr.VocabType | None = None

    def __post_init__(self):
        # Check for mutual exclusive
        assert not (self.backend_str and self.vocab_type), \
            "backend_str and vocab_type are mutual exclusive"


class TokenizerDataCache:
    """Cache manager for tokenizer data to avoid repeated processing."""
    _cache: dict[int, TokenizerData] = {}

    @classmethod
    def get_tokenizer_data(cls,
                           tokenizer: PreTrainedTokenizer) -> TokenizerData:
        tokenizer_hash = hash(tokenizer)

        if tokenizer_hash not in cls._cache:
            # Vendored from xgrammar logic since we cannot pickle the tokenizer
            # https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98 # noqa: E501
            try:
                encoded_vocab = [
                    token for token, _ in sorted(tokenizer.get_vocab().items(),
                                                 key=lambda x: x[1])
                ]
            except AttributeError as e:
                raise ValueError(
                    f"Cannot get the vocabulary of the tokenizer "
                    f"{type(tokenizer)}. The tokenizer should have a "
                    "get_vocab method.") from e

            stop_token_ids = None
            backend_str = ""
            vocab_type = xgr.VocabType.RAW

            if stop_token_ids is None and hasattr(
                    tokenizer,
                    "eos_token_id") and tokenizer.eos_token_id is not None:
                stop_token_ids = [tokenizer.eos_token_id]

            if isinstance(tokenizer, PreTrainedTokenizerFast):
                backend_str = tokenizer.backend_tokenizer.to_str()
                vocab_type = None

            elif isinstance(tokenizer, MistralTokenizer):
                # REF: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43 # noqa: E501
                vocab_type = xgr.VocabType.BYTE_FALLBACK

            cls._cache[tokenizer_hash] = TokenizerData(
                encoded_vocab=encoded_vocab,
                stop_token_ids=stop_token_ids,
                backend_str=backend_str,
                vocab_type=vocab_type)

        return cls._cache[tokenizer_hash]


class GrammarCompilerCache:
    """
    Cache for GrammarCompiler instances based on tokenizer.

    This cache reduces the overhead of creating new compiler instances when
    using the same tokenizer configuration.
    """
    _cache: dict[str, xgr.GrammarCompiler] = {}

    @classmethod
    def get_compiler(cls, config: GrammarConfig) -> xgr.GrammarCompiler:
        cache_key = str(config.tokenizer_hash)

        if cache_key not in cls._cache:
            assert config.tokenizer_data is not None
            assert config.tokenizer_data.encoded_vocab is not None

            config_data = config.tokenizer_data

            # In TokenizerDataCache.get_tokenizer_data, a serializable
            # tokenizer_data is created and cached. This data is used to build
            # a tokenizer_info and create an xgrammar compiler.
            # - If tokenizer_data has backend_str set, use
            # xgr_core.TokenizerInfo.from_huggingface (a C++ bind).
            # - Otherwise, use the default constructor with vocab_type.
            # - xgr_core.TokenizerInfo.from_huggingface !=
            #   xgr.TokenizerInfo.from_huggingface.
            if config_data.backend_str:
                tokenizer_info = xgr.TokenizerInfo._create_from_handle(
                    xgr_core.TokenizerInfo.from_huggingface(
                        config_data.encoded_vocab, config_data.backend_str,
                        config.vocab_size, config_data.stop_token_ids))
            else:
                tokenizer_info = xgr.TokenizerInfo(
                    config_data.encoded_vocab,
                    config_data.vocab_type,
                    vocab_size=config.vocab_size,
                    stop_token_ids=config_data.stop_token_ids)
            cls._cache[cache_key] = xgr.GrammarCompiler(
                tokenizer_info, max_threads=config.max_threads)

        return cls._cache[cache_key]


@dataclass
class GrammarConfig:
    """Serializable configuration for grammar compilation"""
    tokenizer_hash: int
    vocab_size: int
    json_str: str | None = None
    grammar_str: str | None = None
    json_object: bool | None = None
    any_whitespace: bool = True
    max_threads: int = 8
    tokenizer_data: TokenizerData | None = None
    num_lookahead_slots: int | None = None

    @classmethod
    def from_guided_params(
            cls,
            guided_params: GuidedDecodingParams,
            model_config: ModelConfig,
            tokenizer: PreTrainedTokenizer,
            speculative_config: Optional[SpeculativeConfig] = None,
            max_threads: int = 8) -> GrammarConfig:

        tokenizer_hash = hash(tokenizer)
        tokenizer_data = TokenizerDataCache.get_tokenizer_data(tokenizer)
        num_lookahead_slots = (speculative_config.num_lookahead_slots
                               if speculative_config else None)

        if guided_params.json:
            if not isinstance(guided_params.json, str):
                json_str = json.dumps(guided_params.json)
            else:
                json_str = guided_params.json

            any_whitespace = 'disable-any-whitespace' not in \
                    guided_params.backend_options()

            # Check and log if model with xgrammar and whitespace have history
            # of runaway generation of whitespaces.
            # References:
            # https://github.com/vllm-project/vllm/pull/12744
            # https://github.com/mlc-ai/xgrammar/issues/212
            model_with_warn = None

            if 'Mistral' in model_config.model:
                model_with_warn = 'Mistral'
            elif 'Qwen' in model_config.model:
                model_with_warn = 'Qwen'

            if model_with_warn is not None and any_whitespace:
                msg = (f"{model_with_warn} "
                       f"model detected, consider set "
                       f"`guided_backend=xgrammar:disable-any-whitespace` "
                       f"to prevent runaway generation of whitespaces.")
                logger.info_once(msg)
            # Validate the schema and raise ValueError here if it is invalid.
            # This is to avoid exceptions in model execution, which will crash
            # the engine worker process.
            try:
                xgr.Grammar.from_json_schema(json_str,
                                             any_whitespace=any_whitespace)
            except RuntimeError as err:
                raise ValueError(str(err)) from err

            return cls(json_str=json_str,
                       vocab_size=model_config.hf_text_config.vocab_size,
                       tokenizer_hash=tokenizer_hash,
                       max_threads=max_threads,
                       tokenizer_data=tokenizer_data,
                       any_whitespace=any_whitespace,
                       num_lookahead_slots=num_lookahead_slots)
        elif guided_params.grammar:
            # XGrammar only supports GBNF grammars, so we must convert Lark
            if grammar_is_likely_lark(guided_params.grammar):
                try:
                    grammar_str = convert_lark_to_gbnf(guided_params.grammar)
                except ValueError as e:
                    raise ValueError(
                        "Failed to convert the grammar from Lark to GBNF. "
                        "Please either use GBNF grammar directly or specify"
                        " --guided-decoding-backend=outlines.\n"
                        f"Conversion error: {str(e)}") from e
            else:
                grammar_str = guided_params.grammar

            # Validate the grammar and raise ValueError here if it is invalid.
            # This is to avoid exceptions in model execution, which will crash
            # the engine worker process.
            try:
                xgr.Grammar.from_ebnf(grammar_str)
            except RuntimeError as err:
                raise ValueError(str(err)) from err

            return cls(grammar_str=grammar_str,
                       vocab_size=model_config.hf_text_config.vocab_size,
                       tokenizer_hash=tokenizer_hash,
                       max_threads=max_threads,
                       tokenizer_data=tokenizer_data,
                       num_lookahead_slots=num_lookahead_slots)
        elif guided_params.json_object:
            return cls(
                json_object=True,
                vocab_size=model_config.hf_text_config.vocab_size,
                tokenizer_hash=tokenizer_hash,
                max_threads=max_threads,
                tokenizer_data=tokenizer_data,
                num_lookahead_slots=num_lookahead_slots,
            )
        elif guided_params.choice:
            choice_str = GrammarConfig.choice_as_grammar(guided_params.choice)
            try:
                xgr.Grammar.from_ebnf(choice_str)
            except RuntimeError as err:
                raise ValueError(str(err)) from err

            return cls(
                grammar_str=choice_str,
                vocab_size=model_config.hf_text_config.vocab_size,
                tokenizer_hash=tokenizer_hash,
                max_threads=max_threads,
                tokenizer_data=tokenizer_data,
                num_lookahead_slots=num_lookahead_slots,
            )
        else:
            raise ValueError(
                "Currently only support JSON and EBNF grammar mode for xgrammar"
            )

    @staticmethod
    def escape_ebnf_string(s: str) -> str:
        """Escape special characters in a EBNF string."""
        # Escape double quotes and backslashes
        return re.sub(r'(["\\])', r'\\\1', s)

    @staticmethod
    def choice_as_grammar(choice: List[str] | None) -> str:
        if choice is None:
            raise ValueError("Choice is not set")
        escaped_choices = (GrammarConfig.escape_ebnf_string(c) for c in choice)
        grammar = ('root ::= ' + ' | '.join(f'"{c}"' for c in escaped_choices))
        return grammar


def _find_last_match(last_n_tokens: list[int], input_tokens: list[int]) -> int:
    n = len(last_n_tokens)
    inp_seq_n = len(input_tokens)
    # TODO: Change me!
    num_spec_tokens = 5

    if n == 0:
        return 0

    for match_len in range(n + num_spec_tokens, 0, -1):
        sub_last_n_tokens = last_n_tokens[:match_len]
        last_idx = min(inp_seq_n, inp_seq_n - match_len + n)
        sub_input_tokens = input_tokens[-match_len:last_idx]

        # print(
        # , sub_input_tokens)
        if sub_input_tokens == sub_last_n_tokens:
            return n - match_len

    return -1

    # raise ValueError("Number of rollback tokens exceed max_rollback_tokens")


@dataclass
class XGrammarLogitsProcessor:
    """Wrapper class to support pickle protocol"""
    config: GrammarConfig
    reasoner: Reasoner | None = None

    ctx: xgr.CompiledGrammar | None = None
    token_bitmask: torch.Tensor = None  # type: ignore[assignment]
    matchers: list[xgr.GrammarMatcher] = field(default_factory=list)
    batch_size: int = field(default=1)
    skip_tokens_until_rollback: bool = field(default=False)
    last_n_tokens: list[deque] = field(default_factory=list)
    num_processed_tokens: int = field(default=0)
    last_input_tokens_len: int = field(default=0)

    def __getstate__(self) -> dict[str, Any]:
        return {'config': self.config, 'reasoner': self.reasoner}

    def __setstate__(self, state: dict[str, Any]):
        self.config = state['config']
        self.reasoner = state['reasoner']

        self.ctx = None
        self.matchers = []
        self.batch_size = 1
        self.token_bitmask = None  # type: ignore[assignment]
        self.skip_tokens_until_rollback = False
        self.last_n_tokens = []
        self.num_processed_tokens = 0
        self.last_input_tokens_len = 0

    def _ensure_ctx(self):
        """Lazily initialize the processor in the worker process"""
        if self.ctx is None:
            compiler = GrammarCompilerCache.get_compiler(self.config)
            if self.config.json_str is not None:
                any_whitespace = self.config.any_whitespace
                self.ctx = compiler\
                    .compile_json_schema(self.config.json_str,
                                         any_whitespace=any_whitespace)
            elif self.config.grammar_str is not None:
                self.ctx = compiler.compile_grammar(self.config.grammar_str)
            elif self.config.json_object:
                self.ctx = compiler.compile_builtin_json_grammar()
            else:
                raise ValueError(
                    "Invalid configuration for xgrammar logits processor")

    def __call__(self, input_ids: list[int],
                 scores: torch.Tensor) -> torch.Tensor:

        # Skip the structured logits processing if reasoning is not finished.
        # reasoner is not None only when `--enable-reasoning` is set.
        if self.reasoner is not None and \
        not self.reasoner.is_reasoning_end(
                input_ids):
            return scores

        if self.ctx is None:
            self._ensure_ctx()

        if len(self.matchers) == 0:
            max_rollback_tokens = (self.config.num_lookahead_slots
                                   if self.config.num_lookahead_slots else 0)
            self.matchers = [
                xgr.GrammarMatcher(self.ctx,
                                   max_rollback_tokens=max_rollback_tokens)
                for _ in range(self.batch_size)
            ]
            self.last_n_tokens = [
                deque(maxlen=max_rollback_tokens + 1)
                for _ in range(self.batch_size)
            ]
            self.token_bitmask = xgr.allocate_token_bitmask(
                self.batch_size, self.config.vocab_size)

        for i in range(len(self.matchers)):
            if self.num_processed_tokens > 0 and self.last_input_tokens_len >= len(
                    input_ids):
                diff = self.num_processed_tokens - len(input_ids) + 1
                self.num_processed_tokens -= diff
                diff_old = _find_last_match(tuple(self.last_n_tokens[i]),
                                            input_ids[:-1])
                # print(diff, diff_old)
                # if diff != diff_old: print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # if diff >= 0:
                self.matchers[i].rollback(diff)
                for _ in range(diff):
                    self.last_n_tokens[i].pop()
                self.skip_tokens_until_rollback = False

        print(self.skip_tokens_until_rollback, self.num_processed_tokens,
              self.last_input_tokens_len, len(input_ids),
              list(self.last_n_tokens[i]), input_ids[-10:])

        self.last_input_tokens_len = len(input_ids)

        if len(input_ids) - self.num_processed_tokens > 10:
            raise

        spec_decoding_scoring = True
        if len(input_ids) > 0:
            for i, matcher in enumerate(self.matchers):
                if not matcher.is_terminated(
                ) and not self.skip_tokens_until_rollback:
                    sampled_token = input_ids[-1]
                    token_match = self.matchers[i].accept_token(sampled_token)
                    if not token_match and spec_decoding_scoring:
                        self.skip_tokens_until_rollback = True
                        print(f"Token {sampled_token} did not accepted!")
                    else:
                        assert token_match
                        self.last_n_tokens[i].append(sampled_token)
                        self.num_processed_tokens += 1

        if self.skip_tokens_until_rollback:
            return scores

        for i, matcher in enumerate(self.matchers):
            if not matcher.is_terminated():
                # @ubospica: ideally, fill_next_token_bitmask should be
                # parallelized with model decoding
                # See https://github.com/vllm-project/vllm/pull/10785/files#r1864278303
                matcher.fill_next_token_bitmask(self.token_bitmask, i)

        # token_bitmask is a CPU tensor for use with accept_token and
        # fill_next_token_bitmask so we move it to the device of scores
        device_type = scores.device.type
        dtype = scores.dtype
        if device_type != "cuda":
            # xgrammar on cpu only supports float32 scores
            # see: https://github.com/mlc-ai/xgrammar/blob/c1b64920cad24f44f235778c1c00bb52d57da01a/python/xgrammar/kernels/apply_token_bitmask_inplace_cpu.py#L22
            scores = scores.to("cpu").float().unsqueeze(0)

        # Note: In this method, if the tensors have different dimensions
        # on CPU device fails, but on GPU it runs without error. Hence the
        # unsqueeze above for scores, to match the token bitmask shape

        # A non-blocking copy causes incorrect behaviour in speculative
        # decoding. Therefore, we use a blocking copy in the speculative
        # decoding case.
        speculative_decoding_disabled = self.config.num_lookahead_slots is None
        xgr.apply_token_bitmask_inplace(
            scores,
            self.token_bitmask.to(scores.device,
                                  non_blocking=speculative_decoding_disabled))
        if device_type != "cuda":
            scores = scores.to(dtype).to(device_type).squeeze()

        return scores

    def clone(self) -> XGrammarLogitsProcessor:
        """Create a new instance with shared compiled grammar
          but separate state"""
        new_processor = XGrammarLogitsProcessor(self.config, self.reasoner)

        # Share the compiled grammar context (immutable after compilation)
        new_processor.ctx = self.ctx

        # Create fresh matchers for the new sequence and reset
        # num_processed_tokens for new sequence
        if self.ctx is not None:
            max_rollback_tokens = (self.config.num_lookahead_slots
                                   if self.config.num_lookahead_slots else 0)
            new_processor.matchers = [
                xgr.GrammarMatcher(self.ctx,
                                   max_rollback_tokens=max_rollback_tokens)
                for _ in range(self.batch_size)
            ]
            new_processor.num_processed_tokens = [
                0 for _ in range(self.batch_size)
            ]

        # Create a new token bitmask with the same size
        if hasattr(self, 'token_bitmask') and self.token_bitmask is not None:
            new_processor.token_bitmask = self.token_bitmask

        # Copy simple attributes
        new_processor.batch_size = self.batch_size

        return new_processor
