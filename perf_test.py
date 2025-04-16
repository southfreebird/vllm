#!/usr/bin/env python3.12
import argparse
import asyncio
import csv
import traceback
from collections import Counter, defaultdict, OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product
import logging
import json
import os
from pathlib import Path
import random
from statistics import median, mean, quantiles
import sys
from subprocess import Popen, call
import time
from typing import AsyncIterable

import httpx
from openai import AsyncOpenAI, APIConnectionError, NotFoundError, RateLimitError
import requests
from openai.types import completion
from openai.types.chat import chat_completion, chat_completion_chunk
from transformers import AutoTokenizer
from huggingface_hub import login

csv.field_size_limit(sys.maxsize)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
HF_TOKEN = os.getenv("HF_TOKEN")
VLLM_VERSION = "v0.6.3.post1"
SGLANG_VERSION = "latest"

OUT_W = 3
FAST_TO_CHEAP_PRICE_RATIO = 3
WARMUP_S = 20
MIN_DURATION = 30
MIN_REQS = 30
TIMEOUT = httpx.Timeout(timeout=300.0, connect=5.0)
SERVER_START_TIMEOUT_M = 30
MAX_RETRIES = 3
DEFAULT_CONCURRENCY = 1
DOLLY_DS = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-instruct"
DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_URL = "https://api.studio.testing.nebius.cloud/v1"
DOCKER_CONTAINER_NAME = "perf-test"

HF_CACHE = os.path.expanduser("~/.cache/huggingface")


@dataclass
class Prompt:
    messages: list[dict[str, str]]
    model: str | None = None
    token_count: int = 0
    user_id: str | None = None
    params: dict = field(default_factory=lambda: {"max_tokens": 500})

    def get_params(self):
        # Make a copy of params to avoid modifying the original
        params = self.params.copy()
        # Ensure max_tokens is reasonable
        if "max_tokens" in params and params["max_tokens"] > 100000:
            params["max_tokens"] = 100000
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        return params


@dataclass
class RequestStats:
    log_start_ts: float
    input_token_count: int
    output_token_count: int = 0
    first_chunk_ts: float | None = None
    last_chunk_ts: float | None = None
    ttft: float | None = None

    def get_tps(self) -> float | None:
        if self.first_chunk_ts is None or self.last_chunk_ts == self.first_chunk_ts:
            return None
        return self.output_token_count / (self.last_chunk_ts - self.first_chunk_ts)


@dataclass
class RawFetcherStats:
    ok_request_count: int = 0
    failed_request_count: int = 0
    max_load_start_ts: float | None = None
    reqs: list[RequestStats] = field(default_factory=list)
    errors: Counter = field(default_factory=Counter)

    def add_request(self, input_token_count: int) -> int:
        self.reqs.append(RequestStats(time.time(), input_token_count))
        return len(self.reqs) - 1

    def log_chunk(self, request_id, chunk):
        if not chunk.choices:
            return
        if isinstance(chunk, chat_completion.ChatCompletion) or isinstance(
            chunk, chat_completion_chunk.ChatCompletionChunk
        ):
            if not chunk.choices[0].delta.content:
                return
        elif isinstance(chunk, completion.Completion):
            if not chunk.choices[0].text:
                return
        else:
            raise ValueError(f"Unsupported completion type: {type(chunk)}")

        ts = time.time()
        if self.reqs[request_id].first_chunk_ts is None:
            self.reqs[request_id].first_chunk_ts = ts
            self.reqs[request_id].ttft = ts - self.reqs[request_id].log_start_ts
        self.reqs[request_id].last_chunk_ts = ts
        if chunk.usage and chunk.usage.completion_tokens == 0:
            return
        self.reqs[request_id].output_token_count += 1

    def dump_errors(self):
        print()
        print(
            f"OK requests: {self.ok_request_count}, errors: {self.failed_request_count}",
            file=sys.stderr,
        )
        for err, count in self.errors.items():
            print(f"{err} errors: {count}", file=sys.stderr)


class FetcherStats:
    def __init__(
        self, stats: RawFetcherStats, model: str, prompt_len: int, max_concurrency: int
    ):
        self.model = model
        case_name = {100: "short", 1000: "medium", None: "misc"}.get(
            prompt_len, str(prompt_len)
        )
        self.case_name = case_name
        self.max_concurrency = max_concurrency
        self.total_time = time.time() - stats.max_load_start_ts

        input_toks = [
            req.input_token_count for req in stats.reqs if req.input_token_count > 0
        ]
        output_toks = [
            req.output_token_count for req in stats.reqs if req.output_token_count > 0
        ]
        assert len(input_toks) >= MIN_REQS, f"{len(input_toks)} < {MIN_REQS}"
        assert len(output_toks) >= MIN_REQS, f"{len(output_toks)} < {MIN_REQS}"
        self.input_tps = sum(input_toks) / self.total_time
        self.output_tps = sum(output_toks) / self.total_time
        self.avg_input_len = mean(input_toks)
        self.avg_output_len = mean(output_toks)

        tps = [req.get_tps() for req in stats.reqs if req.get_tps() is not None]
        self.median_tps_per_req = median(tps)

        ttfts = [req.ttft for req in stats.reqs if req.ttft is not None]
        self.median_ttft, self.p99_ttft = median(ttfts), quantiles(ttfts, n=100)[-1]

    def get_weighted_tps(self) -> int:
        return int(self.input_tps + OUT_W * self.output_tps)

    def print(
        self,
        fmt: str,
        api_version: str,
        gpu_kind: str,
        gpu_count: str | int,
        cmd: str,
        fout,
        num_loras: int = 0,
        lora_concurrency: int = 0,
    ):
        if fmt == "txt":
            if cmd:
                print(cmd, file=fout)
            print(f"Time at target load: {self.total_time:0.2f}s", file=fout)
            print(file=fout)
            print(f"Prefill toks / s / endpoint avg: {int(self.input_tps)}", file=fout)
            print(
                f"Generated toks / s / endpoint avg: {int(self.output_tps)}", file=fout
            )
            print(
                f"Total toks / s / endpoint avg: {int(self.input_tps + self.output_tps)}",
                file=fout,
            )
            print(file=fout)
            print(
                f"generated toks / s / req median: {self.median_tps_per_req:0.2f}",
                file=fout,
            )
            print(
                f"ttft median: {self.median_ttft:0.2f}s, p99: {self.p99_ttft:0.2f}s",
                file=fout,
            )
            print(f"Avg input len: {int(self.avg_input_len)}", file=fout)
            print(f"Avg output len: {int(self.avg_output_len)}", file=fout)
            print("api version:", api_version, file=fout)
        elif fmt == "tsv":
            writer = csv.writer(fout, delimiter="\t")
            row = [
                self.model,
                self.case_name,
                "",  # input token price
                "",  # output token price
                gpu_kind,  # GPU kind
                "",  # GPU price
                f"{gpu_count}",  # GPU count
                f"{int(self.input_tps)}",
                f"{int(self.output_tps)}",
                f"{int(self.input_tps + self.output_tps)}",
                "",  # revenue
                api_version,
                f"{self.median_tps_per_req:0.2f}",
                f"{self.median_ttft:0.2f}",
                f"{self.p99_ttft:0.2f}",
                f"{self.max_concurrency}",
                cmd,
            ]
            if num_loras > 0:
                row += [num_loras, lora_concurrency]
            writer.writerow(row)
        fout.flush()


class Fetcher:
    def __init__(
        self,
        client: AsyncOpenAI,
        max_concurrency: int,
        default_model: str,
        num_loras: int = 0,
        lora_concurrency: int = 0,
        use_completion_endpoint: bool = False,
    ):
        self.default_model = default_model
        self.client = client
        self.max_concurrency = max_concurrency
        self.generated_token_count = 0
        self.cur_concurrency = 0
        self.cur_lora_concurrency = 0
        self.stats = RawFetcherStats()
        self.max_reqid = -1
        self.num_loras = num_loras
        self.max_lora_concurrency = lora_concurrency if self.num_loras > 0 else 0
        self.use_completion_endpoint = use_completion_endpoint

    async def busy(self) -> bool:
        assert self.max_concurrency > 0
        return self.cur_concurrency == self.max_concurrency

    def finished(self) -> bool:
        if self.stats.max_load_start_ts is None:
            return False
        if time.time() - self.stats.max_load_start_ts < MIN_DURATION:
            return False
        return len(self.stats.reqs) > MIN_REQS

    def new_reqid(self) -> int:
        self.max_reqid += 1
        return self.max_reqid

    def create_task(self, prompt: Prompt | list[Prompt]):
        self.cur_concurrency += 1
        return asyncio.create_task(self.fetch(prompt))

    def maybe_lora(self, prompt: Prompt) -> bool:
        if self.cur_lora_concurrency >= self.max_lora_concurrency:
            return False
        assert self.num_loras > 0
        n = random.randrange(self.num_loras)
        prompt.model = "lora" + str(n)
        self.cur_lora_concurrency += 1
        return True

    async def fetch(self, prompts: Prompt | list[Prompt]):
        if isinstance(prompts, list):
            for prompt in prompts:
                await self.fetch(prompt)
                self.cur_concurrency += 1
            self.cur_concurrency -= 1
            return
        prompt = prompts
        is_lora = self.maybe_lora(prompt)
        if (
            not self.stats.max_load_start_ts
            and self.cur_concurrency > 0.9 * self.max_concurrency
        ):
            self.stats.max_load_start_ts = time.time()
            print("\nReached target load", file=sys.stderr)
        reached_target_load = self.stats.max_load_start_ts is not None
        request_id = (
            self.stats.add_request(prompt.token_count) if reached_target_load else None
        )

        for _ in range(5):
            try:
                if self.use_completion_endpoint:
                    prompt_str = " ".join([msg["content"] for msg in prompt.messages])
                    stream = await self.client.completions.create(
                        prompt=prompt_str,
                        model=prompt.model or self.default_model,
                        **prompt.get_params(),
                    )
                else:
                    messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in prompt.messages
                    ]
                    stream = await self.client.chat.completions.create(
                        messages=messages,
                        model=prompt.model or self.default_model,
                        **prompt.get_params(),
                    )
                break
            except (APIConnectionError, NotFoundError) as e:
                self.cur_concurrency -= 1
                if is_lora:
                    self.cur_lora_concurrency -= 1
                logging.exception(e.message)
                self.stats.failed_request_count += 1
                self.stats.errors[str(type(e))] += 1
                return
            except RateLimitError as e:
                self.stats.errors[str(type(e))] += 1
                await asyncio.sleep(1)
            except Exception as e:
                traceback.print_exc()
                sys.exit(1)

        error = True
        try:
            async for chunk in stream:
                self.generated_token_count += 1
                if reached_target_load:
                    if request_id is None:
                        request_id = self.stats.add_request(0)
                    self.stats.log_chunk(request_id, chunk)
                    if chunk.usage:
                        self.stats.reqs[
                            request_id
                        ].input_token_count = chunk.usage.prompt_tokens
                        self.stats.reqs[
                            request_id
                        ].output_token_count = chunk.usage.completion_tokens
            error = False
        except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ReadError) as e:
            self.stats.errors[str(type(e))] += 1
        except Exception as e:
            traceback.print_exc()
            sys.exit(1)
        finally:
            if error:
                self.stats.failed_request_count += 1
            else:
                self.stats.ok_request_count += 1
            self.cur_concurrency -= 1
            if is_lora:
                self.cur_lora_concurrency -= 1

    async def print_progress(self, period_s=5):
        prev_token_count = self.generated_token_count
        print(file=sys.stderr)
        while True:
            tok_s = (self.generated_token_count - prev_token_count) / period_s
            prev_token_count = self.generated_token_count
            lora_stats = (
                f" (LoRas: {self.cur_lora_concurrency})" if self.num_loras > 0 else ""
            )
            print(
                f"[fetcher] concurrency: {self.cur_concurrency}{lora_stats}, tps: {tok_s:0.2f}",
                f"OK requests: {self.stats.ok_request_count}",
                file=sys.stderr,
            )
            await asyncio.sleep(period_s)


@dataclass
class Dataset:
    prompts: list[Prompt]
    prompt_len: int = 0

    @staticmethod
    def from_databricks_dolly(prompt_len, tokenizer):
        print("Downloading dataset", file=sys.stderr)
        response = requests.get(DOLLY_DS, timeout=60)
        assert response.status_code == 200
        dataset = [json.loads(line) for line in response.text.split("\n")]
        print("Dataset downloaded", file=sys.stderr)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        prompts = []
        for d in dataset:
            user_prompt = "\n".join([d["context"], d["instruction"]])
            cur_prompt_len = len(tokenizer(user_prompt)["input_ids"])
            if not prompt_len or 0.7 * prompt_len < cur_prompt_len < 1.3 * prompt_len:
                prompts.append(
                    Prompt(
                        [{"role": "user", "content": user_prompt}],
                        token_count=cur_prompt_len,
                    )
                )
        random.shuffle(prompts)
        print(f"Got {len(prompts)} prompts", file=sys.stderr)
        return Dataset(prompts, prompt_len)

    @staticmethod
    def from_file(path):
        if path.endswith(".jsonl"):
            return Dataset.from_jsonl(path)
        if path.endswith(".csv"):
            return Dataset.from_csv(path)
        raise ValueError(f"Unknown file type: {path}")

    @staticmethod
    def from_jsonl(path):
        prompts = []
        with open(path) as fin:
            for line in fin:
                data = json.loads(line)
                # Handle completions.jsonl format from fetch_prompts
                if "prompt" in data:
                    try:
                        # Parse each message string into JSON object
                        messages = []
                        for msg_str in data["prompt"]:
                            try:
                                msg = json.loads(msg_str)
                                messages.append(msg)
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse message: {msg_str}\nError: {e}", file=sys.stderr)
                                continue
                        
                        if messages:  # Only add if we have valid messages
                            # Get params with reasonable defaults
                            params = data.get("params", {}) or {}
                            # Set reasonable max_tokens if not specified or too large
                            if "max_tokens" in params and params["max_tokens"] > 100000:
                                params["max_tokens"] = 100000
                            prompts.append(
                                Prompt(
                                    messages=messages,
                                    params=params
                                )
                            )
                    except Exception as e:
                        print(f"Error processing line: {data}\nError: {e}", file=sys.stderr)
                        continue
                # Handle original format 
                elif "messages" in data:
                    prompts.append(
                        Prompt(
                            messages=data["messages"], 
                            model=data.get("model"),
                            user_id=data.get("user_id")
                        )
                    )
                else:
                    print(f"Skipping invalid line: {data}", file=sys.stderr)
                    continue
                
        print(f"Read {len(prompts)} prompts", file=sys.stderr)
        return Dataset(prompts)

    @staticmethod
    def from_csv(path):
        with open(path, encoding="utf-8-sig") as fin:
            reader = csv.DictReader(fin)
            prompts = []
            for row in reader:
                if "id" not in row:
                    print(row)
                    sys.exit(1)
                if not row["id"].startswith("chat"):
                    continue
                try:
                    prompt = Prompt(
                        [
                            json.loads(str_message)
                            for str_message in json.loads(row["prompt"])
                        ],
                        row["model_flavor_id"],
                        params=json.loads(row["params"]),
                    )
                except Exception as e:
                    print(row, e)
                    sys.exit(1)
                prompts.append(prompt)
        return Dataset(prompts)


async def fetch_concurrently(
    api_url: str,
    model: str,
    limits: httpx.Limits,
    token: str,
    dataset: Dataset,
    num_loras: int = 0,
    lora_concurrency: int = 0,
    use_completion_endpoint: bool = False,
) -> FetcherStats | None:
    async with httpx.AsyncClient(limits=limits) as httpx_client:
        openai = AsyncOpenAI(
            api_key=token,
            base_url=api_url,
            http_client=httpx_client,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        fetcher = Fetcher(
            openai,
            limits.max_connections,
            model,
            num_loras,
            lora_concurrency,
            use_completion_endpoint,
        )
        progress = asyncio.create_task(fetcher.print_progress())
        tasks = []
        start_ts = time.time()
        prompt_index = 0
        assert len(dataset.prompts) > 0, "Dataset must not be empty"
        while not fetcher.finished():
            while await fetcher.busy():
                await asyncio.sleep(0.01)
            tasks.append(fetcher.create_task(dataset.prompts[prompt_index]))
            prompt_index = (prompt_index + 1) % len(dataset.prompts)
            expected_concurrency = (
                (time.time() - start_ts) / WARMUP_S * fetcher.max_concurrency
            )
            if fetcher.cur_concurrency >= expected_concurrency:
                await asyncio.sleep(0.1)
        print(file=sys.stderr)
        if fetcher.stats.failed_request_count > 0:
            fetcher.stats.dump_errors()
        if fetcher.stats.failed_request_count > 0.01 * fetcher.stats.ok_request_count:
            print(
                "Too many requests have failed, stats for this session were ignored",
                file=sys.stderr,
            )
            stats = None
        else:
            try:
                stats = FetcherStats(
                    fetcher.stats, model, dataset.prompt_len, fetcher.max_concurrency
                )
            except AssertionError as e:
                stats = None
                print("Can't construct FetcherStats:", e, file=sys.stderr)
        progress.cancel()
        print(file=sys.stderr)
        for task in tasks:
            await task
    return stats


# TODO join fetch concurrently and emulate user load into one class or function
async def emulate_user_load(
    api_url: str,
    token: str,
    dataset: Dataset,
    num_loras: int = 0,
    lora_concurrency: int = 0,
    use_completion_endpoint: bool = False,
) -> FetcherStats:
    user2prompts = defaultdict(list)
    for prompt in dataset.prompts:
        user2prompts[prompt.user_id].append(prompt)
    concurrency = len(user2prompts)
    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=concurrency, max_keepalive_connections=concurrency
        )
    ) as httpx_client:
        client = AsyncOpenAI(
            api_key=token,
            base_url=api_url,
            http_client=httpx_client,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        fetcher = Fetcher(
            client, 0, "", num_loras, lora_concurrency, use_completion_endpoint
        )
        async with asyncio.TaskGroup() as tg:
            active_tasks = []
            progress = tg.create_task(fetcher.print_progress())
            for prompts in user2prompts.values():
                active_tasks.append(fetcher.create_task(prompts))
            for task in active_tasks:
                await task
            progress.cancel()
            fetcher.stats.dump_errors()
    return FetcherStats(fetcher.stats, "", 0, concurrency)


def fetch_api_version(api_url: str) -> str:
    try:
        version_url = api_url.rsplit("/", 1)[0] + "/version"
        resp = requests.get(version_url, timeout=TIMEOUT.connect)
        if resp.status_code != 200:
            return ""
        return resp.json().get("version", "").lstrip("v")
    except Exception:
        return ""


async def collect_metrics(
    models: list[str],
    max_connections: list[int],
    api_url: str,
    token: str,
    datasets: list[Dataset],
    num_loras: int = 0,
    lora_concurrency: int = 0,
    use_completion_endpoint: bool = False,
) -> AsyncIterable[FetcherStats]:
    if len(datasets) == 1 and datasets[0].prompts[0].user_id is not None:
        yield await emulate_user_load(
            api_url,
            token,
            datasets[0],
            num_loras,
            lora_concurrency,
            use_completion_endpoint,
        )
        return

    if not models:
        resp = requests.get(
            api_url + "/models",
            timeout=TIMEOUT.connect,
            headers={"Authorization": f"bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        models = [item["id"] for item in resp.json()["data"]]
        print("Using fetched model ids:", ", ".join(models), file=sys.stderr)

    for model in models:
        for max_conn in max_connections:
            for dataset in datasets:
                print(f"{model}, {max_conn}, {dataset.prompt_len}", file=sys.stderr)
                limits = httpx.Limits(
                    max_connections=max_conn, max_keepalive_connections=max_conn
                )
                stats = await fetch_concurrently(
                    api_url,
                    model,
                    limits,
                    token,
                    dataset,
                    num_loras,
                    lora_concurrency,
                    use_completion_endpoint,
                )
                if stats is None:
                    continue
                yield stats


def wait_server(backend, port) -> bool:
    start_ts = time.time()
    while time.time() - start_ts < 60 * SERVER_START_TIMEOUT_M:
        if backend.startswith("vllm"):
            url = f"http://127.0.0.1:{port}/health"
        elif backend == "sglang":
            url = f"http://127.0.0.1:{port}/get_model_info"
        else:
            raise ValueError(f"Unknown backend: {backend}")
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            pass
        time.sleep(1)
    return False


def kill_vllm():
    call(["bash", "-c", "nvidia-smi | grep python3 | awk '{print $5}' | xargs kill -9"])
    call(["bash", "-c", "ps x | grep 'vllm serve' | awk '{print $1}' | xargs kill -9"])
    call(["bash", "-c", f"docker container stop {DOCKER_CONTAINER_NAME} --time 10"])
    call(["bash", "-c", f"docker container rm {DOCKER_CONTAINER_NAME}"])
    time.sleep(10)


@dataclass
class Profile:
    model_id: str
    served_model_name: str
    tp: int
    extra_args: str
    backend: str
    concurrency: int
    num_loras: int
    lora_path: str
    lora_concurrency: int
    use_completion_endpoint: bool

    CMD_PREFIX = (
        f"docker run --runtime nvidia --gpus all --shm-size=40gb --rm --name {DOCKER_CONTAINER_NAME}"
        f" -v {HF_CACHE}:/root/.cache/huggingface -v /model-storage:/model-storage -v /home:/home"
        f" --env HUGGING_FACE_HUB_TOKEN={HF_TOKEN} --env VLLM_LOGGING_LEVEL=DEBUG"
        f" --ipc=host"
    )

    def get_cmd(self, port=8000) -> str:
        lora_args = ""
        if self.num_loras > 0:
            if self.backend != "sglang":
                lora_args += " --enable-lora"
                lora_args += f" --max-loras {self.num_loras}"
                lora_args += " --lora-modules"
            else:
                lora_args += f" --max-loras-per-batch {self.num_loras}"
                lora_args += " --lora-paths"
            for n in range(self.num_loras):
                lora_args += f" lora{n}={self.lora_path}"
        if self.backend == "vllm_docker":
            return (
                f"{self.CMD_PREFIX} -p {port}:{port} vllm/vllm-openai:{VLLM_VERSION} --disable-log-requests "
                f"--port {port} --model {self.model_id} -tp {self.tp} {self.extra_args} {lora_args}"
            )
        if self.backend == "vllm_serve":
            return f"vllm serve {self.model_id} --disable-log-requests --port {port} -tp {self.tp} {self.extra_args} {lora_args}"
        if self.backend == "sglang":
            return (
                f"{self.CMD_PREFIX} -p {port}:{port} lmsysorg/sglang:{SGLANG_VERSION} python3 -m sglang.launch_server "
                f"--port {port} --host 0.0.0.0 --model-path {self.model_id} --tp {self.tp} {self.extra_args} {lora_args}"
            )
        raise ValueError(f"unknown backend: {self.backend}")


def get_plan(params: dict) -> list[Profile]:
    axes = []
    for key, values in params.items():
        if isinstance(values, list):
            axes.append([(key, value) for value in values])
    profiles: list[Profile] = []
    concurrency = int(params["concurrency"])
    for args in product(*axes):
        tp = 1
        max_loras = 0
        lora_path = None
        lora_concurrency = concurrency
        extra_args = []
        for key, value in args:
            if key.strip("-") in ("tp", "tensor-parallel-size"):
                tp = int(value)
            elif key.strip("-") == "lora-path":
                lora_path = value
            elif key.strip("-") == "lora-concurrency":
                lora_concurrency = int(value)
                if lora_concurrency > concurrency:
                    raise ValueError(
                        f"Number concurrent lora requests {lora_concurrency} "
                        f"cannot be larger than total request concurrency {concurrency}"
                    )
            elif key.strip("-") == "max-loras":
                max_loras = int(value)
            else:
                extra_args.append(" ".join([key, value]))
        profiles.append(
            Profile(
                params["id"],
                params.get("--served-model-name", [params["id"]])[0],
                tp,
                " ".join(extra_args),
                params["backend"],
                concurrency,
                max_loras,
                lora_path,
                lora_concurrency,
                params.get("use_completion_endpoint", False),
            )
        )
    return profiles


async def run_command_and_collect_metrics(
    datasets: list[Dataset],
    output_format: str,
    gpu_kind: str,
    params,
    fout,
    cached_results: set[str],
) -> None:
    for model_params in params:
        plan = get_plan(model_params)
        all_stats = []
        api_version = None

        # Group profiles by command, skipping groups where all profiles are cached
        profile_groups = OrderedDict()
        for profile in plan:
            if (
                signature := get_signature(profile, gpu_kind, profile.served_model_name)
            ) in cached_results:
                print(f"Skipping cached run\n{signature}", file=sys.stderr)
                continue
            cmd = profile.get_cmd()
            if cmd not in profile_groups:
                profile_groups[cmd] = []
            profile_groups[cmd].append(profile)

        for __, profiles in profile_groups.items():
            kill_vllm()
            port = random.randint(8000, 9000)
            cmd = profile.get_cmd(port)
            with Popen(cmd.split(), stdout=sys.stderr) as server:
                print(f"Starting new server\n{cmd}", file=sys.stderr)
                if not wait_server(profiles[0].backend, port):
                    print("Timed out waiting for server to start", file=sys.stderr)
                    server.kill()
                    kill_vllm()
                    continue
                if server.returncode is not None:
                    print("Server failed to start", file=sys.stderr)
                    continue

                if profiles[0].backend == "vllm_docker":
                    api_version = VLLM_VERSION.lstrip("v")
                else:
                    api_version = fetch_api_version(f"http://127.0.0.1:{port}/v1")

                for i, profile in enumerate(profiles):
                    if i > 0:
                        print(f"Reusing running server\n{cmd}", file=sys.stderr)

                    async for stats in collect_metrics(
                        [profile.served_model_name],
                        [profile.concurrency],
                        f"http://127.0.0.1:{port}/v1",
                        "EMPTY",
                        datasets,
                        profile.num_loras,
                        profile.lora_concurrency,
                        profile.use_completion_endpoint,
                    ):
                        stats.print(
                            output_format,
                            api_version,
                            gpu_kind,
                            profile.tp,
                            cmd,
                            fout,
                            profile.num_loras,
                            profile.lora_concurrency,
                        )
                        all_stats.append((profile, stats))
                print("Killing process group", file=sys.stderr)
                server.terminate()
                time.sleep(10)
                server.kill()
                kill_vllm()
                time.sleep(10)
                print("Done killing process group", file=sys.stderr)


def read_cached_results(filename: str) -> set:
    if not os.path.exists(filename):
        return set()
    with open(filename) as fin:
        rows = [line.strip("\n").split("\t") for line in fin.readlines()]
        return {"#".join([p[0], p[4], p[6], p[11], p[15], p[16]]) for p in rows}


def get_signature(profile: Profile, gpu_kind: str, model_id: str):
    return "#".join(
        [
            model_id,
            gpu_kind,
            str(profile.tp),
            VLLM_VERSION.lstrip("v"),
            str(profile.concurrency),
            profile.get_cmd(),
        ]
    )


async def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="dolly",
        help="Path to jsonl dataset. By default a dataset from hf will be downloaded.",
    )
    parser.add_argument(
        "-t", "--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="HF tokenizer"
    )
    parser.add_argument(
        "-f", "--format", type=str, choices=["txt", "tsv"], default="txt"
    )
    parser.add_argument("-u", "--api-url", type=str, default=DEFAULT_URL)
    parser.add_argument("-a", "--api-key", type=str, default="EMPTY")
    parser.add_argument(
        "-r", "--run-server", action="store_true", help="Run own vllm server"
    )
    parser.add_argument(
        "-p", "--plan", type=str, help="plan.json with server run params"
    )
    parser.add_argument("-g", "--gpu-kind", type=str)

    parser.add_argument(
        "-l",
        "--prompt-len",
        nargs="*",
        type=int,
        default=[None],
        help="Filter dataset by input token count",
    )
    parser.add_argument("-j", "--max-connections", nargs="*", type=int, default=[])
    parser.add_argument("-m", "--models", type=str, nargs="*", default=None)
    parser.add_argument(
        "-n", "--note", type=str, default="", help="Note added to each tsv output row"
    )
    parser.add_argument(
        "--max-loras",
        nargs="*",
        type=int,
        default=[],
        help="Number of lora adapters to load on GPU",
    )
    parser.add_argument(
        "-L",
        "--lora-concurrency",
        nargs="*",
        type=int,
        default=[],
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "-C",
        "--use-completion-endpoint",
        action="store_true",
        help="Use completion instead of chat.completion endpoint, for models without chat template",
    )

    parser.add_argument(
        "-o", "--output", type=str, help="report.tsv if format is tsv, otherwise stdout"
    )
    args = parser.parse_args()

    if args.output:
        output_filename = args.output
    elif args.format == "tsv":
        output_filename = "report.tsv"
    else:
        output_filename = "/dev/stdout"

    if not os.path.exists(output_filename):
        Path(output_filename).touch()

    cached_results = set()
    if args.format == "tsv" and args.run_server:
        cached_results = read_cached_results(output_filename)
        print(cached_results)

    login(token=HF_TOKEN)
    if args.dataset == "dolly":
        datasets = [
            Dataset.from_databricks_dolly(l, args.tokenizer) for l in args.prompt_len
        ]
    else:
        datasets = [Dataset.from_file(args.dataset)]

    with open(output_filename, "a", newline="") as fout:
        if args.run_server:
            assert (
                args.api_url == DEFAULT_URL
            ), "--api-url argument can't be used with --run-server"
            with open(args.plan) as fin:
                plan = json.load(fin)
            await run_command_and_collect_metrics(
                datasets, args.format, args.gpu_kind, plan, fout, cached_results
            )
            return

        api_version = fetch_api_version(args.api_url)
        async for stats in collect_metrics(
            args.models,
            args.max_connections,
            args.api_url,
            args.api_key,
            datasets,
            use_completion_endpoint=args.use_completion_endpoint,
        ):
            stats.print(args.format, api_version, args.gpu_kind, "", args.note, fout)


if __name__ == "__main__":
    asyncio.run(main())
