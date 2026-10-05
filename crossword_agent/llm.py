"""LLM client for Nebius Token Factory (OpenAI-compatible chat completions).

Responsibilities:
  * load credentials from the environment / a local .env file
  * JSON-mode completions with retries and lenient JSON parsing
  * an on-disk response cache so evaluation reruns are cheap and reproducible
  * token / latency accounting (thread-safe; batches run concurrently)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "openai/gpt-oss-120b"
API_KEY_ENV = "NEBIUS_API_KEY"

# USD per 1M tokens (prompt, completion) as listed by Nebius Token Factory on 2026-09-19.
# Used only for cost estimates in evaluation reports; unknown models report cost as None.
PRICING: dict[str, tuple[float, float]] = {
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "deepseek-ai/DeepSeek-V4-Pro": (1.75, 3.50),
    "deepseek-ai/DeepSeek-V4-Pro-0813": (1.32, 3.96),
    "moonshotai/Kimi-K3": (3.00, 15.00),
    "moonshotai/Kimi-K2.6": (0.95, 4.00),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "Qwen/Qwen3.5-397B-A17B": (0.60, 3.60),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.20, 0.60),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (0.10, 0.30),
    "zai-org/GLM-5.3": (1.40, 4.40),
    "zai-org/GLM-5.3-Flash": (0.15, 0.50),
    "google/gemma-3-27b-it": (0.10, 0.30),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = PRICING.get(model)
    if price is None:
        return None
    return prompt_tokens / 1e6 * price[0] + completion_tokens / 1e6 * price[1]


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE per line). Never overrides existing env vars."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMError(RuntimeError):
    """Raised when a completion cannot be obtained after retries."""


class LengthExhausted(LLMError):
    """The model spent the whole (already raised) budget reasoning; callers should shrink the request."""


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    errors: int = 0
    # replayed from the on-disk cache: what the same run would have cost live
    cached_prompt_tokens: int = 0
    cached_completion_tokens: int = 0
    cached_latency_s: float = 0.0

    def add(self, other: Usage) -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 12000
    max_tokens_ceiling: int = 24000  # one adaptive retry when reasoning exhausts the budget
    reasoning_effort: str | None = None  # low | medium | high for models that support it (gpt-oss)
    timeout: float = 240.0
    max_retries: int = 3
    cache_dir: Path | None = Path(".cache/llm")
    use_cache: bool = True

    @classmethod
    def from_env(cls, **overrides) -> LLMConfig:
        load_dotenv()
        cfg = cls(
            model=os.environ.get("CROSSWORD_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get(API_KEY_ENV),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


def parse_json_lenient(text: str) -> dict:
    """Parse JSON from model output, tolerating code fences and leading prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class NebiusClient:
    """Thin, resilient wrapper over the OpenAI SDK pointed at Nebius Token Factory."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        if not self.config.api_key:
            raise LLMError(
                f"No API key: set {API_KEY_ENV} in the environment or in a .env file."
            )
        from openai import OpenAI  # lazy import keeps `import crossword_agent` light

        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=0,  # we manage retries ourselves so we can count them
        )
        self.usage = Usage()
        self._lock = threading.Lock()
        if self.config.cache_dir and self.config.use_cache:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- cache ----------------------------------------------------------
    def _cache_path(self, system: str, user: str, max_tokens: int) -> Path | None:
        if not (self.config.cache_dir and self.config.use_cache):
            return None
        h = hashlib.sha256(
            json.dumps(
                [self.config.model, self.config.temperature, max_tokens, self.config.reasoning_effort, system, user]
            ).encode()
        ).hexdigest()[:32]
        return self.config.cache_dir / f"{h}.json"

    # ---- completion -----------------------------------------------------
    def complete_json(self, system: str, user: str, *, max_tokens: int | None = None, nocache: bool = False) -> dict:
        """Return the parsed JSON object the model produced for this prompt."""
        max_tokens = max_tokens or self.config.max_tokens
        cache_path = self._cache_path(system, user, max_tokens)
        if cache_path and cache_path.exists() and not nocache:
            cached = json.loads(cache_path.read_text())
            cu = cached.get("usage") or {}
            with self._lock:
                self.usage.cache_hits += 1
                self.usage.cached_prompt_tokens += cu.get("prompt_tokens") or 0
                self.usage.cached_completion_tokens += cu.get("completion_tokens") or 0
                self.usage.cached_latency_s += cached.get("latency_s") or 0.0
            return cached["data"]

        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )

        last_err: Exception | None = None
        budget = max_tokens
        for attempt in range(self.config.max_retries + 1):
            t0 = time.time()
            try:
                extra = {"reasoning_effort": self.config.reasoning_effort} if self.config.reasoning_effort else {}
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    **extra,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=budget,
                    response_format={"type": "json_object"},
                )
                dt = time.time() - t0
                choice = resp.choices[0]
                content = choice.message.content or ""
                usage = resp.usage
                reasoning_tokens = 0
                details = getattr(usage, "completion_tokens_details", None) if usage else None
                if details is not None:
                    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                with self._lock:
                    self.usage.calls += 1
                    self.usage.latency_s += dt
                    if usage:
                        self.usage.prompt_tokens += usage.prompt_tokens or 0
                        self.usage.completion_tokens += usage.completion_tokens or 0
                        self.usage.reasoning_tokens += reasoning_tokens
                if choice.finish_reason == "length" and not content.strip():
                    if budget >= self.config.max_tokens_ceiling:
                        raise LengthExhausted(f"model exhausted {budget} tokens while reasoning")
                    budget = min(self.config.max_tokens_ceiling, int(budget * 2))
                    raise LLMError(f"model exhausted max_tokens while reasoning; retrying with {budget}")
                data = parse_json_lenient(content)
                if cache_path:
                    cache_path.write_text(
                        json.dumps(
                            {
                                "model": self.config.model,
                                "data": data,
                                "usage": {
                                    "prompt_tokens": usage.prompt_tokens if usage else None,
                                    "completion_tokens": usage.completion_tokens if usage else None,
                                },
                                "latency_s": dt,
                            }
                        )
                    )
                return data
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_err = e
            except APIStatusError as e:
                last_err = e
                if e.status_code < 500 and e.status_code != 429:
                    break  # a 4xx other than rate limiting will not fix itself
            except LengthExhausted:
                raise
            except (json.JSONDecodeError, LLMError) as e:
                last_err = e  # malformed output: retry (temperature 0 but servers vary)
            with self._lock:
                self.usage.errors += 1
            delay = min(30.0, (2**attempt) + random.uniform(0, 1))
            log.warning("LLM call failed (%s); retry %d in %.1fs", last_err, attempt + 1, delay)
            time.sleep(delay)
        raise LLMError(f"LLM call failed after retries: {last_err}")
