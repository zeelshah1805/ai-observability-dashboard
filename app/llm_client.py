"""LLMClient — a thin, provider-agnostic wrapper around an OpenAI-compatible
chat/completions API.

Supported providers: groq, openrouter, ollama (all OpenAI-compatible) and a
built-in `mock` provider that synthesizes plausible responses with no network
or API key, so the whole stack runs and demos with zero setup.

The client is deliberately dumb about observability — it just returns a
`CompletionResult` (or raises a typed error). The `@traced` layer is what turns
that into a Trace. Keeping the two separate is the whole design idea.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import httpx

from .config import Settings, get_settings

_SENTENCE_REQ = re.compile(r"exactly (\d+) sentence", re.IGNORECASE)


# ---- Typed errors so the trace layer can classify status -------------------


class LLMError(Exception):
    """Generic LLM call failure."""


class LLMTimeout(LLMError):
    """The provider did not respond within the timeout."""


class LLMRateLimited(LLMError):
    """The provider returned 429 / rate-limit."""


@dataclass
class CompletionResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: Optional[int] = None
    raw: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _encoder():
    """Load (and cache) the tiktoken encoder once. First load is slow."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(text: str) -> int:
    """Fallback token estimate when the provider returns no usage.

    Uses tiktoken if available; otherwise a ~4-chars-per-token heuristic.
    Estimates are labeled as such by callers.
    """
    try:
        return len(_encoder().encode(text))
    except Exception:
        return max(1, len(text) // 4)


class LLMClient:
    """Synchronous client. One instance per process is fine."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_provider.lower()

    # -- public API ----------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> CompletionResult:
        model = model or self.settings.llm_model
        if self.provider == "mock":
            return self._complete_mock(messages, model, temperature)
        return self._complete_openai_compatible(messages, model, temperature)

    # -- mock provider -------------------------------------------------------

    def _complete_mock(
        self, messages: list[dict[str, str]], model: str, temperature: float
    ) -> CompletionResult:
        """Synthesize a response with no network.

        Fresh entropy per call (so retries can recover and the dashboard gets a
        realistic distribution). Crucially, the mock *simulates instruction
        following*: when the prompt asks for "exactly N sentences" it complies
        most of the time, producing shorter, faster, cheaper output. An
        unconstrained prompt rambles to a random length. That difference is what
        powers the v1-vs-v2 prompt-regression demo.
        """
        rng = random.Random()
        prompt_text = "\n".join(m.get("content", "") for m in messages)

        # Occasional transient/hard failures so error tracking has signal.
        roll = rng.random()
        if roll < 0.02:
            raise LLMTimeout("mock: simulated upstream timeout")
        if roll < 0.04:
            raise LLMRateLimited("mock: simulated rate limit (429)")
        if roll < 0.05:
            raise LLMError("mock: simulated upstream 500")

        # Does the prompt constrain the output length?
        m = _SENTENCE_REQ.search(prompt_text)
        if m:
            target = int(m.group(1))
            # Constrained prompt: comply ~90% of the time.
            if rng.random() < 0.90:
                n_sentences = target
            else:
                n_sentences = max(1, target + rng.choice([-1, 1, 2]))
        else:
            # Unconstrained prompt: variable and often rambling — the kind of
            # output an under-specified prompt produces. Longer => slower and
            # pricier, which is exactly what the v2 fix improves.
            n_sentences = rng.randint(2, 8)

        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            prompt_text,
        )
        snippet = re.sub(r"[.!?]+", "", last_user.strip().replace("\n", " "))[:80]
        # NB: keep sentence punctuation out of the body except the real
        # sentence terminators, so the bad-output validator can count reliably.
        sentences = [f"Synthesized summary regarding {snippet}"]
        for i in range(1, n_sentences):
            sentences.append(f"Additional synthesized detail number {i}")
        text = ". ".join(sentences) + "."

        # Generation dominates real latency, so scale with output length. v2's
        # shorter, constrained output ends up faster than v1's rambling output.
        base_latency = rng.uniform(0.08, 0.25)
        gen_latency = n_sentences * rng.uniform(0.08, 0.22)
        latency = (base_latency + gen_latency) * (1.0 + temperature * 0.2)
        time.sleep(latency)

        prompt_tokens = _estimate_tokens(prompt_text)
        completion_tokens = _estimate_tokens(text)
        ttft = int(latency * 1000 * rng.uniform(0.2, 0.5))
        return CompletionResult(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft,
            raw={"provider": "mock", "sentences": n_sentences},
        )

    # -- real providers (OpenAI-compatible) ----------------------------------

    def _complete_openai_compatible(
        self, messages: list[dict[str, str]], model: str, temperature: float
    ) -> CompletionResult:
        base_url = self.settings.base_url_for(self.provider)
        api_key = self.settings.api_key_for(self.provider)
        if not base_url:
            raise LLMError(f"No base_url configured for provider {self.provider!r}")
        if self.provider in ("groq", "openrouter") and not api_key:
            raise LLMError(f"Missing API key for provider {self.provider!r}")

        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "messages": messages, "temperature": temperature}

        try:
            with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                resp = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc

        if resp.status_code == 429:
            raise LLMRateLimited(resp.text[:300])
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        # Fall back to estimates and keep that fact in raw for transparency.
        estimated = False
        if prompt_tokens is None:
            prompt_tokens = _estimate_tokens(
                "\n".join(m.get("content", "") for m in messages)
            )
            estimated = True
        if completion_tokens is None:
            completion_tokens = _estimate_tokens(text)
            estimated = True

        return CompletionResult(
            text=text,
            model=data.get("model", model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw={"provider": self.provider, "tokens_estimated": estimated},
        )
