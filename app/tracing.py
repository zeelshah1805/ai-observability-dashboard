"""The cross-cutting instrumentation layer.

`trace_call(...)` is a context manager that captures latency, usage, cost,
status, and errors for an LLM call and — on exit — persists a Trace row and
updates Prometheus. Application code stays clean; observability is one `with`
block, and everything else stays unaware of it.

    with trace_call(endpoint="/chat", model=m, provider=p,
                    prompt_version=v, input_text=prompt) as tr:
        result = client.complete(messages, model=m)
        tr.record(result)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from . import db, metrics
from .config import get_settings
from .llm_client import (
    CompletionResult,
    LLMError,
    LLMRateLimited,
    LLMTimeout,
)
from .models import Status, Trace


class _Span:
    """Mutable handle a caller fills in during the traced block."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self._bad_output: Optional[str] = None

    def record(self, result: CompletionResult) -> None:
        """Attach a successful completion's usage/cost/output to the trace."""
        from .pricing import compute_cost

        t = self.trace
        t.model = result.model or t.model
        t.prompt_tokens = result.prompt_tokens
        t.completion_tokens = result.completion_tokens
        t.cost_usd = compute_cost(t.model, t.prompt_tokens, t.completion_tokens)
        t.ttft_ms = result.ttft_ms
        if get_settings().store_prompts:
            t.output = result.text

    def mark_bad_output(self, reason: str) -> None:
        """Flag a structurally-valid-but-wrong response."""
        self._bad_output = reason


def _classify(exc: BaseException) -> tuple[Status, str]:
    if isinstance(exc, LLMTimeout):
        return Status.TIMEOUT, "timeout"
    if isinstance(exc, LLMRateLimited):
        return Status.RATE_LIMITED, "rate_limited"
    if isinstance(exc, LLMError):
        return Status.ERROR, type(exc).__name__
    return Status.ERROR, type(exc).__name__


@contextmanager
def trace_call(
    *,
    endpoint: str,
    model: str,
    provider: str,
    prompt_version: Optional[str] = None,
    input_text: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    retries: int = 0,
) -> Iterator[_Span]:
    settings = get_settings()
    trace = Trace(
        endpoint=endpoint,
        model=model,
        provider=provider,
        prompt_version=prompt_version,
        input=input_text if settings.store_prompts else None,
        metadata=metadata or {},
        retries=retries,
    )
    span = _Span(trace)
    start = time.perf_counter()
    try:
        yield span
    except BaseException as exc:  # noqa: BLE001 — we re-raise after recording
        trace.status, trace.error_type = _classify(exc)
        trace.latency_ms = int((time.perf_counter() - start) * 1000)
        _persist(trace)
        raise
    else:
        trace.latency_ms = int((time.perf_counter() - start) * 1000)
        if span._bad_output is not None:
            trace.status = Status.BAD_OUTPUT
            trace.error_type = span._bad_output
        else:
            trace.status = Status.OK
        _persist(trace)


def _persist(trace: Trace) -> None:
    db.insert_trace(trace)
    try:
        metrics.update_from_trace(trace)
    except Exception:
        # Metrics must never take down the request path.
        pass
