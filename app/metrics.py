"""Prometheus instruments (wired early so traces feed them).

Exposed at /metrics. A single update_from_trace() call keeps the Prometheus
view and the SQLite trace store in lockstep.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from .models import Trace

REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM requests processed.",
    ["endpoint", "model", "prompt_version", "status"],
)

COST_USD = Counter(
    "llm_cost_usd_total",
    "Cumulative (simulated) USD cost.",
    ["model", "prompt_version"],
)

TOKENS = Counter(
    "llm_tokens_total",
    "Cumulative tokens processed.",
    ["model", "kind"],  # kind = prompt | completion
)

LATENCY = Histogram(
    "llm_request_latency_ms",
    "End-to-end request latency in milliseconds.",
    ["model", "prompt_version"],
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000),
)


def update_from_trace(trace: Trace) -> None:
    pv = trace.prompt_version or "none"
    REQUESTS.labels(trace.endpoint, trace.model, pv, trace.status.value).inc()
    COST_USD.labels(trace.model, pv).inc(trace.cost_usd)
    TOKENS.labels(trace.model, "prompt").inc(trace.prompt_tokens)
    TOKENS.labels(trace.model, "completion").inc(trace.completion_tokens)
    LATENCY.labels(trace.model, pv).observe(trace.latency_ms)
