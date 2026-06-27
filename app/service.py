"""Chat service: glue between the API layer and the instrumented LLM call.

Holds the retry/backoff policy and resolves prompts from
the registry so every request is tagged with a version.
"""

from __future__ import annotations

import time
from typing import Optional

from .config import get_settings
from .llm_client import LLMClient, LLMError, LLMRateLimited, LLMTimeout
from .models import ChatRequest, ChatResponse
from .prompt_registry import registry
from .tracing import trace_call
from .validators import validate

_client = LLMClient()


def _build_messages(req: ChatRequest) -> tuple[list[dict[str, str]], Optional[str], str]:
    """Resolve the request into (messages, prompt_version, input_text)."""
    if req.prompt_name:
        template = registry.get(req.prompt_name, req.prompt_version)
        rendered = template.render(req.variables)
        return (
            [{"role": "user", "content": rendered}],
            template.version,
            rendered,
        )
    if req.messages:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        input_text = "\n".join(m["content"] for m in messages)
        return messages, req.prompt_version, input_text
    raise ValueError("Request must supply either prompt_name or messages.")


def run_chat(req: ChatRequest) -> ChatResponse:
    settings = get_settings()
    messages, prompt_version, input_text = _build_messages(req)
    model = req.model or settings.llm_model

    with trace_call(
        endpoint="/chat",
        model=model,
        provider=settings.llm_provider,
        prompt_version=prompt_version,
        input_text=input_text,
        metadata=req.metadata,
    ) as span:
        result = _call_with_retries(messages, model, req.temperature, span)
        span.record(result)
        # Output-quality check: flag responses that violate the prompt contract.
        reason = validate(req.prompt_name, result.text)
        if reason:
            span.mark_bad_output(reason)

    t = span.trace
    return ChatResponse(
        trace_id=t.id,
        output=t.output or result.text,
        model=t.model,
        prompt_version=t.prompt_version,
        prompt_tokens=t.prompt_tokens,
        completion_tokens=t.completion_tokens,
        cost_usd=t.cost_usd,
        latency_ms=t.latency_ms,
        status=t.status,
    )


def _call_with_retries(messages, model, temperature, span):
    """Retry transient failures (timeout / rate limit) with exponential backoff.

    Records the retry count on the span's trace. Non-transient errors and the
    final attempt's failure propagate to the trace layer for classification.
    """
    settings = get_settings()
    attempt = 0
    while True:
        try:
            return _client.complete(messages, model=model, temperature=temperature)
        except (LLMTimeout, LLMRateLimited) as exc:
            if attempt >= settings.llm_max_retries:
                span.trace.retries = attempt
                raise
            attempt += 1
            span.trace.retries = attempt
            time.sleep(min(0.2 * (2 ** (attempt - 1)), 2.0))
        except LLMError:
            span.trace.retries = attempt
            raise
