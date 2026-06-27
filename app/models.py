"""Core data model: the Trace and supporting enums / request schemas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Terminal status of a traced LLM call."""

    OK = "ok"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    BAD_OUTPUT = "bad_output"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Trace(BaseModel):
    """A single instrumented LLM request — the backbone record of the system."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=_now)
    endpoint: str = "/chat"
    model: str = ""
    provider: str = ""
    prompt_version: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    ttft_ms: Optional[int] = None
    status: Status = Status.OK
    error_type: Optional[str] = None
    retries: int = 0
    input: Optional[str] = None
    output: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---- API request/response schemas ------------------------------------------


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    """Inbound /chat payload.

    Either supply a registered prompt (`prompt_name` + `variables`) so the
    request is tagged with a version, or pass raw `messages` directly.
    """

    prompt_name: Optional[str] = None
    prompt_version: Optional[str] = None  # pin a specific version; else latest
    variables: dict[str, Any] = Field(default_factory=dict)
    messages: Optional[list[ChatMessage]] = None
    model: Optional[str] = None
    temperature: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    trace_id: str
    output: str
    model: str
    prompt_version: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    status: Status
