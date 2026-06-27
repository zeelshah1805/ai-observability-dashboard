"""FastAPI application wiring (PLAN Phase 0/1/5).

Endpoints:
  GET  /health            liveness + provider/model info
  POST /chat              instrumented LLM call -> produces a complete Trace
  GET  /traces            recent traces (filterable) — feeds the dashboard/API
  GET  /traces/{id}       single trace drill-down
  GET  /prompts           registered prompt names + versions
  GET  /metrics           Prometheus exposition
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import db
from .config import get_settings
from .llm_client import LLMError
from .models import ChatRequest, ChatResponse, Trace
from .prompt_registry import registry
from .service import run_chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="AI Observability Dashboard",
    description="Instruments every LLM call for cost, latency, errors, and "
    "prompt-version performance.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "provider": s.llm_provider,
        "model": s.llm_model,
        "store_prompts": s.store_prompts,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return run_chat(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMError as exc:
        # The failure is already traced; surface it to the caller as 502.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/traces", response_model=list[Trace])
def traces(
    limit: int = Query(100, ge=1, le=1000),
    model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    status: Optional[str] = None,
) -> list[Trace]:
    return db.list_traces(
        limit=limit, model=model, prompt_version=prompt_version, status=status
    )


@app.get("/traces/{trace_id}", response_model=Trace)
def trace_detail(trace_id: str) -> Trace:
    trace = db.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace


@app.get("/prompts")
def prompts() -> dict:
    return {
        name: [
            {
                "version": registry.get(name, v).version,
                "description": registry.get(name, v).description,
            }
            for v in registry.versions(name)
        ]
        for name in registry.names()
    }


@app.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
