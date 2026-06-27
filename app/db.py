"""SQLite-backed trace store.

A single `traces` table is the backbone. Postgres is a drop-in
later swap; the access layer here is intentionally thin and SQL-portable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Optional

from .config import get_settings
from .models import Status, Trace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    model           TEXT NOT NULL,
    provider        TEXT NOT NULL,
    prompt_version  TEXT,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    ttft_ms         INTEGER,
    status          TEXT NOT NULL,
    error_type      TEXT,
    retries         INTEGER NOT NULL DEFAULT 0,
    input           TEXT,
    output          TEXT,
    metadata        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_model ON traces(model);
CREATE INDEX IF NOT EXISTS idx_traces_prompt_version ON traces(prompt_version);
CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
"""

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    db_path = settings.db_path
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema if it does not yet exist. Safe to call repeatedly."""
    with _lock, get_conn() as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def insert_trace(trace: Trace) -> None:
    row = (
        trace.id,
        trace.timestamp.isoformat(),
        trace.endpoint,
        trace.model,
        trace.provider,
        trace.prompt_version,
        trace.prompt_tokens,
        trace.completion_tokens,
        trace.cost_usd,
        trace.latency_ms,
        trace.ttft_ms,
        trace.status.value,
        trace.error_type,
        trace.retries,
        trace.input,
        trace.output,
        json.dumps(trace.metadata),
    )
    with _lock, get_conn() as conn:
        conn.execute(
            """
            INSERT INTO traces (
                id, timestamp, endpoint, model, provider, prompt_version,
                prompt_tokens, completion_tokens, cost_usd, latency_ms, ttft_ms,
                status, error_type, retries, input, output, metadata
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            row,
        )
        conn.commit()


def _row_to_trace(row: sqlite3.Row) -> Trace:
    return Trace(
        id=row["id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        endpoint=row["endpoint"],
        model=row["model"],
        provider=row["provider"],
        prompt_version=row["prompt_version"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        cost_usd=row["cost_usd"],
        latency_ms=row["latency_ms"],
        ttft_ms=row["ttft_ms"],
        status=Status(row["status"]),
        error_type=row["error_type"],
        retries=row["retries"],
        input=row["input"],
        output=row["output"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


def get_trace(trace_id: str) -> Optional[Trace]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
    return _row_to_trace(row) if row else None


def list_traces(
    limit: int = 100,
    model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    status: Optional[str] = None,
) -> list[Trace]:
    clauses: list[str] = []
    params: list[Any] = []
    if model:
        clauses.append("model = ?")
        params.append(model)
    if prompt_version:
        clauses.append("prompt_version = ?")
        params.append(prompt_version)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM traces {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
    return [_row_to_trace(r) for r in rows]
