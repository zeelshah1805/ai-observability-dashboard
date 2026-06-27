"""In-process smoke test: boots the app, hits /chat, verifies a complete trace."""

from fastapi.testclient import TestClient

from app.main import app

with TestClient(app) as client:
    # health
    h = client.get("/health")
    print("HEALTH:", h.status_code, h.json())

    # prompts registered
    p = client.get("/prompts")
    print("PROMPTS:", p.json())

    # a chat call via a registered, versioned prompt
    r = client.post(
        "/chat",
        json={
            "prompt_name": "summarize",
            "variables": {"text": "FastAPI is a modern Python web framework."},
            "metadata": {"source": "smoke_test"},
        },
    )
    print("CHAT:", r.status_code)
    body = r.json()
    print("CHAT BODY:", body)

    trace_id = body["trace_id"]
    t = client.get(f"/traces/{trace_id}")
    trace = t.json()
    print("TRACE:", t.status_code)

    # Assert the trace is *complete*
    required = [
        "id", "timestamp", "endpoint", "model", "provider", "prompt_version",
        "prompt_tokens", "completion_tokens", "cost_usd", "latency_ms", "status",
    ]
    missing = [
        k for k in required
        if trace.get(k) is None and k not in ("prompt_version",)
    ]
    assert not missing, f"incomplete trace, missing: {missing}"
    assert trace["prompt_tokens"] > 0, "no prompt tokens captured"
    assert trace["completion_tokens"] > 0, "no completion tokens captured"
    assert trace["latency_ms"] >= 0, "no latency captured"
    assert trace["prompt_version"], "prompt not versioned"
    print(
        f"\nOK — complete trace: tokens={trace['prompt_tokens']}+"
        f"{trace['completion_tokens']} cost=${trace['cost_usd']:.8f} "
        f"latency={trace['latency_ms']}ms version={trace['prompt_version']} "
        f"status={trace['status']}"
    )

    # metrics exposed
    m = client.get("/metrics")
    print("METRICS:", m.status_code, "llm_requests_total" in m.text)
