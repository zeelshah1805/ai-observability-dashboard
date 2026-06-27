"""Prompt-regression demo.

Runs the *same* summarization workload under prompt v1 (unconstrained baseline)
and v2 (constrained, improved), then prints a side-by-side table built from real
trace data: avg cost, p95 latency, bad-output rate. It's a quick way to compare
two prompt versions on actual telemetry rather than guessing.

One command, no separate server needed (uses an in-process client):

    python scripts/regression_demo.py --n 120
"""

from __future__ import annotations

import argparse
import random
import sys

# Force UTF-8 output so the table renders on Windows consoles (cp1252) too.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app import db
from app.models import ChatRequest, Status
from app.prompt_registry import registry
from app.service import run_chat

SAMPLE_TEXTS = [
    "The mitochondria is the powerhouse of the cell, producing ATP through "
    "oxidative phosphorylation across the inner membrane.",
    "Quarterly revenue grew 18% year over year, driven by strong cloud "
    "adoption and improved net revenue retention among enterprise accounts.",
    "The hiking trail climbs 1,200 meters over 8 kilometers, offering sweeping "
    "views of the glacial valley and the river winding far below.",
    "Transformers use self-attention to model long-range dependencies in "
    "sequences without recurrence, enabling efficient parallel training.",
    "Slowly browning the onions develops a deep, caramelized sweetness that "
    "forms the flavor base for the entire stew.",
]


def run_workload(version: str, n: int) -> list:
    """Fire n summarize requests pinned to a prompt version; return trace ids."""
    ids = []
    for i in range(n):
        req = ChatRequest(
            prompt_name="summarize",
            prompt_version=version,
            variables={"text": random.choice(SAMPLE_TEXTS)},
            metadata={"source": "regression_demo"},
        )
        try:
            resp = run_chat(req)
            ids.append(resp.trace_id)
        except Exception:
            # Hard failures are still traced; we read them back from the DB.
            pass
        if (i + 1) % 25 == 0:
            print(f"  {version}: {i + 1}/{n}")
    return ids


def pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize_version(version: str) -> dict:
    traces = db.list_traces(limit=100000, prompt_version=version)
    traces = [t for t in traces if t.metadata.get("source") == "regression_demo"]
    n = len(traces)
    bad = [t for t in traces if t.status == Status.BAD_OUTPUT]
    errs = [t for t in traces if t.status not in (Status.OK, Status.BAD_OUTPUT)]
    # Latency is a property of the request regardless of output quality, so
    # measure it over every completed generation (ok + bad_output), not just
    # the ones that passed validation — otherwise v1's slow rambling outputs
    # (marked bad) would be excluded and hide the regression.
    completed = [t for t in traces if t.status in (Status.OK, Status.BAD_OUTPUT)]
    return {
        "version": version,
        "requests": n,
        "avg_cost": sum(t.cost_usd for t in traces) / n if n else 0.0,
        "p95_latency": pctl([t.latency_ms for t in completed], 0.95),
        "bad_rate": (len(bad) / n * 100) if n else 0.0,
        "err_rate": (len(errs) / n * 100) if n else 0.0,
    }


def fmt_delta(v1: float, v2: float) -> str:
    if v1 == 0:
        return "-"
    pct = (v2 - v1) / v1 * 100
    return f"{pct:+.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="v1-vs-v2 prompt regression demo.")
    ap.add_argument("--n", type=int, default=120, help="requests per version")
    args = ap.parse_args()

    db.init_db()
    versions = registry.versions("summarize")
    if len(versions) < 2:
        raise SystemExit("Need at least two 'summarize' versions registered.")
    v1, v2 = versions[0], versions[1]
    t1 = registry.get("summarize", v1)
    t2 = registry.get("summarize", v2)

    print(f"v1 = {v1}  ({t1.description})")
    print(f"v2 = {v2}  ({t2.description})")
    print(f"\nRunning {args.n} requests per version (provider=mock)...\n")

    run_workload(v1, args.n)
    run_workload(v2, args.n)

    a = summarize_version(v1)
    b = summarize_version(v2)

    print("\n" + "=" * 70)
    print("PROMPT REGRESSION COMPARISON (real trace data)")
    print("=" * 70)
    rows = [
        ("Metric", f"v1 {v1}", f"v2 {v2}", "Delta"),
        (
            "Avg cost / req",
            f"${a['avg_cost']:.6f}",
            f"${b['avg_cost']:.6f}",
            fmt_delta(a["avg_cost"], b["avg_cost"]),
        ),
        (
            "p95 latency",
            f"{a['p95_latency']:.0f} ms",
            f"{b['p95_latency']:.0f} ms",
            fmt_delta(a["p95_latency"], b["p95_latency"]),
        ),
        (
            "Bad-output rate",
            f"{a['bad_rate']:.1f}%",
            f"{b['bad_rate']:.1f}%",
            fmt_delta(a["bad_rate"], b["bad_rate"]),
        ),
        (
            "Error rate",
            f"{a['err_rate']:.1f}%",
            f"{b['err_rate']:.1f}%",
            fmt_delta(a["err_rate"], b["err_rate"]),
        ),
        ("Requests sampled", str(a["requests"]), str(b["requests"]), "-"),
    ]
    widths = [max(len(r[c]) for r in rows) for c in range(4)]
    for ri, row in enumerate(rows):
        line = " | ".join(row[c].ljust(widths[c]) for c in range(4))
        print(line)
        if ri == 0:
            print("-+-".join("-" * widths[c] for c in range(4)))
    print("=" * 70)
    print("\nView the same data live: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
