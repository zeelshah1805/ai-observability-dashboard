"""Demo data generator.

Writes synthetic but realistic trace rows directly to the store — no network,
no LLM calls, no sleeps — so a freshly deployed dashboard shows the full
cost/latency/error picture and the v1-vs-v2 comparison the moment it loads.

The distributions mirror the mock provider: the unconstrained v1 prompt rambles
(2-8 sentences, frequently violating the 2-sentence contract), while the
constrained v2 prompt complies ~90% of the time, so it ends up shorter, faster,
cheaper, and far less likely to produce bad output.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone

from . import db
from .config import get_settings
from .models import Status, Trace
from .pricing import compute_cost
from .prompt_registry import registry

_SAMPLE_TEXTS = [
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


def _make_output(snippet: str, n_sentences: int) -> str:
    parts = [f"Synthesized summary regarding {snippet}"]
    for i in range(1, n_sentences):
        parts.append(f"Additional synthesized detail number {i}")
    return ". ".join(parts) + "."


def _make_trace(
    *,
    version: str,
    constrained: bool,
    model: str,
    provider: str,
    now: datetime,
    window_minutes: int,
    store_prompts: bool,
    rng: random.Random,
) -> Trace:
    if constrained:
        n = 2 if rng.random() < 0.90 else max(1, 2 + rng.choice([-1, 1, 2]))
    else:
        n = rng.randint(2, 8)

    base = rng.uniform(0.08, 0.25)
    gen = n * rng.uniform(0.08, 0.22)
    latency_ms = int((base + gen) * (1.0 + 0.7 * 0.2) * 1000)

    prompt_tokens = rng.randint(38, 58)
    completion_tokens = 6 + n * rng.randint(6, 9)

    text = _SAMPLE_TEXTS[rng.randrange(len(_SAMPLE_TEXTS))]
    output = _make_output(text[:80], n)
    ts = now - timedelta(seconds=rng.uniform(0, window_minutes * 60))

    status = Status.OK
    error_type = None
    ttft_ms = int(latency_ms * rng.uniform(0.2, 0.5))

    roll = rng.random()
    if roll < 0.015:
        status, error_type = Status.TIMEOUT, "timeout"
    elif roll < 0.025:
        status, error_type = Status.RATE_LIMITED, "rate_limited"
    elif roll < 0.03:
        status, error_type = Status.ERROR, "LLMError"

    if status in (Status.TIMEOUT, Status.RATE_LIMITED, Status.ERROR):
        # A failed call produced no completion: no output, tokens, or cost.
        completion_tokens, ttft_ms, output = 0, None, None
        latency_ms = int(latency_ms * rng.uniform(0.2, 0.6))
    elif n != 2:
        status, error_type = Status.BAD_OUTPUT, f"expected_2_sentences_got_{n}"

    cost = compute_cost(model, prompt_tokens, completion_tokens)

    return Trace(
        id=str(uuid.uuid4()),
        timestamp=ts,
        endpoint="/chat",
        model=model,
        provider=provider,
        prompt_version=version,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        status=status,
        error_type=error_type,
        retries=0,
        input=(text if store_prompts else None),
        output=(output if store_prompts else None),
        metadata={"source": "seed_demo"},
    )


def seed_demo(
    n_per_version: int = 150, window_minutes: int = 45, clear: bool = False
) -> int:
    """Populate the store with demo traces. Returns the number written."""
    db.init_db()
    if clear:
        db.clear_traces()

    settings = get_settings()
    model = settings.llm_model
    provider = settings.llm_provider
    versions = registry.versions("summarize")
    if len(versions) < 2:
        raise RuntimeError("Need two 'summarize' versions registered to seed.")
    v1, v2 = versions[0], versions[1]

    rng = random.Random()
    now = datetime.now(timezone.utc)
    written = 0
    for version, constrained in ((v1, False), (v2, True)):
        for _ in range(n_per_version):
            db.insert_trace(
                _make_trace(
                    version=version,
                    constrained=constrained,
                    model=model,
                    provider=provider,
                    now=now,
                    window_minutes=window_minutes,
                    store_prompts=settings.store_prompts,
                    rng=rng,
                )
            )
            written += 1
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Seed the trace store with demo data.")
    ap.add_argument("--n", type=int, default=150, help="traces per prompt version")
    ap.add_argument("--clear", action="store_true", help="wipe existing traces first")
    args = ap.parse_args()
    total = seed_demo(n_per_version=args.n, clear=args.clear)
    print(f"Seeded {total} demo traces into {get_settings().db_path}")
