"""Load generator.

Replays a set of prompts against the running service with jitter so the
dashboard has realistic data. Also powers the v1-vs-v2 regression demo:
run the same workload under two prompt versions and compare the telemetry.

Usage:
    python scripts/load_gen.py --n 200
    python scripts/load_gen.py --n 200 --prompt summarize --version v-xxxxxxxx
"""

from __future__ import annotations

import argparse
import random
import time

import httpx

SAMPLE_TEXTS = [
    "The mitochondria is the powerhouse of the cell, producing ATP through "
    "oxidative phosphorylation.",
    "Quarterly revenue grew 18% YoY driven by strong cloud adoption and "
    "improved net retention.",
    "The hiking trail climbs 1,200 meters over 8 kilometers with sweeping "
    "views of the valley below.",
    "Transformers use self-attention to model long-range dependencies in "
    "sequences without recurrence.",
    "The recipe calls for browning the onions slowly to develop a deep, "
    "caramelized sweetness.",
]

SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Why is the sky blue?",
    "What causes inflation?",
    "How do vaccines work?",
]


def fire(
    base_url: str,
    prompt: str,
    version: str | None,
    n: int,
    rps: float,
) -> None:
    ok = err = 0
    interval = 1.0 / rps if rps > 0 else 0.0
    with httpx.Client(timeout=60.0) as client:
        for i in range(n):
            if prompt == "qa":
                payload = {
                    "prompt_name": "qa",
                    "variables": {"question": random.choice(SAMPLE_QUESTIONS)},
                    "metadata": {"source": "load_gen"},
                }
            else:
                payload = {
                    "prompt_name": prompt,
                    "variables": {"text": random.choice(SAMPLE_TEXTS)},
                    "metadata": {"source": "load_gen"},
                }
            if version:
                payload["prompt_version"] = version
            try:
                r = client.post(f"{base_url}/chat", json=payload)
                if r.status_code == 200:
                    ok += 1
                else:
                    err += 1
            except httpx.HTTPError:
                err += 1
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{n}  ok={ok} err={err}")
            if interval:
                time.sleep(interval * random.uniform(0.5, 1.5))
    print(f"Done. ok={ok} err={err} (errors are expected — simulated failures)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay prompts to populate traces.")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--prompt", default="summarize", choices=["summarize", "qa"])
    ap.add_argument("--version", default=None, help="pin a prompt version hash")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rps", type=float, default=10.0, help="target requests/sec")
    args = ap.parse_args()

    print(
        f"Firing {args.n} requests at {args.base_url} "
        f"(prompt={args.prompt}, version={args.version or 'latest'})"
    )
    fire(args.base_url, args.prompt, args.version, args.n, args.rps)


if __name__ == "__main__":
    main()
