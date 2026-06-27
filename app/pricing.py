"""Model price table and cost computation.

Prices are USD per 1,000,000 tokens (in, out). Free-tier providers cost $0 in
reality, but we keep representative list prices so the dashboard can show a
*simulated* cost: it's attributed from token usage × a per-model rate, not
actually incurred during testing.
"""

from __future__ import annotations

# model id -> (input_price_per_1m, output_price_per_1m)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    # Groq (representative public list prices)
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
    # OpenRouter free tier
    "meta-llama/llama-3.1-8b-instruct:free": (0.0, 0.0),
    # Ollama (local, free)
    "llama3.1": (0.0, 0.0),
    "llama3.2": (0.0, 0.0),
    # Mock provider — give it a non-zero rate so the dashboard has cost data
    "mock-llama-3.1-8b": (0.05, 0.08),
    "mock-llama-3.3-70b": (0.59, 0.79),
}

# Fallback rate for unknown models (per 1M tokens). Labeled as an estimate.
_DEFAULT_PRICE: tuple[float, float] = (0.10, 0.10)


def price_for(model: str) -> tuple[float, float]:
    return MODEL_PRICES.get(model, _DEFAULT_PRICE)


def is_known_model(model: str) -> bool:
    return model in MODEL_PRICES


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a call given token usage."""
    in_price, out_price = price_for(model)
    cost = (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
    return round(cost, 8)
