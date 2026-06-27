"""Output validators (bad-output detection).

An LLM call can return HTTP 200 and still be *wrong* for the use case. A
validator inspects the output and returns a reason string if it violates the
contract, or None if it's acceptable. The trace layer turns a reason into
status=bad_output — quality failures, not just status-code failures.

This is what makes the v1-vs-v2 comparison meaningful: the constrained v2
summarize prompt satisfies the 2-sentence contract far more often than v1.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def count_sentences(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT.split(text) if s.strip()])


def validate_summary(output: str) -> Optional[str]:
    """The `summarize` contract: a 2-sentence summary."""
    n = count_sentences(output)
    if n != 2:
        return f"expected_2_sentences_got_{n}"
    return None


# prompt_name -> validator
VALIDATORS: dict[str, Callable[[str], Optional[str]]] = {
    "summarize": validate_summary,
}


def validate(prompt_name: Optional[str], output: str) -> Optional[str]:
    if not prompt_name:
        return None
    fn = VALIDATORS.get(prompt_name)
    return fn(output) if fn else None
