"""Prompt versioning (PLAN §2 / Phase 2).

Every prompt template is registered under a name and assigned a deterministic
version hash derived from its text. Requests rendered through the registry are
tagged with that version so v1 vs v2 performance can be compared in the
dashboard. Multiple versions of the same name can coexist; the most recently
registered one is treated as "latest".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    template: str
    version: str  # short content hash, e.g. "v-a1b2c3d4"
    description: str = ""

    def render(self, variables: dict[str, Any]) -> str:
        """Render with simple str.format substitution."""
        return self.template.format(**variables)


def _hash_template(template: str) -> str:
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()[:8]
    return f"v-{digest}"


@dataclass
class PromptRegistry:
    # name -> {version -> template}
    _by_name: dict[str, dict[str, PromptTemplate]] = field(default_factory=dict)
    # name -> latest version
    _latest: dict[str, str] = field(default_factory=dict)

    def register(
        self, name: str, template: str, description: str = ""
    ) -> PromptTemplate:
        version = _hash_template(template)
        pt = PromptTemplate(
            name=name, template=template, version=version, description=description
        )
        self._by_name.setdefault(name, {})[version] = pt
        self._latest[name] = version
        return pt

    def get(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        if name not in self._by_name:
            raise KeyError(f"Unknown prompt: {name!r}")
        if version is None:
            version = self._latest[name]
        try:
            return self._by_name[name][version]
        except KeyError as exc:
            raise KeyError(
                f"Prompt {name!r} has no version {version!r}"
            ) from exc

    def versions(self, name: str) -> list[str]:
        return list(self._by_name.get(name, {}).keys())

    def names(self) -> list[str]:
        return list(self._by_name.keys())


# Module-level singleton registry with a couple of seeded prompts so the
# service and the v1/v2 regression demo (PLAN §6) work out of the box.
registry = PromptRegistry()

registry.register(
    name="summarize",
    template=(
        "Summarize the following text.\n\n{text}\n\nSummary:"
    ),
    description="v1 - terse, no constraints (baseline).",
)

registry.register(
    name="summarize",
    template=(
        "You are a precise summarization assistant. Summarize the text below "
        "in exactly 2 sentences. Do not add information that is not present.\n\n"
        "Text:\n{text}\n\nTwo-sentence summary:"
    ),
    description="v2 - constrained, lower variance (improved).",
)

registry.register(
    name="qa",
    template="Answer the question concisely.\n\nQuestion: {question}\nAnswer:",
    description="Generic single-turn QA prompt.",
)
