from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .store import Store


@dataclass(frozen=True)
class TeacherReview:
    action_probabilities: dict[str, float]
    preferred_action: str
    reasoning: str
    evidence_concerns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        labels = {"A", "B", "C"}
        if set(self.action_probabilities) != labels:
            raise ValueError("teacher probabilities must cover A, B, and C")
        if self.preferred_action not in labels:
            raise ValueError("teacher action must be A, B, or C")
        if any(value < 0 or value > 1 for value in self.action_probabilities.values()):
            raise ValueError("teacher probabilities must be between zero and one")
        if not math.isclose(sum(self.action_probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("teacher probabilities must sum to one")


class LLMTeacherProvider(Protocol):
    name: str

    def review(self, request: dict) -> TeacherReview: ...


def process_pending_reviews(
    store: Store, provider: LLMTeacherProvider, limit: int = 20,
) -> dict[str, int]:
    """Run queued reviews outside the trading workflow and record every result."""
    completed = failed = 0
    for job in store.pending_llm_reviews(limit):
        try:
            review = provider.review(job["request"])
            store.complete_llm_review(job["decision_id"], provider.name, asdict(review))
            completed += 1
        except Exception as error:  # noqa: BLE001 - worker must isolate all provider failures.
            store.fail_llm_review(job["decision_id"], provider.name, error)
            failed += 1
        store.commit()
    return {"completed": completed, "failed": failed}


def export_pending_reviews(store: Store, output_path: str | Path, limit: int = 1000) -> Path:
    """Export pending jobs for a separately deployed LLM review worker."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = store.pending_llm_reviews(limit)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output
