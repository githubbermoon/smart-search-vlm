from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SearchResponse:
    routing_mode: str
    routing_reason: str
    latency_ms: int
    results: list[dict[str, Any]]
    normalization_explanation: str
    rerank_todo: str
    query_intent: dict[str, Any] | None = None
    policy_applied: dict[str, Any] | None = None
    confidence_explanation: str = ""
    verification: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    retrieval_debug: dict[str, Any] | None = None
    rerank_debug: dict[str, Any] | None = None
    confidence_debug: dict[str, Any] | None = None
    abstain_recommended: bool = False
    abstain_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
