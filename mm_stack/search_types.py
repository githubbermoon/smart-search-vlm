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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
