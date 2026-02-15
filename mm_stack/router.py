from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import OCR_INTENT_KEYWORDS, VISUAL_INTENT_KEYWORDS


@dataclass(frozen=True)
class RoutingDecision:
    mode: str  # clip | text | hybrid
    reason: str


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def route_query(query: str | None = None, image_path: str | None = None) -> RoutingDecision:
    if image_path:
        return RoutingDecision(mode="clip", reason="image input -> clip index")

    text = (query or "").strip()
    if not text:
        raise ValueError("Query text is empty")

    if _contains_any(text, OCR_INTENT_KEYWORDS):
        return RoutingDecision(mode="text", reason="ocr intent keyword -> text index")

    if _contains_any(text, VISUAL_INTENT_KEYWORDS):
        return RoutingDecision(mode="clip", reason="visual keyword -> clip index")

    return RoutingDecision(mode="hybrid", reason="default fallback -> hybrid fusion")


def is_image_query(input_value: str) -> bool:
    p = Path(input_value)
    return p.exists() and p.is_file()
