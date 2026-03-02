from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import StackConfig
from .intent_types import QueryIntent
from .vlm_analyzer import VLMAnalyzer


def should_verify(
    *,
    enabled: bool,
    query_intent: QueryIntent,
    confidence_score: float,
    abstain_threshold: float,
    verify_threshold: float,
) -> tuple[bool, str]:
    if not enabled:
        return False, "disabled"
    if not query_intent.has_constraints():
        return False, "no_constraints"

    score = max(0.0, min(1.0, float(confidence_score)))
    lo = max(0.0, min(1.0, float(abstain_threshold)))
    hi = max(0.0, min(1.0, float(verify_threshold)))
    if hi < lo:
        hi = lo

    if score < lo:
        return False, "below_abstain_threshold"
    if score >= hi:
        return False, "above_verify_threshold"
    return True, "medium_confidence"


def _default_verification() -> dict[str, Any]:
    return {
        "satisfies": False,
        "missing_constraints": [],
        "evidence": [],
        "attribute_answers": {},
    }


def verify_candidates(
    cfg: StackConfig,
    *,
    intent: QueryIntent,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    """Run low-confidence verification on top-k images.

    Returns map image_id -> verification payload.
    """
    subset = candidates[: max(1, top_k)]
    if not subset:
        return {}

    prompt = (
        "Return JSON only.\n"
        "Schema:\n"
        '{"satisfies":true,"missing_constraints":[],"evidence":[],"attribute_answers":{}}\n'
        "Evaluate whether image satisfies query constraints.\n"
        f"query: {intent.raw_query}\n"
        f"retrieval_terms: {intent.retrieval_terms}\n"
        f"relation_terms: {intent.relation_terms}\n"
        f"attribute_terms: {intent.attribute_terms}\n"
        f"presence_terms: {intent.presence_terms}\n"
    )

    out: dict[str, dict[str, Any]] = {}
    with VLMAnalyzer(cfg.vlm_model_name) as verifier:
        for row in subset:
            image_id = str(row.get("image_id", ""))
            image_path = str(row.get("file_path", ""))
            payload = _default_verification()
            try:
                text = verifier.generate_text(Path(image_path), prompt)
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(text[start : end + 1])
                    if isinstance(parsed, dict):
                        payload["satisfies"] = bool(parsed.get("satisfies", False))
                        payload["missing_constraints"] = list(parsed.get("missing_constraints", []))
                        payload["evidence"] = list(parsed.get("evidence", []))
                        attr = parsed.get("attribute_answers", {})
                        payload["attribute_answers"] = attr if isinstance(attr, dict) else {}
            except Exception:
                # keep deterministic fallback payload
                pass
            out[image_id] = payload
    return out
