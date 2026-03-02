from __future__ import annotations

import re
from typing import Any

from .config import StackConfig

_STOPWORDS: set[str] = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "when",
    "where",
    "which",
    "show",
    "image",
    "images",
    "photo",
    "photos",
    "find",
    "does",
    "is",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "tell",
    "please",
    "can",
    "could",
    "would",
    "should",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for tok in re.findall(r"[a-z0-9_]+", (query or "").lower()):
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if tok not in terms:
            terms.append(tok)
    return terms


def _row_text(row: dict[str, Any]) -> str:
    tags = row.get("tags", [])
    tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags)
    return (
        f"{row.get('caption', '')} "
        f"{row.get('summary', '')} "
        f"{row.get('ocr_structured', '')} "
        f"{tags_text}"
    ).lower()


def _lexical_support_top3(query: str, rows: list[dict[str, Any]]) -> float:
    terms = _query_terms(query)
    if not terms or not rows:
        return 0.0
    subset = rows[:3]
    matched = 0
    for term in terms:
        if any(term in _row_text(row) for row in subset):
            matched += 1
    return _clamp01(matched / max(1, len(terms)))


def compute_confidence(
    query: str,
    rows: list[dict[str, Any]],
    *,
    rerank_applied: bool,
    pre_rerank_top1: float,
    cfg: StackConfig,
) -> dict[str, Any]:
    top1_score = _clamp01(float(rows[0].get("score", 0.0) or 0.0)) if rows else 0.0
    top2_score = _clamp01(float(rows[1].get("score", 0.0) or 0.0)) if len(rows) > 1 else 0.0
    top2_margin = _clamp01(top1_score - top2_score) if len(rows) > 1 else top1_score
    lexical_support = _lexical_support_top3(query, rows)
    rerank_signal = 0.0
    if rerank_applied:
        rerank_signal = _clamp01(top1_score - max(0.0, float(pre_rerank_top1)))

    w_top1 = max(0.0, float(cfg.search_confidence_w_top1))
    w_margin = max(0.0, float(cfg.search_confidence_w_margin))
    w_lexical = max(0.0, float(cfg.search_confidence_w_lexical))
    w_rerank = max(0.0, float(cfg.search_confidence_w_rerank))
    weight_sum = w_top1 + w_margin + w_lexical + w_rerank
    if weight_sum <= 0.0:
        w_top1, w_margin, w_lexical, w_rerank = 0.45, 0.25, 0.20, 0.10
    else:
        w_top1 /= weight_sum
        w_margin /= weight_sum
        w_lexical /= weight_sum
        w_rerank /= weight_sum

    confidence_score = _clamp01(
        (w_top1 * top1_score)
        + (w_margin * top2_margin)
        + (w_lexical * lexical_support)
        + (w_rerank * rerank_signal)
    )

    abstain_threshold = _clamp01(float(cfg.search_confidence_abstain_threshold))
    verify_threshold = _clamp01(float(cfg.search_confidence_verify_threshold))
    if verify_threshold < abstain_threshold:
        verify_threshold = abstain_threshold

    if confidence_score >= verify_threshold:
        band = "high"
    elif confidence_score >= abstain_threshold:
        band = "medium"
    else:
        band = "low"

    abstain_recommended = band == "low"
    if not rows:
        abstain_reason = "no_results"
    elif abstain_recommended:
        reasons: list[str] = []
        if lexical_support < 0.20:
            reasons.append("weak_lexical_support")
        if top1_score < 0.60:
            reasons.append("low_top1_score")
        if top2_margin < 0.08:
            reasons.append("weak_top2_margin")
        abstain_reason = ",".join(reasons) if reasons else "low_confidence"
    else:
        abstain_reason = ""

    return {
        "confidence_score": round(confidence_score, 6),
        "confidence_band": band,
        "lexical_support_top3": round(lexical_support, 6),
        "top1_score": round(top1_score, 6),
        "top2_margin": round(top2_margin, 6),
        "rerank_signal": round(rerank_signal, 6),
        "abstain_recommended": abstain_recommended,
        "abstain_reason": abstain_reason,
        "thresholds": {
            "abstain": round(abstain_threshold, 6),
            "verify": round(verify_threshold, 6),
        },
    }
