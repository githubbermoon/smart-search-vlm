from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(value: float) -> float:
    # Guard overflow for very large magnitude logits.
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class CrossRerankResult:
    rows: list[dict[str, Any]]
    debug: dict[str, Any]


class CrossEncoderReranker:
    _model_cache: dict[str, Any] = {}

    def __init__(self, model_name: str):
        self.model_name = (model_name or "").strip()

    @staticmethod
    def _candidate_text(row: dict[str, Any]) -> str:
        tags = row.get("tags", [])
        tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags)
        return (
            f"{row.get('caption', '')}\n"
            f"{row.get('summary', '')}\n"
            f"{row.get('ocr_structured', '')}\n"
            f"{tags_text}"
        ).strip()

    def _ensure_model(self) -> Any:
        if not self.model_name:
            raise RuntimeError("cross-rerank model is empty")
        cached = self._model_cache.get(self.model_name)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError("sentence-transformers CrossEncoder is unavailable") from exc
        model = CrossEncoder(self.model_name)
        self._model_cache[self.model_name] = model
        return model

    def score_pairs(self, query: str, candidate_texts: list[str], *, batch_size: int = 16) -> list[float]:
        if not query or not candidate_texts:
            return []
        model = self._ensure_model()
        pairs = [(query, text) for text in candidate_texts]
        raw_scores = model.predict(
            pairs,
            batch_size=max(1, int(batch_size)),
            show_progress_bar=False,
        )
        # `predict` may return ndarray/list and occasionally nested values.
        out: list[float] = []
        for raw in raw_scores:
            if isinstance(raw, (list, tuple)) and raw:
                value = float(raw[0])
            else:
                value = float(raw)
            out.append(_clamp01(_sigmoid(value)))
        return out

    def rerank_rows(
        self,
        query: str,
        rows: list[dict[str, Any]],
        *,
        rerank_k: int,
        weight: float,
        batch_size: int = 16,
    ) -> CrossRerankResult:
        debug: dict[str, Any] = {
            "enabled": True,
            "applied": False,
            "model": self.model_name,
            "rerank_k": max(0, int(rerank_k)),
            "weight": round(max(0.0, min(1.0, float(weight))), 6),
            "batch_size": max(1, int(batch_size)),
            "reason": "",
        }
        if not query:
            debug["reason"] = "empty_query"
            return CrossRerankResult(rows=rows, debug=debug)
        if len(rows) < 2:
            debug["reason"] = "insufficient_candidates"
            return CrossRerankResult(rows=rows, debug=debug)
        if rerank_k < 2:
            debug["reason"] = "rerank_k_too_small"
            return CrossRerankResult(rows=rows, debug=debug)
        weight = max(0.0, min(1.0, float(weight)))
        if weight <= 0.0:
            debug["reason"] = "weight_zero"
            return CrossRerankResult(rows=rows, debug=debug)

        limited_k = min(len(rows), max(2, int(rerank_k)))
        head = [dict(r) for r in rows[:limited_k]]
        tail = [dict(r) for r in rows[limited_k:]]
        candidate_texts = [self._candidate_text(r) for r in head]

        try:
            cross_scores = self.score_pairs(
                query,
                candidate_texts,
                batch_size=max(1, int(batch_size)),
            )
        except Exception as exc:
            debug["reason"] = "model_error"
            debug["error"] = str(exc)
            return CrossRerankResult(rows=rows, debug=debug)

        if len(cross_scores) != len(head):
            debug["reason"] = "score_count_mismatch"
            return CrossRerankResult(rows=rows, debug=debug)

        combined: list[dict[str, Any]] = []
        for idx, row in enumerate(head):
            base = float(row.get("score", 0.0) or 0.0)
            cross = _clamp01(float(cross_scores[idx]))
            fused = ((1.0 - weight) * base) + (weight * cross)
            row["base_score_pre_cross"] = round(base, 6)
            row["cross_score"] = round(cross, 6)
            row["score_post_cross"] = round(fused, 6)
            row["score"] = round(fused, 6)
            combined.append(row)
        combined.extend(tail)

        scored = list(enumerate(combined))
        scored.sort(
            key=lambda x: (
                float(x[1].get("score", 0.0) or 0.0),
                float(x[1].get("cross_score", 0.0) or 0.0),
                float(x[1].get("base_score_pre_cross", x[1].get("score", 0.0)) or 0.0),
                -x[0],
            ),
            reverse=True,
        )
        reranked = [row for _, row in scored]
        debug["applied"] = True
        debug["rerank_k"] = limited_k
        debug["reason"] = "applied"
        return CrossRerankResult(rows=reranked, debug=debug)
