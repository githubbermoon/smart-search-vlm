from __future__ import annotations

from typing import Any


def distance_to_similarity(distance: Any) -> float:
    try:
        d = float(distance)
    except Exception:
        return 0.0
    return max(-1.0, min(1.0, 1.0 - d))


def normalize_scores(rows: list[dict[str, Any]], score_key: str = "score") -> dict[str, float]:
    # Normalization is required because CLIP/text indexes can have different score scales.
    if not rows:
        return {}
    max_score = max(max(0.0, float(r.get(score_key, 0.0))) for r in rows)
    if max_score <= 0.0:
        max_score = 1.0
    out: dict[str, float] = {}
    for row in rows:
        image_id = str(row["image_id"])
        out[image_id] = max(0.0, float(row.get(score_key, 0.0))) / max_score
    return out


def hybrid_fuse(
    clip_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    clip_weight: float = 0.6,
    text_weight: float = 0.4,
) -> list[dict[str, Any]]:
    clip_norm = normalize_scores(clip_rows)
    text_norm = normalize_scores(text_rows)

    merged_ids = set(clip_norm) | set(text_norm)
    out: list[dict[str, Any]] = []
    for image_id in merged_ids:
        cs = clip_norm.get(image_id, 0.0)
        ts = text_norm.get(image_id, 0.0)
        out.append(
            {
                "image_id": image_id,
                "clip_score": cs,
                "text_score": ts,
                "score": (clip_weight * cs) + (text_weight * ts),
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def weighted_rrf_fuse(
    keyword_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    w_keyword: float = 0.62,
    w_semantic: float = 0.38,
) -> list[dict[str, Any]]:
    keyword_rank: dict[str, int] = {}
    keyword_score: dict[str, float] = {}
    for idx, row in enumerate(keyword_rows, start=1):
        image_id = str(row["image_id"])
        if image_id in keyword_rank:
            continue
        keyword_rank[image_id] = idx
        keyword_score[image_id] = float(row.get("score", 0.0) or 0.0)

    semantic_rank: dict[str, int] = {}
    semantic_score: dict[str, float] = {}
    for idx, row in enumerate(semantic_rows, start=1):
        image_id = str(row["image_id"])
        if image_id in semantic_rank:
            continue
        semantic_rank[image_id] = idx
        semantic_score[image_id] = float(row.get("score", 0.0) or 0.0)

    merged_ids = sorted(set(keyword_rank) | set(semantic_rank))
    out: list[dict[str, Any]] = []
    for image_id in merged_ids:
        kr = keyword_rank.get(image_id)
        sr = semantic_rank.get(image_id)
        keyword_rrf = (w_keyword / (rrf_k + kr)) if kr is not None else 0.0
        semantic_rrf = (w_semantic / (rrf_k + sr)) if sr is not None else 0.0
        fused = keyword_rrf + semantic_rrf
        out.append(
            {
                "image_id": image_id,
                "score": fused,
                "keyword_rank": kr,
                "semantic_rank": sr,
                "keyword_score": keyword_score.get(image_id, 0.0),
                "semantic_score": semantic_score.get(image_id, 0.0),
                "fusion_score": fused,
            }
        )

    # Stable deterministic order for tie cases.
    out.sort(
        key=lambda row: (
            -float(row.get("fusion_score", 0.0) or 0.0),
            min(
                int(row.get("keyword_rank") or 10**9),
                int(row.get("semantic_rank") or 10**9),
            ),
            str(row.get("image_id", "")),
        )
    )
    return out
