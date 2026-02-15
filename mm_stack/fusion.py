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
