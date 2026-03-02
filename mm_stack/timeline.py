from __future__ import annotations

from typing import Any

from .config import StackConfig
from .db import connect_sqlite, ensure_schema


_VALID_GRANULARITIES = ("year", "month", "day")


def _bucket_expr(granularity: str) -> str:
    if granularity == "year":
        return "substr(created_at, 1, 4)"
    if granularity == "month":
        return "substr(created_at, 1, 7)"
    return "substr(created_at, 1, 10)"


class SemanticTimelineEngine:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()

    def timeline(
        self,
        *,
        granularity: str = "month",
        query: str | None = None,
        limit: int = 240,
    ) -> dict[str, Any]:
        gran = (granularity or "month").strip().lower()
        if gran not in _VALID_GRANULARITIES:
            raise ValueError(f"granularity must be one of {_VALID_GRANULARITIES}")

        safe_limit = max(1, min(int(limit), 2000))
        query_text = (query or "").strip().lower()
        bucket_sql = _bucket_expr(gran)

        where_parts = ["created_at != ''"]
        params: list[Any] = []
        if query_text:
            like = f"%{query_text}%"
            where_parts.append(
                "("
                "lower(file_path) LIKE ? OR "
                "lower(caption) LIKE ? OR "
                "lower(summary) LIKE ? OR "
                "lower(tags) LIKE ? OR "
                "lower(ocr_structured) LIKE ?"
                ")"
            )
            params.extend([like, like, like, like, like])

        where_clause = " AND ".join(where_parts)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)

        total_row = conn.execute(
            f"SELECT COUNT(*) AS n FROM images WHERE {where_clause}",
            params,
        ).fetchone()
        total_items = int(total_row["n"] if total_row else 0)

        sql = (
            f"SELECT {bucket_sql} AS bucket_key, "
            "COUNT(*) AS item_count, "
            "MIN(created_at) AS start_at, "
            "MAX(created_at) AS end_at, "
            "MIN(file_path) AS sample_path "
            f"FROM images WHERE {where_clause} "
            "GROUP BY bucket_key "
            "ORDER BY bucket_key DESC LIMIT ?"
        )
        rows = conn.execute(sql, params + [safe_limit]).fetchall()
        conn.close()

        buckets_desc = [
            {
                "key": str(r["bucket_key"] or ""),
                "item_count": int(r["item_count"] or 0),
                "start_at": str(r["start_at"] or ""),
                "end_at": str(r["end_at"] or ""),
                "sample_path": str(r["sample_path"] or ""),
            }
            for r in rows
            if str(r["bucket_key"] or "").strip()
        ]
        buckets = list(reversed(buckets_desc))

        max_count = max((b["item_count"] for b in buckets), default=0)
        avg_count = (sum(b["item_count"] for b in buckets) / len(buckets)) if buckets else 0.0
        peak_bucket = next((b for b in buckets if b["item_count"] == max_count), None)

        return {
            "granularity": gran,
            "query": query or "",
            "total_items": total_items,
            "bucket_count": len(buckets),
            "buckets": buckets,
            "stats": {
                "max_count": max_count,
                "avg_count": round(avg_count, 3),
                "peak_key": peak_bucket["key"] if peak_bucket else "",
                "peak_count": peak_bucket["item_count"] if peak_bucket else 0,
            },
        }
