from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import StackConfig
from .db import connect_sqlite, ensure_schema, get_images_by_ids
from .fusion import distance_to_similarity
from .lancedb_store import LanceStore


STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "this",
    "that",
    "image",
    "photo",
    "screenshot",
    "document",
    "page",
    "file",
}


@dataclass
class ContextTarget:
    image_id: str
    file_path: str
    caption: str
    tags: list[str]
    created_at: str


class ContextLensEngine:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()
        self.store = LanceStore(self.cfg)

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        try:
            value = json.loads(raw or "[]")
            if isinstance(value, list):
                return [str(x) for x in value if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in (raw or "").split(",") if x.strip()]

    @staticmethod
    def _sanitize_text(text: str) -> str:
        return (text or "").strip()

    def _resolve_target(self, *, image_id: str | None, file_path: str | None) -> ContextTarget:
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row = None
        if image_id:
            row = conn.execute("SELECT * FROM images WHERE id = ? LIMIT 1", (image_id,)).fetchone()
        elif file_path:
            row = conn.execute("SELECT * FROM images WHERE file_path = ? LIMIT 1", (file_path,)).fetchone()
        conn.close()
        if row is None:
            raise ValueError("Target image not found in index")
        return ContextTarget(
            image_id=str(row["id"]),
            file_path=str(row["file_path"]),
            caption=self._sanitize_text(str(row["caption"])),
            tags=self._parse_tags(str(row["tags"])),
            created_at=str(row["created_at"]),
        )

    def _similarity_neighbors(self, target: ContextTarget, top_k: int) -> list[dict[str, Any]]:
        vec = self.store.get_clip_vector_for_image(target.image_id)
        if not vec:
            return []
        rows = self.store.search_clip(vec, top_k + 1)
        out: list[dict[str, Any]] = []
        ids: list[str] = []
        for r in rows:
            iid = str(r.get("image_id", ""))
            if not iid or iid == target.image_id:
                continue
            ids.append(iid)
            out.append(
                {
                    "image_id": iid,
                    "score": round(distance_to_similarity(r.get("_distance")), 6),
                    "distance": float(r.get("_distance", 0.0) or 0.0),
                }
            )
            if len(out) >= top_k:
                break
        if not out:
            return []
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        meta = get_images_by_ids(conn, ids)
        conn.close()
        for item in out:
            row = meta.get(item["image_id"])
            if row is None:
                continue
            item["file_path"] = str(row["file_path"])
            item["caption"] = self._sanitize_text(str(row["caption"]))
            item["tags"] = self._parse_tags(str(row["tags"]))
            item["relation"] = "similarity"
        return [x for x in out if x.get("file_path")]

    def _cluster_neighbors(self, target: ContextTarget, top_k: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        cluster_row = conn.execute(
            """
            SELECT a.cluster_id, c.name, c.topic_label
            FROM cluster_assignments a
            JOIN clusters c ON a.cluster_id = c.id
            WHERE a.item_id = ?
            LIMIT 1
            """,
            (target.image_id,),
        ).fetchone()
        if cluster_row is None:
            conn.close()
            return None, []

        cluster_id = str(cluster_row["cluster_id"])
        neighbors = conn.execute(
            """
            SELECT a.item_id, a.distance
            FROM cluster_assignments a
            WHERE a.cluster_id = ? AND a.item_id != ?
            ORDER BY a.distance ASC
            LIMIT ?
            """,
            (cluster_id, target.image_id, max(1, top_k)),
        ).fetchall()
        ids = [str(r["item_id"]) for r in neighbors]
        meta = get_images_by_ids(conn, ids)
        conn.close()

        out: list[dict[str, Any]] = []
        for n in neighbors:
            iid = str(n["item_id"])
            row = meta.get(iid)
            if row is None:
                continue
            out.append(
                {
                    "image_id": iid,
                    "file_path": str(row["file_path"]),
                    "caption": self._sanitize_text(str(row["caption"])),
                    "tags": self._parse_tags(str(row["tags"])),
                    "distance": float(n["distance"] or 0.0),
                    "relation": "cluster",
                }
            )
        info = {
            "cluster_id": cluster_id,
            "cluster_name": str(cluster_row["name"]),
            "cluster_label": self._sanitize_text(str(cluster_row["topic_label"] or "")),
        }
        return info, out

    def _entity_terms(self, target: ContextTarget) -> list[str]:
        terms: list[str] = []
        for tag in target.tags:
            t = tag.strip().lower()
            if len(t) >= 3 and t not in STOPWORDS:
                terms.append(t)
        for word in re.findall(r"[A-Za-z0-9_]{3,}", target.caption.lower()):
            if word not in STOPWORDS:
                terms.append(word)
        uniq: list[str] = []
        for t in terms:
            if t not in uniq:
                uniq.append(t)
            if len(uniq) >= 6:
                break
        return uniq

    def _entity_neighbors(self, target: ContextTarget, top_k: int) -> tuple[list[str], list[dict[str, Any]]]:
        terms = self._entity_terms(target)
        if not terms:
            return [], []
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)

        likes = []
        params: list[Any] = [target.image_id]
        for t in terms:
            likes.append("(lower(tags) LIKE ? OR lower(caption) LIKE ? OR lower(summary) LIKE ?)")
            wildcard = f"%{t}%"
            params.extend([wildcard, wildcard, wildcard])
        sql = (
            "SELECT id, file_path, caption, tags, created_at "
            "FROM images WHERE id != ? AND (" + " OR ".join(likes) + ") "
            "ORDER BY created_at DESC LIMIT 60"
        )
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        scored: list[dict[str, Any]] = []
        for r in rows:
            hay = f"{str(r['caption']).lower()} {str(r['tags']).lower()}"
            matched = [t for t in terms if t in hay]
            if not matched:
                continue
            scored.append(
                {
                    "image_id": str(r["id"]),
                    "file_path": str(r["file_path"]),
                    "caption": self._sanitize_text(str(r["caption"])),
                    "tags": self._parse_tags(str(r["tags"])),
                    "entity_overlap": len(matched),
                    "matched_terms": matched[:4],
                    "relation": "entity",
                }
            )
        scored.sort(key=lambda x: (-int(x["entity_overlap"]), x["image_id"]))
        return terms, scored[:top_k]

    def _time_neighbors(self, target: ContextTarget, top_k: int) -> list[dict[str, Any]]:
        if not target.created_at:
            return []
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT id, file_path, caption, tags, created_at,
                   ABS(julianday(created_at) - julianday(?)) AS delta_days
            FROM images
            WHERE id != ? AND created_at != ''
            ORDER BY delta_days ASC
            LIMIT ?
            """,
            (target.created_at, target.image_id, max(1, top_k)),
        ).fetchall()
        conn.close()
        out: list[dict[str, Any]] = []
        for r in rows:
            delta = float(r["delta_days"] or 0.0)
            out.append(
                {
                    "image_id": str(r["id"]),
                    "file_path": str(r["file_path"]),
                    "caption": self._sanitize_text(str(r["caption"])),
                    "tags": self._parse_tags(str(r["tags"])),
                    "created_at": str(r["created_at"]),
                    "delta_days": round(delta, 4),
                    "delta_hours": round(delta * 24.0, 2),
                    "relation": "time",
                }
            )
        return out

    def context_lens(
        self,
        *,
        image_id: str | None = None,
        file_path: str | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        if not image_id and not file_path:
            raise ValueError("Provide image_id or file_path")
        k = max(1, int(top_k))
        target = self._resolve_target(image_id=image_id, file_path=file_path)

        similarity = self._similarity_neighbors(target, k)
        cluster_info, cluster_neighbors = self._cluster_neighbors(target, k)
        entity_terms, entity_neighbors = self._entity_neighbors(target, k)
        time_neighbors = self._time_neighbors(target, k)

        return {
            "target": {
                "image_id": target.image_id,
                "file_path": target.file_path,
                "caption": target.caption,
                "tags": target.tags,
                "created_at": target.created_at,
            },
            "rings": {
                "similarity": similarity,
                "cluster": cluster_neighbors,
                "entity": entity_neighbors,
                "time": time_neighbors,
            },
            "cluster_info": cluster_info,
            "entity_terms": entity_terms,
            "meta": {
                "top_k": k,
                "counts": {
                    "similarity": len(similarity),
                    "cluster": len(cluster_neighbors),
                    "entity": len(entity_neighbors),
                    "time": len(time_neighbors),
                },
            },
            "todo": "Future: weighted dynamic routing for context lens and VLM reranking.",
        }
