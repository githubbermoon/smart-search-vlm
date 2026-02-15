from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .cache import QueryEmbeddingCache
from .clip_embedder import OpenCLIPEmbedder
from .config import StackConfig
from .db import connect_sqlite, ensure_schema, get_images_by_ids, log_search
from .fusion import distance_to_similarity, hybrid_fuse
from .lancedb_store import LanceStore
from .preprocess import preprocess_image
from .router import route_query
from .search_types import SearchResponse
from .text_embedder import TextEmbedder


class MultimodalSearchEngine:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()
        self.cache = QueryEmbeddingCache(max_size=128)

    def _attach_metadata(self, image_rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        image_ids = [str(r["image_id"]) for r in image_rows]
        meta_map = get_images_by_ids(conn, image_ids)
        conn.close()

        out: list[dict[str, Any]] = []
        for row in image_rows:
            image_id = str(row["image_id"])
            meta = meta_map.get(image_id)
            if meta is None:
                continue
            out.append(
                {
                    "image_id": image_id,
                    "file_path": str(meta["file_path"]),
                    "caption": str(meta["caption"]),
                    "summary": str(meta["summary"]),
                    "tags": self._parse_tags(meta["tags"]),
                    "score": round(float(row["score"]), 6),
                    "source": str(row.get("source", "unknown")),
                }
            )
            if len(out) >= max(1, top_k):
                break
        return out

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        import json

        try:
            parsed = json.loads(raw or "[]")
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [x.strip() for x in (raw or "").split(",") if x.strip()]

    def _search_clip_text(self, query: str, store: LanceStore, top_k: int) -> list[dict[str, Any]]:
        cache_key = ("clip_text", self.cfg.clip_model_name, query)
        vec = self.cache.get(cache_key)
        if vec is None:
            with OpenCLIPEmbedder(self.cfg.clip_model_name) as clip:
                vec = clip.encode_texts([query])[0]
            self.cache.put(cache_key, vec)
        rows = store.search_clip(vec, top_k)
        return [
            {"image_id": str(r["image_id"]), "score": distance_to_similarity(r.get("_distance")), "source": "clip"}
            for r in rows
        ]

    def _search_clip_image(self, image_path: Path, store: LanceStore, top_k: int) -> list[dict[str, Any]]:
        prepared = preprocess_image(image_path, self.cfg)
        with OpenCLIPEmbedder(self.cfg.clip_model_name) as clip:
            vec = clip.encode_images([prepared.normalized_path])[0]
        rows = store.search_clip(vec, top_k)
        return [
            {"image_id": str(r["image_id"]), "score": distance_to_similarity(r.get("_distance")), "source": "clip"}
            for r in rows
        ]

    def _search_text(self, query: str, store: LanceStore, top_k: int) -> list[dict[str, Any]]:
        cache_key = ("text", self.cfg.text_model_name, query)
        vec = self.cache.get(cache_key)
        if vec is None:
            with TextEmbedder(self.cfg.text_model_name) as text_model:
                vec = text_model.encode([query], is_query=True)[0]
            self.cache.put(cache_key, vec)
        rows = store.search_text(vec, top_k)
        return [
            {"image_id": str(r["image_id"]), "score": distance_to_similarity(r.get("_distance")), "source": "text"}
            for r in rows
        ]

    def search(self, *, query: str | None = None, image_path: str | None = None, top_k: int = 10) -> SearchResponse:
        start = time.perf_counter()
        store = LanceStore(self.cfg)
        decision = route_query(query=query, image_path=image_path)

        if decision.mode == "clip":
            if image_path:
                base_rows = self._search_clip_image(Path(image_path), store, top_k)
            else:
                base_rows = self._search_clip_text(str(query), store, top_k)
            enriched = self._attach_metadata(base_rows, top_k)
        elif decision.mode == "text":
            base_rows = self._search_text(str(query), store, top_k)
            enriched = self._attach_metadata(base_rows, top_k)
        else:
            clip_rows = self._search_clip_text(str(query), store, 20)
            text_rows = self._search_text(str(query), store, 20)
            merged = hybrid_fuse(clip_rows, text_rows, clip_weight=0.6, text_weight=0.4)
            for row in merged:
                row["source"] = "hybrid"
            enriched = self._attach_metadata(merged, top_k)

        latency_ms = int((time.perf_counter() - start) * 1000)
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        log_search(conn, query=(query or image_path or ""), routing_decision=decision.mode, latency_ms=latency_ms, result_ids=[r["image_id"] for r in enriched])
        conn.close()

        return SearchResponse(
            routing_mode=decision.mode,
            routing_reason=decision.reason,
            latency_ms=latency_ms,
            results=enriched,
            normalization_explanation=(
                "CLIP and text indexes produce different score scales; each index is normalized by its own max score "
                "before weighted fusion to prevent one index from dominating purely due to scale."
            ),
            rerank_todo="TODO: add cross-modal VLM reranking hook for top-N results.",
        )

    def search_forced_mode(self, *, query: str, mode: str, top_k: int = 10) -> SearchResponse:
        mode = mode.strip().lower()
        if mode not in {"clip", "text", "hybrid"}:
            raise ValueError("mode must be one of: clip, text, hybrid")
        start = time.perf_counter()
        store = LanceStore(self.cfg)
        if mode == "clip":
            base_rows = self._search_clip_text(query, store, top_k)
            enriched = self._attach_metadata(base_rows, top_k)
        elif mode == "text":
            base_rows = self._search_text(query, store, top_k)
            enriched = self._attach_metadata(base_rows, top_k)
        else:
            clip_rows = self._search_clip_text(query, store, 20)
            text_rows = self._search_text(query, store, 20)
            merged = hybrid_fuse(clip_rows, text_rows, clip_weight=0.6, text_weight=0.4)
            for row in merged:
                row["source"] = "hybrid"
            enriched = self._attach_metadata(merged, top_k)

        latency_ms = int((time.perf_counter() - start) * 1000)
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        log_search(
            conn,
            query=query,
            routing_decision=f"forced:{mode}",
            latency_ms=latency_ms,
            result_ids=[r["image_id"] for r in enriched],
        )
        conn.close()
        return SearchResponse(
            routing_mode=mode,
            routing_reason="forced evaluation mode",
            latency_ms=latency_ms,
            results=enriched,
            normalization_explanation=(
                "CLIP and text indexes produce different score scales; each index is normalized by its own max score "
                "before weighted fusion to prevent one index from dominating purely due to scale."
            ),
            rerank_todo="TODO: add cross-modal VLM reranking hook for top-N results.",
        )
