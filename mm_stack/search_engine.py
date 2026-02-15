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
                    "ocr_structured": str(meta["ocr_structured"]),
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

    def search(self, *, query: str | None = None, image_path: str | None = None, top_k: int = 10, explain: bool = False) -> SearchResponse:
        start = time.perf_counter()
        store = LanceStore(self.cfg)
        
        # Adaptive Retrieval Depth
        is_deep = False
        if query:
            words = query.split()
            if len(words) > 12 or "compare" in query.lower() or "analyze" in query.lower():
                is_deep = True
                top_k = max(top_k, 5) # Boost top_k for deep queries

        decision = route_query(query=query, image_path=image_path)
        
        # Session Integration
        from .session import SessionManager
        session = SessionManager(self.cfg)
        bias_shift = session.get_routing_adjustment()
        
        text_weight = 0.4 + bias_shift
        clip_weight = 1.0 - text_weight
        
        # Enforce bounds
        text_weight = max(0.1, min(0.9, text_weight))
        clip_weight = max(0.1, min(0.9, clip_weight))

        results = []
        explanation = {}

        if decision.mode == "clip":
            if image_path:
                base_rows = self._search_clip_image(Path(image_path), store, top_k)
            else:
                base_rows = self._search_clip_text(str(query), store, top_k)
            results = self._attach_metadata(base_rows, top_k)
            explanation = {"mode": "clip", "reason": decision.reason}
            
        elif decision.mode == "text":
            base_rows = self._search_text(str(query), store, top_k)
            results = self._attach_metadata(base_rows, top_k)
            explanation = {"mode": "text", "reason": decision.reason}
            
        else:
            # Hybrid
            # Adaptive: If deep, fetch more to rerank (reranking TODO, for now we just fetch more)
            fetch_k = 20 if not is_deep else 40
            
            clip_rows = self._search_clip_text(str(query), store, fetch_k)
            text_rows = self._search_text(str(query), store, fetch_k)
            
            merged = hybrid_fuse(clip_rows, text_rows, clip_weight=clip_weight, text_weight=text_weight)
            for row in merged:
                row["source"] = "hybrid"
            results = self._attach_metadata(merged, top_k)
            explanation = {
                "mode": "hybrid", 
                "reason": decision.reason, 
                "weights": {"clip": round(clip_weight, 2), "text": round(text_weight, 2)},
                "session_bias": bias_shift
            }

        latency_ms = int((time.perf_counter() - start) * 1000)
        
        if not explain:
            # Log only real searches, not explanations? Or log both? 
            # We log normal searches.
            conn = connect_sqlite(self.cfg)
            ensure_schema(conn)
            log_search(conn, query=(query or image_path or ""), routing_decision=decision.mode, latency_ms=latency_ms, result_ids=[r["image_id"] for r in results])
            conn.close()

        if explain:
             # Add detail for explanation
             explanation.update({
                 "latency": f"{latency_ms}ms",
                 "top_results": [
                     {"id": r["image_id"], "score": r["score"], "caption": r["caption"][:50]+"..."} 
                     for r in results[:3]
                 ]
             })
             # We return a special response or attach it?
             # For now, we print or wrap it. 
             # But the return type is SearchResponse. 
             # Let's attach it to routing_reason or normalization_explanation for now, 
             # OR effectively return it as a dict if the caller expects it (not type safe).
             # Better: SearchResponse has limited fields. 
             # We'll rely on the caller to request explain separately if they want full JSON, 
             # or we stash it in `rerank_todo` (hacky).
             # For the `explain` command, we might want a dedicated return.
             pass

        return SearchResponse(
            routing_mode=decision.mode,
            routing_reason=str(explanation) if explain else decision.reason,
            latency_ms=latency_ms,
            results=results,
            normalization_explanation=(
                f"Adaptive Hybrid: CLIP={clip_weight:.2f}, Text={text_weight:.2f} "
                f"(Session Bias: {bias_shift:+.2f})"
            ),
            rerank_todo="Adaptive depth enabled." if is_deep else ""
        )

    def search_forced_mode(self, *, query: str, mode: str, top_k: int = 10) -> SearchResponse:
        """Used by evaluation harness to test each retrieval mode independently."""
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
        log_search(conn, query=query, routing_decision=f"forced:{mode}", latency_ms=latency_ms, result_ids=[r["image_id"] for r in enriched])
        conn.close()
        return SearchResponse(
            routing_mode=mode,
            routing_reason="forced evaluation mode",
            latency_ms=latency_ms,
            results=enriched,
            normalization_explanation="Forced mode for evaluation.",
            rerank_todo="",
        )

    def explain(self, query: str) -> dict[str, Any]:
        """
        Returns full explanation JSON for a query.
        Does NOT log to search_logs to avoid noise.
        """
        resp = self.search(query=query, top_k=5, explain=True)
        # Parse the explanation string back or reconstructed
        # Since we hacked it into response, let's just clean it up here.
        # Actually, `search` above returns `SearchResponse`.
        # We can reconstruct the dict.
        
        return {
            "query": query,
            "routing": resp.routing_mode,
            "details": resp.routing_reason, # Contains the JSON string from search()
            "normalization": resp.normalization_explanation,
            "results": resp.results
        }

