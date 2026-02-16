from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .cache import QueryEmbeddingCache
from .clip_embedder import OpenCLIPEmbedder
from .config import StackConfig
from .db import connect_sqlite, ensure_schema, get_images_by_ids, log_search
from .entity_memory import load_entity_memory_for_images
from .fusion import distance_to_similarity, hybrid_fuse
from .intent_reranker import rerank_with_intent
from .lancedb_store import LanceStore
from .preprocess import preprocess_image
from .query_normalization import combined_rank, normalize_query
from .query_planner import parse_query, rerank_with_query_intent
from .router import route_query
from .search_types import SearchResponse
from .text_embedder import TextEmbedder
from .verification import should_verify, verify_candidates


class MultimodalSearchEngine:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()
        self.cache = QueryEmbeddingCache(max_size=128)

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

    def _attach_metadata(self, image_rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        image_ids = [str(r["image_id"]) for r in image_rows]
        meta_map = get_images_by_ids(conn, image_ids)
        entity_map = load_entity_memory_for_images(conn, image_ids)
        conn.close()

        out: list[dict[str, Any]] = []
        for row in image_rows:
            image_id = str(row["image_id"])
            meta = meta_map.get(image_id)
            if meta is None:
                continue
            mem = entity_map.get(image_id, {})
            out.append(
                {
                    "image_id": image_id,
                    "file_path": str(meta["file_path"]),
                    "caption": str(meta["caption"]),
                    "summary": str(meta["summary"]),
                    "tags": self._parse_tags(str(meta["tags"])),
                    "ocr_structured": str(meta["ocr_structured"]),
                    "score": round(float(row["score"]), 6),
                    "clip_score": round(float(row.get("clip_score", 0.0) or 0.0), 6),
                    "text_score": round(float(row.get("text_score", 0.0) or 0.0), 6),
                    "source": str(row.get("source", "unknown")),
                    "entities": mem.get("entities", []),
                    "attributes": mem.get("attributes", {}),
                    "relation_evidence": mem.get("relations", []),
                    "mentions": mem.get("mentions", []),
                }
            )
            if len(out) >= max(1, top_k):
                break
        return out

    def _attach_video_metadata(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return results

        conn = connect_sqlite(self.cfg)
        id_map = {str(r["image_id"]): r for r in results}
        img_ids = list(id_map.keys())
        if not img_ids:
            conn.close()
            return results
        placeholders = ",".join("?" for _ in img_ids)
        sql = f"""
            SELECT s.embedding_id, s.video_id, s.start_time, s.end_time, v.file_path as video_path
            FROM video_segments s
            JOIN videos v ON s.video_id = v.id
            WHERE s.embedding_id IN ({placeholders})
        """
        try:
            rows = conn.execute(sql, img_ids).fetchall()
            for row in rows:
                img_id = str(row["embedding_id"])
                if img_id in id_map:
                    item = id_map[img_id]
                    item["video_id"] = row["video_id"]
                    item["start_time"] = row["start_time"]
                    item["end_time"] = row["end_time"]
                    item["video_path"] = row["video_path"]
        except Exception:
            pass
        conn.close()
        return results

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

    def _planner_adjusted_weights(self, intent) -> tuple[float, float]:
        clip_weight = 0.6
        text_weight = 0.4
        if intent.query_type_flags.attribute_heavy:
            text_weight = 0.6
            clip_weight = 0.4
        elif intent.query_type_flags.relation_heavy:
            clip_weight = 0.65
            text_weight = 0.35
        return clip_weight, text_weight

    def _apply_intent_rerank(self, query: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Any]:
        if not query or not rows:
            return rows, None
        intent = parse_query(query)

        # Stage 2 fuzzy scoring (kept from existing stack).
        nq = normalize_query(query)
        fuzzy_ranked = combined_rank(
            rows,
            nq,
            alpha=self.cfg.fuzzy_alpha,
            beta=self.cfg.fuzzy_beta,
            fuzzy_threshold=self.cfg.fuzzy_ratio_threshold,
            min_combined_score=self.cfg.fuzzy_min_combined_score,
        )

        # Backward compatibility signal.
        compat_ranked = rerank_with_query_intent(
            fuzzy_ranked,
            query,
            appearance_weight=self.cfg.intent_appearance_weight,
            activity_weight=self.cfg.intent_activity_weight,
            presence_weight=self.cfg.intent_presence_weight,
            missing_person_penalty=self.cfg.intent_missing_person_penalty,
            missing_clothing_penalty=self.cfg.intent_missing_clothing_penalty,
            semi_hard_enabled=self.cfg.intent_semi_hard_enabled,
        )

        # Stage 3-6 intent reranking.
        reranked = rerank_with_intent(
            compat_ranked,
            intent,
            retrieval_weight=self.cfg.intent_weight_retrieval,
            attribute_weight=self.cfg.intent_weight_attribute,
            relation_weight=self.cfg.intent_weight_relation,
            required_entity_penalty=self.cfg.intent_required_entity_penalty,
            activity_boost=self.cfg.intent_activity_boost,
            color_boost=self.cfg.intent_color_boost,
            pattern_boost=self.cfg.intent_pattern_boost,
            presence_required=self.cfg.intent_presence_required,
        )
        return reranked, intent

    def search(
        self,
        *,
        query: str | None = None,
        image_path: str | None = None,
        top_k: int = 10,
        explain: bool = False,
        enable_verification: bool = True,
    ) -> SearchResponse:
        start = time.perf_counter()
        store = LanceStore(self.cfg)
        timings: dict[str, int] = {}
        t0 = time.perf_counter()
        decision = route_query(query=query, image_path=image_path)
        timings["route_ms"] = int((time.perf_counter() - t0) * 1000)
        intent = parse_query(query or "") if query else None
        retrieval_query = query or ""
        if intent and intent.retrieval_terms:
            retrieval_query = " ".join(intent.retrieval_terms)

        results: list[dict[str, Any]] = []
        confidence_explanation = "No constraints detected."
        verification_payload: dict[str, Any] | None = None

        t_retrieval = time.perf_counter()
        if decision.mode == "clip":
            if image_path:
                base_rows = self._search_clip_image(Path(image_path), store, top_k)
            else:
                base_rows = self._search_clip_text(str(retrieval_query), store, max(20, top_k))
            results = self._attach_metadata(base_rows, max(20, top_k))
            if query:
                results, intent = self._apply_intent_rerank(str(query), results)
            results = self._attach_video_metadata(results)
        elif decision.mode == "text":
            base_rows = self._search_text(str(retrieval_query), store, max(20, top_k))
            results = self._attach_metadata(base_rows, max(20, top_k))
            if query:
                results, intent = self._apply_intent_rerank(str(query), results)
            results = self._attach_video_metadata(results)
        else:
            clip_weight, text_weight = self._planner_adjusted_weights(intent) if intent else (0.6, 0.4)
            clip_rows = self._search_clip_text(str(retrieval_query), store, 20)
            text_rows = self._search_text(str(retrieval_query), store, 20)
            merged = hybrid_fuse(clip_rows, text_rows, clip_weight=clip_weight, text_weight=text_weight)
            for row in merged:
                row["source"] = "hybrid"
            results = self._attach_metadata(merged, max(20, top_k))
            if query:
                results, intent = self._apply_intent_rerank(str(query), results)
            results = self._attach_video_metadata(results)
        timings["retrieval_ms"] = int((time.perf_counter() - t_retrieval) * 1000)

        # Stage 7 verification (low confidence + constrained queries only).
        t_verify = time.perf_counter()
        top_score = float(results[0].get("final_score", results[0].get("score", 0.0))) if results else 0.0
        if intent and should_verify(
            enabled=(self.cfg.verify_enabled and enable_verification),
            query_intent=intent,
            top_score=top_score,
            threshold=self.cfg.verify_low_conf_threshold,
        ):
            verifications = verify_candidates(
                self.cfg,
                intent=intent,
                candidates=results,
                top_k=self.cfg.verify_top_k,
            )
            for row in results:
                image_id = str(row.get("image_id", ""))
                payload = verifications.get(image_id)
                if not payload:
                    continue
                row["verification"] = payload
                if payload.get("satisfies"):
                    row["score"] = round(float(row.get("score", 0.0) or 0.0) * 1.20, 6)
                else:
                    row["score"] = round(float(row.get("score", 0.0) or 0.0) * 0.35, 6)
            results.sort(key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)
            verification_payload = verifications
            confidence_explanation = "Low-confidence verification executed on top candidates."
        elif intent and intent.has_constraints():
            confidence_explanation = (
                f"Constraint-aware reranking applied using retrieval={self.cfg.intent_weight_retrieval:.2f}, "
                f"attribute={self.cfg.intent_weight_attribute:.2f}, relation={self.cfg.intent_weight_relation:.2f}."
            )
        timings["verification_ms"] = int((time.perf_counter() - t_verify) * 1000)

        # trim after reranks/verification
        results = results[: max(1, top_k)]

        latency_ms = int((time.perf_counter() - start) * 1000)
        timings["total_ms"] = latency_ms
        if not explain:
            conn = connect_sqlite(self.cfg)
            ensure_schema(conn)
            log_search(
                conn,
                query=(query or image_path or ""),
                routing_decision=decision.mode,
                latency_ms=latency_ms,
                result_ids=[r["image_id"] for r in results],
            )
            conn.close()

        normalization_explanation = "Intent-aware hybrid ranking active."
        if intent:
            normalization_explanation = (
                f"retrieval_terms={intent.retrieval_terms or [query or '']} "
                f"attributes={intent.attribute_terms} relations={intent.relation_terms}"
            )
        return SearchResponse(
            routing_mode=decision.mode,
            routing_reason=decision.reason,
            latency_ms=latency_ms,
            results=results,
            normalization_explanation=normalization_explanation,
            rerank_todo="" if not explain else "Explain mode does not persist logs.",
            query_intent=intent.to_dict() if intent else None,
            confidence_explanation=confidence_explanation,
            verification=verification_payload,
            timings=timings,
        )

    def search_forced_mode(self, *, query: str, mode: str, top_k: int = 10) -> SearchResponse:
        mode = mode.strip().lower()
        if mode not in {"clip", "text", "hybrid"}:
            raise ValueError("mode must be one of: clip, text, hybrid")
        start = time.perf_counter()
        store = LanceStore(self.cfg)
        intent = parse_query(query)
        retrieval_query = " ".join(intent.retrieval_terms) if intent.retrieval_terms else query
        if mode == "clip":
            base_rows = self._search_clip_text(retrieval_query, store, top_k)
            enriched = self._attach_metadata(base_rows, top_k)
        elif mode == "text":
            base_rows = self._search_text(retrieval_query, store, top_k)
            enriched = self._attach_metadata(base_rows, top_k)
        else:
            clip_rows = self._search_clip_text(retrieval_query, store, 20)
            text_rows = self._search_text(retrieval_query, store, 20)
            merged = hybrid_fuse(clip_rows, text_rows, clip_weight=0.6, text_weight=0.4)
            for row in merged:
                row["source"] = "hybrid"
            enriched = self._attach_metadata(merged, top_k)

        enriched, intent = self._apply_intent_rerank(query, enriched)
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
            normalization_explanation="Forced mode for evaluation.",
            rerank_todo="",
            query_intent=intent.to_dict() if intent else None,
            confidence_explanation="Forced-mode evaluation with intent rerank.",
            verification=None,
            timings={"total_ms": latency_ms},
        )

    def explain(self, query: str) -> dict[str, Any]:
        resp = self.search(query=query, top_k=5, explain=True)
        return {
            "query": query,
            "routing": resp.routing_mode,
            "details": resp.routing_reason,
            "normalization": resp.normalization_explanation,
            "query_intent": resp.query_intent,
            "confidence_explanation": resp.confidence_explanation,
            "results": resp.results,
        }
