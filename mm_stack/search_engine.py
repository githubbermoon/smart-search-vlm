from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .cache import QueryEmbeddingCache
from .clip_embedder import OpenCLIPEmbedder
from .config import StackConfig
from .cross_rerank import CrossEncoderReranker
from .db import connect_sqlite, ensure_schema, get_images_by_ids, log_search
from .entity_memory import load_entity_memory_for_images
from .fusion import distance_to_similarity, hybrid_fuse, weighted_rrf_fuse
from .intent_reranker import rerank_with_intent
from .lancedb_store import LanceStore
from .preprocess import preprocess_image
from .query_policy import AdaptiveQueryPolicy, build_query_policy
from .query_normalization import combined_rank, normalize_query
from .query_planner import parse_query, rerank_with_query_intent
from .retrieval_confidence import compute_confidence
from .router import route_query
from .search_types import SearchResponse
from .text_embed_client import encode_with_daemon
from .text_embedder import TextEmbedder
from .verification import should_verify, verify_candidates


class MultimodalSearchEngine:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()
        self.cache = QueryEmbeddingCache(max_size=128)

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        try:
            parsed = json.loads(raw or "[]")
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [x.strip() for x in (raw or "").split(",") if x.strip()]

    @staticmethod
    def _extract_json_string_field(blob: str, field: str) -> str:
        pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(pattern, blob or "", flags=re.DOTALL)
        if not match:
            return ""
        raw = match.group(1)
        try:
            return str(json.loads(f"\"{raw}\"")).strip()
        except Exception:
            return raw.strip()

    def _repair_legacy_metadata_text(self, caption: str, summary: str) -> tuple[str, str]:
        cap = (caption or "").strip()
        summ = (summary or "").strip()
        combined = f"{cap}\n{summ}".strip()
        if not combined:
            return "", ""

        if self._looks_malformed_text(cap):
            recovered_cap = self._extract_json_string_field(combined, "caption")
            if recovered_cap:
                cap = recovered_cap
        if self._looks_malformed_text(summ) or summ.startswith('"caption":'):
            recovered_summ = self._extract_json_string_field(combined, "summary")
            if recovered_summ:
                summ = recovered_summ
            elif cap:
                summ = cap
        if not cap and summ:
            cap = summ
        if not summ and cap:
            summ = cap
        return cap, summ

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
            caption, summary = self._repair_legacy_metadata_text(
                str(meta["caption"]),
                str(meta["summary"]),
            )
            out.append(
                {
                    "image_id": image_id,
                    "file_path": str(meta["file_path"]),
                    "caption": caption,
                    "summary": summary,
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

    @staticmethod
    def _build_fts_match_query(query: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
        uniq: list[str] = []
        for token in tokens:
            if token and token not in uniq:
                uniq.append(token)
        if not uniq:
            return ""
        parts: list[str] = []
        for token in uniq:
            # For very short terms, wildcard prefix is too broad:
            # "car*" matches "carrier". Keep exact token matching there.
            # Porter stemming still handles plural forms for exact terms.
            if len(token) <= 3:
                parts.append(token)
            else:
                parts.append(f"{token}*")
        return " ".join(parts)

    def _search_keyword(self, query: str, top_k: int) -> list[dict[str, Any]]:
        match_query = self._build_fts_match_query(query)
        if not match_query:
            return []
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        sql = """
            SELECT f.image_id AS image_id, bm25(images_fts) AS bm25_score
            FROM images_fts AS f
            LEFT JOIN images i ON i.id = f.image_id
            WHERE images_fts MATCH ?
            ORDER BY bm25_score ASC, i.updated_at DESC
            LIMIT ?
        """
        rows = conn.execute(sql, (match_query, max(1, int(top_k)))).fetchall()
        conn.close()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                bm25_score = float(row["bm25_score"])
            except Exception:
                bm25_score = 0.0
            lexical_score = 1.0 / (1.0 + max(0.0, bm25_score))
            out.append(
                {
                    "image_id": str(row["image_id"]),
                    "score": round(lexical_score, 6),
                    "source": "keyword",
                    "bm25_score": round(bm25_score, 6),
                }
            )
        return out

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
            vec = encode_with_daemon(self.cfg, text=query, is_query=True)
            if vec is None:
                with TextEmbedder(self.cfg.text_model_name) as text_model:
                    vec = text_model.encode([query], is_query=True)[0]
            self.cache.put(cache_key, vec)
        rows = store.search_text(vec, top_k)
        return [
            {"image_id": str(r["image_id"]), "score": distance_to_similarity(r.get("_distance")), "source": "text"}
            for r in rows
        ]

    @staticmethod
    def _looks_malformed_text(value: str) -> bool:
        text = (value or "").strip()
        if not text:
            return True
        if text in {"{", "}", "[", "]", "\""}:
            return True
        if text.startswith("\"caption\":") or text.startswith("{\"caption\""):
            return True
        return False

    def _apply_metadata_quality_penalty(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        for row in rows:
            penalty = 1.0
            if self._looks_malformed_text(str(row.get("caption", ""))):
                penalty *= 0.55
            if self._looks_malformed_text(str(row.get("summary", ""))):
                penalty *= 0.70
            if penalty < 1.0:
                row["score"] = round(float(row.get("score", 0.0) or 0.0) * penalty, 6)
                row["metadata_quality_penalty"] = round(penalty, 3)
        rows.sort(key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)
        return rows

    def _apply_intent_rerank(
        self,
        query: str,
        rows: list[dict[str, Any]],
        policy: AdaptiveQueryPolicy,
    ) -> tuple[list[dict[str, Any]], Any]:
        if not query or not rows:
            return rows, None
        intent = parse_query(query)

        # Stage 2 fuzzy scoring (kept from existing stack).
        nq = normalize_query(query)
        fuzzy_alpha = self.cfg.fuzzy_alpha
        fuzzy_beta = self.cfg.fuzzy_beta
        if policy.lexical_mode == "boost":
            fuzzy_alpha = max(fuzzy_alpha, 0.90)
            fuzzy_beta = min(fuzzy_beta, 0.10)
        if policy.query_type == "generic":
            fuzzy_alpha = max(fuzzy_alpha, 0.93)
            fuzzy_beta = min(fuzzy_beta, 0.07)
        elif policy.query_type == "attribute":
            fuzzy_alpha = max(fuzzy_alpha, 0.88)
            fuzzy_beta = min(fuzzy_beta, 0.12)
        fuzzy_ranked = combined_rank(
            rows,
            nq,
            alpha=fuzzy_alpha,
            beta=fuzzy_beta,
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
            retrieval_weight=(
                max(self.cfg.intent_weight_retrieval, 0.72)
                if policy.query_type == "generic"
                else (
                    min(self.cfg.intent_weight_retrieval, 0.58)
                    if policy.query_type == "attribute"
                    else self.cfg.intent_weight_retrieval
                )
            ),
            attribute_weight=self.cfg.intent_weight_attribute,
            relation_weight=self.cfg.intent_weight_relation,
            required_entity_penalty=policy.required_entity_penalty,
            activity_boost=self.cfg.intent_activity_boost,
            color_boost=self.cfg.intent_color_boost,
            pattern_boost=self.cfg.intent_pattern_boost,
            presence_required=policy.presence_required,
        )
        return reranked, intent

    @staticmethod
    def _query_token_count(query: str) -> int:
        return len(re.findall(r"[a-z0-9_]+", (query or "").lower()))

    def _default_cross_rerank_debug(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.cfg.search_cross_rerank_enabled),
            "applied": False,
            "model": str(self.cfg.search_cross_rerank_model),
            "rerank_k": 0,
            "weight": round(max(0.0, min(1.0, float(self.cfg.search_cross_rerank_weight))), 6),
            "batch_size": max(1, int(self.cfg.search_cross_rerank_batch_size)),
            "latency_ms": 0,
            "reason": "not_attempted",
        }

    def _compute_cross_rerank_k(
        self,
        *,
        top_k: int,
        query: str,
        keyword_hits: int,
        row_count: int,
    ) -> int:
        base_k = max(2 * max(1, int(top_k)), int(self.cfg.search_cross_rerank_k_min))
        rerank_k = min(base_k, int(self.cfg.search_cross_rerank_k_max))
        if self._query_token_count(query) > int(self.cfg.search_cross_rerank_long_query_token_cutoff):
            rerank_k = max(2, int(round(rerank_k * 0.70)))
        if keyword_hits > int(self.cfg.search_cross_rerank_high_keyword_hits):
            rerank_k = min(rerank_k, int(self.cfg.search_cross_rerank_high_keyword_cap))
        return max(0, min(max(0, int(row_count)), max(2, int(rerank_k))))

    def _maybe_apply_cross_rerank(
        self,
        *,
        query: str,
        rows: list[dict[str, Any]],
        top_k: int,
        keyword_hits: int,
        allow_cross_rerank: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
        debug = self._default_cross_rerank_debug()
        pre_top1 = float(rows[0].get("score", 0.0) or 0.0) if rows else 0.0
        if not allow_cross_rerank:
            debug["reason"] = "disabled_for_route"
            return rows, debug, pre_top1
        if not self.cfg.search_cross_rerank_enabled:
            debug["reason"] = "disabled"
            return rows, debug, pre_top1
        if not query:
            debug["reason"] = "empty_query"
            return rows, debug, pre_top1
        if len(rows) < 2:
            debug["reason"] = "insufficient_candidates"
            return rows, debug, pre_top1

        rerank_k = self._compute_cross_rerank_k(
            top_k=top_k,
            query=query,
            keyword_hits=keyword_hits,
            row_count=len(rows),
        )
        debug["rerank_k"] = rerank_k
        if rerank_k < 2:
            debug["reason"] = "rerank_k_too_small"
            return rows, debug, pre_top1

        t0 = time.perf_counter()
        reranker = CrossEncoderReranker(str(self.cfg.search_cross_rerank_model))
        rerank_result = reranker.rerank_rows(
            query,
            rows,
            rerank_k=rerank_k,
            weight=float(self.cfg.search_cross_rerank_weight),
            batch_size=max(1, int(self.cfg.search_cross_rerank_batch_size)),
        )
        debug.update(rerank_result.debug)
        debug["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return rerank_result.rows, debug, pre_top1

    def search(
        self,
        *,
        query: str | None = None,
        image_path: str | None = None,
        top_k: int = 10,
        mode: str | None = None,
        auto_strategy: str | None = None,
        semantic_fallback_threshold: int | None = None,
        explain: bool = False,
        enable_verification: bool = False,
    ) -> SearchResponse:
        start = time.perf_counter()
        selected_mode = (mode or self.cfg.search_default_mode or "auto").strip().lower()
        if selected_mode not in {"auto", "keyword", "semantic"}:
            raise ValueError("mode must be one of: auto, keyword, semantic")
        selected_auto_strategy = (
            auto_strategy
            or self.cfg.search_auto_strategy_default
            or "hybrid"
        ).strip().lower()
        if selected_auto_strategy not in {"legacy", "hybrid"}:
            raise ValueError("auto_strategy must be one of: legacy, hybrid")
        if not query and not image_path:
            raise ValueError("Provide query text or image_path")
        if image_path and selected_mode == "keyword":
            raise ValueError("keyword mode does not support --image-path")

        # Ensure schema for FTS/triggers before retrieval starts.
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        conn.close()

        store = LanceStore(self.cfg)
        timings: dict[str, int] = {}
        route_reason = ""
        route_mode = ""
        t0 = time.perf_counter()
        intent = parse_query(query or "") if query else None
        policy = build_query_policy(intent, self.cfg, requested_top_k=max(1, top_k))
        retrieval_query = query or ""
        if intent and intent.retrieval_terms:
            retrieval_query = " ".join(intent.retrieval_terms)
        retrieval_limit = policy.candidate_top_k(max(1, top_k), minimum=max(20, top_k))
        fallback_threshold = (
            self.cfg.search_semantic_fallback_threshold
            if semantic_fallback_threshold is None
            else max(0, int(semantic_fallback_threshold))
        )
        candidate_k = max(1, max(4 * max(1, int(top_k)), int(self.cfg.search_hybrid_candidate_k_min)))
        candidate_k = min(candidate_k, int(self.cfg.search_hybrid_candidate_k_max))
        keyword_hard_cutoff = max(1, int(self.cfg.search_keyword_hard_cutoff))
        rrf_k = max(1, int(self.cfg.search_hybrid_rrf_k))
        w_keyword = max(0.0, float(self.cfg.search_hybrid_weight_keyword))
        w_semantic = max(0.0, float(self.cfg.search_hybrid_weight_semantic))
        weight_sum = w_keyword + w_semantic
        if weight_sum <= 0.0:
            w_keyword, w_semantic = 0.62, 0.38
        else:
            w_keyword = w_keyword / weight_sum
            w_semantic = w_semantic / weight_sum
        timings["route_ms"] = int((time.perf_counter() - t0) * 1000)

        results: list[dict[str, Any]] = []
        confidence_explanation = f"Adaptive policy active: {policy.explanation}"
        verification_payload: dict[str, Any] | None = None
        retrieval_debug: dict[str, Any] | None = None
        rerank_debug: dict[str, Any] = self._default_cross_rerank_debug()
        confidence_debug: dict[str, Any] | None = None
        abstain_recommended = False
        abstain_reason = ""
        pre_rerank_top1 = 0.0

        t_retrieval = time.perf_counter()
        if image_path:
            route_mode = "clip"
            route_reason = "image input -> clip index"
            base_rows = self._search_clip_image(Path(image_path), store, top_k)
            results = self._attach_metadata(base_rows, retrieval_limit)
            results = self._apply_metadata_quality_penalty(results)
            if query:
                results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                    query=str(query),
                    rows=results,
                    top_k=max(1, top_k),
                    keyword_hits=0,
                    allow_cross_rerank=False,
                )
                results, intent = self._apply_intent_rerank(str(query), results, policy)
            results = self._attach_video_metadata(results)
        elif selected_mode == "keyword":
            route_mode = "keyword"
            route_reason = "fts5 keyword search"
            keyword_rows = self._search_keyword(str(query or ""), max(1, top_k))
            results = self._attach_metadata(keyword_rows, max(1, top_k))
            results = self._apply_metadata_quality_penalty(results)
            if query:
                results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                    query=str(query),
                    rows=results,
                    top_k=max(1, top_k),
                    keyword_hits=len(keyword_rows),
                )
                results, intent = self._apply_intent_rerank(str(query), results, policy)
            results = self._attach_video_metadata(results)
        elif selected_mode == "auto":
            keyword_rows = self._search_keyword(str(query or ""), candidate_k)
            keyword_hits = len(keyword_rows)
            semantic_rows: list[dict[str, Any]] = []
            hard_cutoff_applied = False

            if selected_auto_strategy == "legacy":
                if keyword_hits > fallback_threshold:
                    route_mode = "keyword"
                    route_reason = (
                        f"legacy auto: fts5 keyword hits={keyword_hits} "
                        f"> fallback_threshold={fallback_threshold}"
                    )
                    results = self._attach_metadata(keyword_rows, max(1, top_k))
                    results = self._apply_metadata_quality_penalty(results)
                    if query:
                        results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                            query=str(query),
                            rows=results,
                            top_k=max(1, top_k),
                            keyword_hits=keyword_hits,
                        )
                        results, intent = self._apply_intent_rerank(str(query), results, policy)
                    results = self._attach_video_metadata(results)
                else:
                    route_mode = "keyword+semantic_fallback"
                    route_reason = (
                        f"legacy auto: fts5 keyword hits={keyword_hits} "
                        f"<= fallback_threshold={fallback_threshold}; semantic text fallback"
                    )
                    semantic_rows = self._search_text(str(retrieval_query), store, retrieval_limit)
                    results = self._attach_metadata(semantic_rows, retrieval_limit)
                    results = self._apply_metadata_quality_penalty(results)
                    if query:
                        results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                            query=str(query),
                            rows=results,
                            top_k=max(1, top_k),
                            keyword_hits=keyword_hits,
                        )
                        results, intent = self._apply_intent_rerank(str(query), results, policy)
                    results = self._attach_video_metadata(results)
            else:
                if keyword_hits == 0:
                    route_mode = "keyword+semantic_fallback"
                    route_reason = "hybrid auto: no keyword hits; semantic text fallback"
                    semantic_rows = self._search_text(str(retrieval_query), store, candidate_k)
                    results = self._attach_metadata(semantic_rows, candidate_k)
                    results = self._apply_metadata_quality_penalty(results)
                    if query:
                        results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                            query=str(query),
                            rows=results,
                            top_k=max(1, top_k),
                            keyword_hits=keyword_hits,
                        )
                        results, intent = self._apply_intent_rerank(str(query), results, policy)
                    results = self._attach_video_metadata(results)
                elif keyword_hits > keyword_hard_cutoff:
                    hard_cutoff_applied = True
                    route_mode = "keyword"
                    route_reason = (
                        f"hybrid auto: keyword hits={keyword_hits} "
                        f"> hard_cutoff={keyword_hard_cutoff}; keyword-only fast path"
                    )
                    results = self._attach_metadata(keyword_rows, max(1, top_k))
                    results = self._apply_metadata_quality_penalty(results)
                    if query:
                        results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                            query=str(query),
                            rows=results,
                            top_k=max(1, top_k),
                            keyword_hits=keyword_hits,
                        )
                        results, intent = self._apply_intent_rerank(str(query), results, policy)
                    results = self._attach_video_metadata(results)
                else:
                    route_mode = "keyword+semantic_hybrid"
                    route_reason = (
                        f"hybrid auto: keyword hits={keyword_hits} in (0,{keyword_hard_cutoff}] "
                        f"-> weighted RRF fusion"
                    )
                    semantic_rows = self._search_text(str(retrieval_query), store, candidate_k)
                    merged = weighted_rrf_fuse(
                        keyword_rows,
                        semantic_rows,
                        rrf_k=rrf_k,
                        w_keyword=w_keyword,
                        w_semantic=w_semantic,
                    )
                    for row in merged:
                        row["source"] = "keyword+semantic_hybrid"
                    results = self._attach_metadata(merged, candidate_k)
                    results = self._apply_metadata_quality_penalty(results)
                    if query:
                        results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                            query=str(query),
                            rows=results,
                            top_k=max(1, top_k),
                            keyword_hits=keyword_hits,
                        )
                        results, intent = self._apply_intent_rerank(str(query), results, policy)
                    results = self._attach_video_metadata(results)

            retrieval_debug = {
                "auto_strategy": selected_auto_strategy,
                "keyword_hits": keyword_hits,
                "semantic_hits": len(semantic_rows),
                "candidate_k": candidate_k,
                "rrf_k": rrf_k,
                "weights": {
                    "keyword": round(w_keyword, 6),
                    "semantic": round(w_semantic, 6),
                },
                "hard_cutoff_applied": hard_cutoff_applied,
            }
        else:
            decision = route_query(query=query, image_path=image_path)
            route_mode = decision.mode
            route_reason = decision.reason
            if decision.mode == "clip":
                base_rows = self._search_clip_text(str(retrieval_query), store, retrieval_limit)
                results = self._attach_metadata(base_rows, retrieval_limit)
                results = self._apply_metadata_quality_penalty(results)
                if query:
                    results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                        query=str(query),
                        rows=results,
                        top_k=max(1, top_k),
                        keyword_hits=0,
                    )
                    results, intent = self._apply_intent_rerank(str(query), results, policy)
                results = self._attach_video_metadata(results)
            elif decision.mode == "text":
                base_rows = self._search_text(str(retrieval_query), store, retrieval_limit)
                results = self._attach_metadata(base_rows, retrieval_limit)
                results = self._apply_metadata_quality_penalty(results)
                if query:
                    results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                        query=str(query),
                        rows=results,
                        top_k=max(1, top_k),
                        keyword_hits=0,
                    )
                    results, intent = self._apply_intent_rerank(str(query), results, policy)
                results = self._attach_video_metadata(results)
            else:
                clip_weight, text_weight = policy.clip_weight, policy.text_weight
                clip_rows = self._search_clip_text(str(retrieval_query), store, retrieval_limit)
                text_rows = self._search_text(str(retrieval_query), store, retrieval_limit)
                merged = hybrid_fuse(clip_rows, text_rows, clip_weight=clip_weight, text_weight=text_weight)
                for row in merged:
                    row["source"] = "hybrid"
                results = self._attach_metadata(merged, retrieval_limit)
                results = self._apply_metadata_quality_penalty(results)
                if query:
                    results, rerank_debug, pre_rerank_top1 = self._maybe_apply_cross_rerank(
                        query=str(query),
                        rows=results,
                        top_k=max(1, top_k),
                        keyword_hits=0,
                    )
                    results, intent = self._apply_intent_rerank(str(query), results, policy)
                results = self._attach_video_metadata(results)
        timings["retrieval_ms"] = int((time.perf_counter() - t_retrieval) * 1000)

        if query:
            confidence_debug = compute_confidence(
                str(query),
                results,
                rerank_applied=bool(rerank_debug.get("applied", False)),
                pre_rerank_top1=pre_rerank_top1,
                cfg=self.cfg,
            )
            abstain_recommended = bool(confidence_debug.get("abstain_recommended", False))
            abstain_reason = str(confidence_debug.get("abstain_reason", ""))
            confidence_explanation = (
                f"Composite confidence={float(confidence_debug.get('confidence_score', 0.0)):.2f} "
                f"band={confidence_debug.get('confidence_band', 'low')} "
                f"lexical_support={float(confidence_debug.get('lexical_support_top3', 0.0)):.2f}."
            )
            if retrieval_debug is None:
                retrieval_debug = {}
            retrieval_debug["cross_rerank"] = rerank_debug
            retrieval_debug["confidence_score"] = confidence_debug.get("confidence_score", 0.0)
            retrieval_debug["confidence_band"] = confidence_debug.get("confidence_band", "low")
            retrieval_debug["abstain_recommended"] = abstain_recommended
            retrieval_debug["abstain_reason"] = abstain_reason

        # Stage 7 verification (medium-confidence + constrained queries only).
        t_verify = time.perf_counter()
        verify_allowed = False
        verify_reason = "no_intent"
        if intent:
            confidence_score = (
                float(confidence_debug.get("confidence_score", 0.0))
                if confidence_debug is not None
                else float(results[0].get("final_score", results[0].get("score", 0.0))) if results else 0.0
            )
            verify_allowed, verify_reason = should_verify(
                enabled=(self.cfg.verify_enabled and enable_verification),
                query_intent=intent,
                confidence_score=confidence_score,
                abstain_threshold=float(self.cfg.search_confidence_abstain_threshold),
                verify_threshold=float(self.cfg.search_confidence_verify_threshold),
            )

        if verify_allowed:
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
            confidence_explanation = "Medium-confidence verification executed on top candidates."
        elif intent and intent.has_constraints() and not confidence_debug:
            confidence_explanation = (
                f"Constraint-aware reranking with policy={policy.query_type}, "
                f"retrieval={self.cfg.intent_weight_retrieval:.2f}, "
                f"attribute={self.cfg.intent_weight_attribute:.2f}, relation={self.cfg.intent_weight_relation:.2f}."
            )
        if retrieval_debug is not None:
            retrieval_debug["verification_decision"] = {
                "enabled": bool(self.cfg.verify_enabled and enable_verification),
                "applied": bool(verify_allowed),
                "reason": verify_reason,
            }
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
                routing_decision=route_mode,
                latency_ms=latency_ms,
                result_ids=[r["image_id"] for r in results],
            )
            conn.close()

        normalization_explanation = f"Intent-aware ranking active ({policy.query_type})."
        if route_mode.startswith("keyword"):
            normalization_explanation = (
                f"keyword_mode={route_mode} auto_strategy={selected_auto_strategy} "
                f"fallback_threshold={fallback_threshold} candidate_k={candidate_k} "
                f"retrieval_terms={intent.retrieval_terms if intent else [query or '']}"
            )
        elif intent:
            normalization_explanation = (
                f"retrieval_terms={intent.retrieval_terms or [query or '']} "
                f"attributes={intent.attribute_terms} relations={intent.relation_terms} "
                f"policy={policy.query_type}"
            )
        return SearchResponse(
            routing_mode=route_mode,
            routing_reason=route_reason,
            latency_ms=latency_ms,
            results=results,
            normalization_explanation=normalization_explanation,
            rerank_todo="" if not explain else "Explain mode does not persist logs.",
            query_intent=intent.to_dict() if intent else None,
            policy_applied=policy.to_dict(),
            confidence_explanation=confidence_explanation,
            verification=verification_payload,
            timings=timings,
            retrieval_debug=retrieval_debug,
            rerank_debug=rerank_debug,
            confidence_debug=confidence_debug,
            abstain_recommended=abstain_recommended,
            abstain_reason=abstain_reason,
        )

    def search_forced_mode(self, *, query: str, mode: str, top_k: int = 10) -> SearchResponse:
        mode = mode.strip().lower()
        if mode not in {"clip", "text", "hybrid"}:
            raise ValueError("mode must be one of: clip, text, hybrid")
        start = time.perf_counter()
        store = LanceStore(self.cfg)
        intent = parse_query(query)
        policy = build_query_policy(intent, self.cfg, requested_top_k=max(1, top_k))
        retrieval_query = " ".join(intent.retrieval_terms) if intent.retrieval_terms else query
        if mode == "clip":
            base_rows = self._search_clip_text(retrieval_query, store, policy.candidate_top_k(top_k, minimum=top_k))
            enriched = self._attach_metadata(base_rows, top_k)
            enriched = self._apply_metadata_quality_penalty(enriched)
        elif mode == "text":
            base_rows = self._search_text(retrieval_query, store, policy.candidate_top_k(top_k, minimum=top_k))
            enriched = self._attach_metadata(base_rows, top_k)
            enriched = self._apply_metadata_quality_penalty(enriched)
        else:
            retrieval_limit = policy.candidate_top_k(top_k, minimum=20)
            clip_rows = self._search_clip_text(retrieval_query, store, retrieval_limit)
            text_rows = self._search_text(retrieval_query, store, retrieval_limit)
            merged = hybrid_fuse(
                clip_rows,
                text_rows,
                clip_weight=policy.clip_weight,
                text_weight=policy.text_weight,
            )
            for row in merged:
                row["source"] = "hybrid"
            enriched = self._attach_metadata(merged, top_k)
            enriched = self._apply_metadata_quality_penalty(enriched)

        enriched, intent = self._apply_intent_rerank(query, enriched, policy)
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
            policy_applied=policy.to_dict(),
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
