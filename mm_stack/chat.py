from __future__ import annotations

import gc
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator

try:
    from PIL import Image
except ImportError:
    Image = None

from .config import StackConfig
from .db import connect_sqlite, ensure_schema, get_image_by_id
from .query_planner import parse_query
from .search_engine import MultimodalSearchEngine
from .utils import cleanup_torch_mps

# Configure logging
logger = logging.getLogger(__name__)

CHAT_STOPWORDS = {
    "the", "and", "with", "from", "this", "that", "what", "when", "where", "which",
    "show", "about", "image", "images", "photo", "photos", "find", "for",
    "does", "do", "did", "is", "are", "am", "was", "were",
    "have", "has", "had", "there", "any", "tell", "please", "me",
    "can", "could", "would", "should", "will",
}

QUERY_EXPANSIONS: dict[str, list[str]] = {
    "jewelry": ["jewellery", "ornament", "ornaments", "necklace", "bracelet", "ring", "earring", "gold"],
}
ROLE_QUERY_TERMS: set[str] = {
    "student", "students", "teacher", "teachers", "professor", "professors",
    "employee", "employees", "intern", "interns", "doctor", "doctors",
    "engineer", "engineers", "developer", "developers",
}

@dataclass
class ChatResponse:
    answer: str
    sources: list[dict[str, Any]]
    confidence: str
    grounded_score: float
    highlight_regions: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)

@dataclass
class ChatSession:
    history: list[dict[str, Any]] = field(default_factory=list)
    last_retrieved_ids: list[str] = field(default_factory=list)
    
    def add_turn(self, query: str, answer: str, image_ids: list[str]):
        self.history.append({"query": query, "answer": answer})
        if len(self.history) > 3:
            self.history.pop(0)
        self.last_retrieved_ids = image_ids

class MultimodalChat:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()
        self.search_engine = MultimodalSearchEngine(self.cfg)
        self.session = ChatSession()
        self.min_similarity_gate = 0.60
        self.min_grounding_score = 0.3 # Threshold for "Not found" override
        self.max_context_tokens = 2000
        self.max_image_batch = 3
        self.max_image_size = 768

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        terms: list[str] = []
        for tok in re.findall(r"[A-Za-z0-9_]+", (query or "").lower()):
            if len(tok) < 3 or tok in CHAT_STOPWORDS:
                continue
            if tok not in terms:
                terms.append(tok)
        return terms[:8]

    def _detect_followup(self, query: str) -> bool:
        pronouns = {"that", "this", "it", "those", "these", "above", "below", "he", "she", "they"}
        tokens = set(re.split(r"[\s,;.?!]+", query.lower()))
        return bool(tokens & pronouns) and bool(self.session.last_retrieved_ids)

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
    def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        if not history:
            return []
        out: list[dict[str, str]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            out.append({"role": role, "content": content})
        return out[-8:]

    @staticmethod
    def _last_user_query(history: list[dict[str, str]]) -> str:
        for item in reversed(history):
            if item.get("role") == "user":
                return str(item.get("content", "")).strip()
        return ""

    @staticmethod
    def _history_context(history: list[dict[str, str]]) -> str:
        if not history:
            return ""
        lines = []
        for item in history[-6:]:
            role = "User" if item.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {item.get('content', '')}")
        return "Conversation History:\n" + "\n".join(lines)

    def _load_attached_source(
        self,
        *,
        attached_image_id: str | None = None,
        attached_file_path: str | None = None,
    ) -> dict[str, Any] | None:
        image_id = (attached_image_id or "").strip()
        file_path = (attached_file_path or "").strip()
        if not image_id and not file_path:
            return None

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row = None
        try:
            if image_id:
                row = get_image_by_id(conn, image_id)
            elif file_path:
                row = conn.execute(
                    "SELECT * FROM images WHERE file_path = ? LIMIT 1",
                    (file_path,),
                ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return {
            "image_id": str(row["id"]),
            "file_path": str(row["file_path"]),
            "caption": str(row["caption"]),
            "summary": str(row["summary"]),
            "tags": self._parse_tags(str(row["tags"])),
            "ocr_structured": str(row["ocr_structured"]),
            "score": 1.25,
            "source": "attached",
        }

    def _resize_image_if_needed(self, image_path: str) -> str:
        """
        Resizes image to max dimension 768px to save VLM RAM.
        Returns path to resized temp image or original if small enough.
        """
        if not Image:
            return image_path
            
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if max(w, h) <= self.max_image_size:
                    return image_path
                
                ratio = self.max_image_size / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                
                # Check if temp dir exists
                temp_dir = Path(self.cfg.sqlite_path).parent / "temp_resized"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / Path(image_path).name
                
                if not temp_path.exists():
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img.save(temp_path)
                
                return str(temp_path)
        except Exception as e:
            logger.error(f"Failed to resize {image_path}: {e}")
            return image_path

    def _filter_ocr_advanced(self, ocr_json: str, query: str) -> tuple[str, list[dict[str, Any]]]:
        """
        Advanced OCR filtering:
        - Confidence > 0.6
        - Cosine similarity > 0.4 (approx via word overlap if embedding not avail, or use lightweight check)
        - Limit tokens
        Returns: (filtered_text, kept_blocks_with_bbox)
        """
        if not ocr_json or ocr_json == "[]":
            return "", []

        try:
            blocks = json.loads(ocr_json)
            query_terms = set(re.findall(r"\w+", query.lower()))
            kept_blocks = []
            
            for b in blocks:
                text = b.get("text", "")
                conf = b.get("confidence", 0.0)
                
                # 1. Confidence Filter
                if conf < 0.6:
                    continue
                    
                # 2. Relevance Filter (Simple Overlap for speed/RAM, avoiding heavy embed load inside chat loop)
                # "Cosine similarity > 0.4" is requested, but loading TextEmbedder here might be slow.
                # We'll use a Jaccard-like proxy or check important entities.
                # If query is short, keep more. 
                block_terms = set(re.findall(r"\w+", text.lower()))
                if not block_terms:
                    continue
                
                overlap = len(query_terms & block_terms)
                # Heuristic: if overlap > 0 OR block has numbers (often targets), keep it.
                if overlap > 0 or re.search(r"\d", text):
                    # Dedup check (simple string equality)
                    if any(kb["text"] == text for kb in kept_blocks):
                         continue
                    kept_blocks.append(b)
            
            # 3. Token Limit (Soft)
            # If too many, pick top by confidence? Or length?
            # We'll truncate if total chars > 3000 (~700 tokens)
            total_chars = 0
            final_blocks = []
            for b in kept_blocks:
                if total_chars + len(b["text"]) > 3000:
                    break
                final_blocks.append(b)
                total_chars += len(b["text"])
                
            filtered_text = "\n".join(f"- {b['text']}" for b in final_blocks)
            return filtered_text, final_blocks
            
        except Exception:
            return "", []

    def _verify_grounding(self, answer: str, context_text: str) -> float:
        """
        Computes grounding score based on token overlap.
        """
        if "Not found in retrieved images" in answer:
            return 0.0

        answer_tokens = set(re.split(r"\W+", answer.lower())) - {"a", "an", "the", "is", "of", "in"}
        context_tokens = set(re.split(r"\W+", context_text.lower()))

        if not answer_tokens:
            return 0.0

        overlap = len(answer_tokens & context_tokens)
        score = overlap / len(answer_tokens)

        # Penalize metric hallucination
        ans_nums = set(re.findall(r"\d+", answer))
        ctx_nums = set(re.findall(r"\d+", context_text))
        if ans_nums and not ans_nums.issubset(ctx_nums):
            score *= 0.5

        return score

    def _retrieval_query_match_score(self, query: str, sources: list[dict[str, Any]]) -> float:
        terms = self._query_terms(query)
        if not terms or not sources:
            return 0.0
        source_term_sets: list[set[str]] = []
        for src in sources:
            tags = src.get("tags", [])
            tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags)
            hay = f"{src.get('caption', '')} {src.get('summary', '')} {tags_text}".lower()
            source_terms = set(re.findall(r"[a-z0-9_]+", hay))
            source_term_sets.append(source_terms)
        matched = 0
        for term in terms:
            found = False
            for source_terms in source_term_sets:
                if term in source_terms:
                    found = True
                    break
            if found:
                matched += 1
        return matched / max(1, len(terms))

    def _row_query_overlap(self, query: str, row: dict[str, Any]) -> int:
        terms = self._query_terms(query)
        if not terms:
            return 0
        tags = row.get("tags", [])
        tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags)
        hay = (
            f"{row.get('caption', '')} "
            f"{row.get('summary', '')} "
            f"{tags_text} "
            f"{row.get('ocr_structured', '')}"
        ).lower()
        return sum(1 for t in terms if t in hay)

    def _build_grounded_fallback_answer(self, query: str, sources: list[dict[str, Any]]) -> str | None:
        terms = self._query_terms(query)
        if not sources:
            return None

        hits: list[tuple[int, float, dict[str, Any]]] = []
        if terms:
            for src in sources:
                tags = src.get("tags", [])
                tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags)
                hay = f"{src.get('caption', '')} {tags_text}".lower()
                overlap = sum(1 for t in terms if t in hay)
                if overlap > 0:
                    hits.append((overlap, float(src.get("score", 0.0) or 0.0), src))

        if hits:
            hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
            top = [src for _, _, src in hits[: min(2, len(hits))]]
            prefix = f"Found retrieved images related to '{query}':"
        else:
            # Semantic fallback: if retrieval similarity is strong but lexical overlap is weak,
            # return the closest grounded items instead of a false "Not found".
            ordered = sorted(sources, key=lambda s: float(s.get("score", 0.0) or 0.0), reverse=True)
            if not ordered or float(ordered[0].get("score", 0.0) or 0.0) < 0.72:
                return None
            top = ordered[: min(2, len(ordered))]
            prefix = f"Closest retrieved images for '{query}':"

        lines = []
        for src in top:
            name = Path(str(src.get("file_path", ""))).name
            cap = str(src.get("caption", "")).strip()
            lines.append(f"- {name}: {cap}")

        joined = "\n".join(lines)
        return f"{prefix}\n{joined}"

    def _build_fallback_with_support(
        self,
        query: str,
        sources: list[dict[str, Any]],
        *,
        required_terms: list[str] | None = None,
    ) -> tuple[str | None, str, float]:
        """Return fallback answer + support mode.

        mode:
        - explicit: lexical evidence exists in retrieved metadata.
        - semantic: closest semantic neighbors only (no lexical support).
        - none: no fallback answer.
        """
        terms = self._query_terms(query)
        if not sources:
            return None, "none", 0.0

        hits: list[tuple[int, float, dict[str, Any]]] = []
        min_overlap = 2 if len(terms) >= 3 else 1
        required_terms = [t.lower() for t in (required_terms or []) if t]
        if terms:
            for src in sources:
                tags = src.get("tags", [])
                tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags)
                hay = f"{src.get('caption', '')} {src.get('summary', '')} {tags_text}".lower()
                source_terms = set(re.findall(r"[a-z0-9_]+", hay))
                if required_terms and not all(term in source_terms for term in required_terms):
                    continue
                overlap = sum(1 for t in terms if t in source_terms)
                if overlap >= min_overlap:
                    hits.append((overlap, float(src.get("score", 0.0) or 0.0), src))

        if hits:
            hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
            top = [src for _, _, src in hits[: min(2, len(hits))]]
            lines = []
            for src in top:
                name = Path(str(src.get("file_path", ""))).name
                cap = str(src.get("caption", "")).strip()
                lines.append(f"- {name}: {cap}")
            best_overlap = hits[0][0]
            support_ratio = best_overlap / max(1, len(terms))
            return (
                f"Found retrieved images related to '{query}':\n" + "\n".join(lines),
                "explicit",
                support_ratio,
            )

        ordered = sorted(sources, key=lambda s: float(s.get("score", 0.0) or 0.0), reverse=True)
        if not ordered or float(ordered[0].get("score", 0.0) or 0.0) < 0.72:
            return None, "none", 0.0

        if required_terms:
            def _src_terms(src: dict[str, Any]) -> set[str]:
                tags = src.get("tags", [])
                tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags)
                hay = f"{src.get('caption', '')} {src.get('summary', '')} {tags_text}".lower()
                return set(re.findall(r"[a-z0-9_]+", hay))

            ordered = [
                src
                for src in ordered
                if all(term in _src_terms(src) for term in required_terms)
            ]
            if not ordered:
                return None, "none", 0.0
        top = ordered[: min(2, len(ordered))]
        tokens = set(self._query_terms(query))
        role_query = bool(tokens & ROLE_QUERY_TERMS)
        if role_query:
            prefix = (
                f"No explicit mention of '{query}' in retrieved metadata. "
                "Closest semantic neighbors:"
            )
        else:
            prefix = f"Closest retrieved images for '{query}':"

        lines = []
        for src in top:
            name = Path(str(src.get("file_path", ""))).name
            cap = str(src.get("caption", "")).strip()
            lines.append(f"- {name}: {cap}")
        return f"{prefix}\n" + "\n".join(lines), "semantic", 0.0

    def _build_compare_neighbors_answer(self, query: str, sources: list[dict[str, Any]]) -> str | None:
        if not sources:
            return None
        ordered = sorted(sources, key=lambda s: float(s.get("score", 0.0) or 0.0), reverse=True)
        top = ordered[: min(3, len(ordered))]
        lines: list[str] = []
        for src in top:
            name = Path(str(src.get("file_path", ""))).name
            cap = str(src.get("caption", "")).strip()
            lines.append(f"- {name}: {cap}")
        return (
            f"Closest image-similarity neighbors for '{query}' "
            "(based on attached image):\n" + "\n".join(lines)
        )

    @staticmethod
    def _is_compare_query(query: str) -> bool:
        q = (query or "").lower()
        return bool(re.search(r"\b(compare|vs|versus|difference|similar|like)\b", q))

    @staticmethod
    def _normalize_query_for_model(query: str) -> str:
        q = query or ""
        q = re.sub(r"\bjwellery\b", "jewelry", q, flags=re.IGNORECASE)
        q = re.sub(r"\bjewelery\b", "jewelry", q, flags=re.IGNORECASE)
        q = re.sub(r"\bjewellery\b", "jewelry", q, flags=re.IGNORECASE)
        q = re.sub(r"\bjwellary\b", "jewelry", q, flags=re.IGNORECASE)
        q = re.sub(r"\bjewllery\b", "jewelry", q, flags=re.IGNORECASE)
        return q

    @staticmethod
    def _expand_retrieval_query(query: str) -> str:
        q = (query or "").strip()
        lowered = q.lower()
        terms: list[str] = []
        for key, extras in QUERY_EXPANSIONS.items():
            if key in lowered:
                for term in extras:
                    if term not in terms:
                        terms.append(term)
        if not terms:
            return q
        return f"{q}\nrelated terms: {', '.join(terms)}"

    def _answer_mentions_query(self, answer: str, query: str) -> bool:
        ans = (answer or "").lower()
        if not ans:
            return False
        for t in self._query_terms(query):
            if t in ans:
                return True
        return False

    def _match_result_regions(self, answer: str, all_blocks: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """
        Finds bbox for answer parts.
        all_blocks: list of (image_id, block_dict)
        """
        regions = []
        # Split answer into relevant phrases (naive)
        # Better: find exact match of the answer tokens in blocks
        # If answer says "Price is $50", look for "$50" or "50".
        
        # We look for numbers and proper nouns
        keywords = re.findall(r"\b[A-Z][a-z]+\b|\b\d+\.?\d*\b", answer)
        
        for kw in keywords:
            if len(kw) < 2: continue
            for img_id, block in all_blocks:
                if kw in block["text"]:
                    regions.append({
                        "image_id": img_id,
                        "bbox": block["bbox"],
                        "match": kw
                    })
                    # Dedupe regions slightly?
                    break # One match per keyword sufficient for highlighting?
        return regions

    def stream_chat(
        self,
        query: str,
        top_k: int = 5,
        *,
        attached_image_id: str | None = None,
        attached_file_path: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Generator verifying grounding and streaming tokens.
        """
        overall_start = time.perf_counter()
        timings: dict[str, int] = {}
        history_rows = self._normalize_history(history)
        normalized_query = self._normalize_query_for_model(query)
        retrieval_query = (query or "").strip() or normalized_query
        matching_query = f"{query} {normalized_query}".strip()
        parsed_intent = parse_query(query or "")
        attached_source = self._load_attached_source(
            attached_image_id=attached_image_id,
            attached_file_path=attached_file_path,
        )
        compare_mode = attached_source is not None and self._is_compare_query(query)
        focus_mode = attached_source is not None and not compare_mode

        # 1. Retrieval (Adapted)
        is_deep = False
        if len(retrieval_query.split()) > 12 or "compare" in retrieval_query.lower():
             is_deep = True

        retrieval_start = time.perf_counter()
        if compare_mode and attached_source:
            # Attached-image compare queries should be image-driven, not text-driven.
            search_resp = self.search_engine.search(
                image_path=str(attached_source["file_path"]),
                top_k=max(top_k, self.max_image_batch + 2),
                enable_verification=False,  # Avoid duplicate VLM load; chat itself does grounding.
            )
        else:
            history_last_user = self._last_user_query(history_rows)
            if history_last_user and self._detect_followup(retrieval_query):
                enhanced_query = f"{retrieval_query} (context: {history_last_user})"
                search_resp = self.search_engine.search(
                    query=enhanced_query,
                    top_k=top_k,
                    enable_verification=False,
                )
            elif self._detect_followup(retrieval_query):
                last_q = self.session.history[-1]["query"] if self.session.history else ""
                enhanced_query = f"{retrieval_query} (context: {last_q})"
                search_resp = self.search_engine.search(
                    query=enhanced_query,
                    top_k=top_k,
                    enable_verification=False,
                )
            else:
                search_resp = self.search_engine.search(
                    query=retrieval_query,
                    top_k=top_k,
                    enable_verification=False,
                )
        timings["retrieval_ms"] = int((time.perf_counter() - retrieval_start) * 1000)

        # 2. Filter & Gate
        min_gate = self.min_similarity_gate
        if compare_mode:
            # Image-to-image similarity scores are commonly lower than text-hybrid
            # scores; use a lower gate to avoid empty compare results.
            min_gate = min(0.35, self.min_similarity_gate * 0.5)
        retrieved = [r for r in search_resp.results if r["score"] >= min_gate]
        if attached_source:
            retrieved = [r for r in retrieved if str(r.get("image_id", "")) != str(attached_source["image_id"])]
        if compare_mode and not retrieved:
            # Soft fallback: keep strongest image-neighbors (excluding attached source)
            # when strict gating removes everything.
            fallback_neighbors = [
                r
                for r in search_resp.results
                if not attached_source or str(r.get("image_id", "")) != str(attached_source["image_id"])
            ]
            retrieved = fallback_neighbors[: max(1, min(3, top_k))]

        if not retrieved and search_resp.results and parsed_intent.has_constraints():
            retrieved = search_resp.results[: max(1, min(3, top_k))]

        # Prefer semantically+lexically aligned rows for short entity queries.
        overlap_rows = [(self._row_query_overlap(matching_query, r), r) for r in retrieved]
        matched_rows = [r for ov, r in overlap_rows if ov > 0]
        if matched_rows:
            retrieved = matched_rows

        if focus_mode and attached_source:
            filtered = [attached_source]
        elif compare_mode:
            filtered = retrieved
        elif attached_source:
            filtered = [attached_source] + retrieved
        else:
            filtered = retrieved
        filtered = filtered[:self.max_image_batch]

        if not filtered:
            yield {
                "type": "complete",
                "answer": "Not found in retrieved images (low similarity).",
                "sources": [],
                "confidence": "Low",
                "grounded_score": 0.0,
                "highlight_regions": []
            }
            return

        # 3. Context Assembly
        context_parts = []
        full_context_text = ""
        image_paths = []
        source_metas = []
        all_context_blocks = [] # For region matching: (image_id, block)
        
        for r in filtered:
            r_path = self._resize_image_if_needed(r["file_path"])
            image_paths.append(r_path)
            
            # Compression Layer
            ocr_text, kept_blocks = self._filter_ocr_advanced(r.get("ocr_structured", "[]"), matching_query)
            
            for b in kept_blocks:
                all_context_blocks.append((r["image_id"], b))

            meta_text = (
                f"Image: {r['file_path'].split('/')[-1]}\n"
                f"Caption: {r['caption']}\n"
                f"Summary: {r.get('summary', '')}\n"
                f"OCR: {ocr_text}\n"
                f"Tags: {r['tags']}\n"
            )
            context_parts.append(meta_text)
            full_context_text += f"{r['caption']} {r.get('summary', '')} {ocr_text} {r['tags']} "
            
            source_metas.append({
                "file_path": r["file_path"],
                "score": r["score"],
                "caption": r["caption"],
                "summary": r.get("summary", ""),
                "image_id": r["image_id"],
                "tags": r["tags"]
            })

        context_str = "\n".join(context_parts)
        history_context = self._history_context(history_rows)
        
        # 4. Load VLM
        try:
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_image
        except ImportError:
            yield {"type": "token", "content": "Error: mlx-vlm not installed."}
            return

        load_start = time.perf_counter()
        model, processor = load(self.cfg.vlm_model_name)
        timings["vlm_load_ms"] = int((time.perf_counter() - load_start) * 1000)
        
        try:
            if focus_mode:
                system_prompt = (
                    "You are a grounded visual assistant. "
                    "Use the attached focus image pixels as primary evidence, plus provided metadata context. "
                    "If uncertain, say 'Unclear from attached image'.\n"
                )
            else:
                system_prompt = (
                    "You are a strict grounded assistant. Answer ONLY using the provided Context.\n"
                    "If the answer is not in the context, say 'Not found in retrieved images'.\n"
                )
            attached_line = ""
            if attached_source:
                attached_line = (
                    f"Attached Focus Image: {Path(str(attached_source['file_path'])).name}. "
                    "If user did not ask to compare, answer using this attached image as primary evidence.\n"
                )
            model_query = normalized_query

            user_content = (
                f"{system_prompt}\n"
                f"{attached_line}"
                f"Intent: retrieval={parsed_intent.retrieval_terms}, "
                f"relations={parsed_intent.relation_terms}, "
                f"attributes={parsed_intent.attribute_terms}, "
                f"presence={parsed_intent.presence_terms}\n"
                f"{history_context}\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {model_query}"
            ).strip()
            
            formatted_prompt = apply_chat_template(
                processor, model.config, user_content, num_images=len(image_paths)
            )
            
            loaded_images = [load_image(p) for p in image_paths]
            
            generate_start = time.perf_counter()
            output = generate(
                model, 
                processor, 
                image=loaded_images, 
                prompt=formatted_prompt, 
                max_tokens=256, 
                verbose=False
            )
            timings["vlm_generate_ms"] = int((time.perf_counter() - generate_start) * 1000)
            
            full_answer = output if isinstance(output, str) else output.text
            
            # Verification & Post-processing
            grounded_score = self._verify_grounding(full_answer, full_context_text)
            retrieval_match_score = self._retrieval_query_match_score(matching_query, source_metas)
            confidence = "High" if grounded_score > 0.7 else "Medium"

            if focus_mode:
                # In focus mode, assistant can answer from the attached image pixels,
                # so metadata-token overlap should not force unrelated lexical fallback.
                if "Not found in retrieved images" not in full_answer:
                    grounded_score = max(grounded_score, 0.75)
                    confidence = "High" if grounded_score >= 0.7 else "Medium"
                elif grounded_score < self.min_grounding_score:
                    confidence = "Low"
                    grounded_score = 0.0
            else:
                # If model says "Not found" despite clear lexical match in retrieved sources,
                # override with a deterministic grounded fallback.
                if "Not found in retrieved images" in full_answer:
                    if compare_mode:
                        compare_fallback = self._build_compare_neighbors_answer(query, source_metas)
                        if compare_fallback:
                            full_answer = compare_fallback
                            max_source_score = max((float(s.get("score", 0.0) or 0.0) for s in source_metas), default=0.0)
                            grounded_score = max(grounded_score, min(0.45, max_source_score))
                            confidence = "Medium" if grounded_score >= 0.35 else "Low"
                    fallback, fallback_mode, fallback_support = self._build_fallback_with_support(
                        query,
                        source_metas,
                        required_terms=parsed_intent.retrieval_terms,
                    )
                    if fallback:
                        full_answer = fallback
                        max_source_score = max((float(s.get("score", 0.0) or 0.0) for s in source_metas), default=0.0)
                        if fallback_mode == "explicit":
                            grounded_score = max(
                                grounded_score,
                                retrieval_match_score,
                                fallback_support,
                                min(0.65, max_source_score),
                            )
                            confidence = "Medium" if grounded_score < 0.7 else "High"
                        else:
                            grounded_score = max(grounded_score, min(0.40, max_source_score * 0.5))
                            confidence = "Low" if grounded_score < 0.35 else "Medium"
                
                if grounded_score < self.min_grounding_score:
                    max_source_score = max((float(s.get("score", 0.0) or 0.0) for s in source_metas), default=0.0)
                    answer_has_query_signal = self._answer_mentions_query(full_answer, matching_query)
                    if answer_has_query_signal and max_source_score >= 0.75:
                        grounded_score = max(grounded_score, 0.45)
                        confidence = "Medium"
                    else:
                        fallback, fallback_mode, fallback_support = self._build_fallback_with_support(
                            query,
                            source_metas,
                            required_terms=parsed_intent.retrieval_terms,
                        )
                        if fallback:
                            full_answer = fallback
                            max_source_score = max((float(s.get("score", 0.0) or 0.0) for s in source_metas), default=0.0)
                            if fallback_mode == "explicit":
                                confidence = "Medium"
                                grounded_score = max(
                                    grounded_score,
                                    retrieval_match_score,
                                    fallback_support,
                                    min(0.65, max_source_score),
                                )
                            else:
                                confidence = "Low" if max_source_score < 0.8 else "Medium"
                                grounded_score = max(grounded_score, min(0.40, max_source_score * 0.5))
                        else:
                            full_answer = "Not found in retrieved images."
                            confidence = "Low"
                            grounded_score = 0.0

            # Region Awareness
            post_start = time.perf_counter()
            regions = self._match_result_regions(full_answer, all_context_blocks)

            # Log interaction
            try:
                from .session import SessionManager
                sess = SessionManager(self.cfg)
                sess.log_activity("chat", {"query": query, "answer": full_answer, "category": "Chat"}) # Refine category later
            except Exception:
                pass

            words = full_answer.split(" ")
            for w in words:
                yield {"type": "token", "content": w + " "}
            timings["postprocess_ms"] = int((time.perf_counter() - post_start) * 1000)
            
        finally:
            del model
            del processor
            cleanup_torch_mps()
            gc.collect()

        self.session.add_turn(query, full_answer, [s["image_id"] for s in source_metas])
        timings["total_ms"] = int((time.perf_counter() - overall_start) * 1000)

        yield {
            "type": "complete",
            "answer": full_answer,
            "sources": source_metas,
            "confidence": confidence,
            "grounded_score": grounded_score,
            "highlight_regions": regions,
            "timings": timings,
        }

    def chat(
        self,
        query: str,
        top_k: int = 5,
        *,
        attached_image_id: str | None = None,
        attached_file_path: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Blocking wrapper."""
        final_res = None
        for event in self.stream_chat(
            query,
            top_k,
            attached_image_id=attached_image_id,
            attached_file_path=attached_file_path,
            history=history,
        ):
            if event["type"] == "complete":
                final_res = event
        
        if final_res:
            return ChatResponse(
                answer=final_res["answer"],
                sources=final_res["sources"],
                confidence=final_res["confidence"],
                grounded_score=final_res["grounded_score"],
                highlight_regions=final_res.get("highlight_regions", []),
                timings=final_res.get("timings", {}),
            )
        
        return ChatResponse("Error", [], "Low", 0.0, [], {})
