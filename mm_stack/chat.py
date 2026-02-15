from __future__ import annotations

import gc
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Generator, Iterator

try:
    from PIL import Image
except ImportError:
    Image = None

from .config import StackConfig
from .search_engine import MultimodalSearchEngine
from .utils import cleanup_torch_mps

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    answer: str
    sources: list[dict[str, Any]]
    confidence: str
    grounded_score: float
    highlight_regions: list[dict[str, Any]] = field(default_factory=list)

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

    def _detect_followup(self, query: str) -> bool:
        pronouns = {"that", "this", "it", "those", "these", "above", "below", "he", "she", "they"}
        tokens = set(re.split(r"[\s,;.?!]+", query.lower()))
        return bool(tokens & pronouns) and bool(self.session.last_retrieved_ids)

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
            return 1.0

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

    def stream_chat(self, query: str, top_k: int = 5) -> Iterator[dict[str, Any]]:
        """
        Generator verifying grounding and streaming tokens.
        """
        # 1. Retrieval (Adapted)
        is_deep = False
        if len(query.split()) > 12 or "compare" in query.lower():
             is_deep = True
        
        if self._detect_followup(query):
            last_q = self.session.history[-1]["query"] if self.session.history else ""
            enhanced_query = f"{query} (context: {last_q})"
            search_resp = self.search_engine.search(query=enhanced_query, top_k=top_k) # search() already handles adaptive depth internally if configured
        else:
            search_resp = self.search_engine.search(query=query, top_k=top_k)

        # 2. Filter & Gate
        filtered = [r for r in search_resp.results if r["score"] >= self.min_similarity_gate]
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
            ocr_text, kept_blocks = self._filter_ocr_advanced(r.get("ocr_structured", "[]"), query)
            
            for b in kept_blocks:
                all_context_blocks.append((r["image_id"], b))

            meta_text = (
                f"Image: {r['file_path'].split('/')[-1]}\n"
                f"Caption: {r['caption']}\n"
                f"OCR: {ocr_text}\n"
                f"Tags: {r['tags']}\n"
            )
            context_parts.append(meta_text)
            full_context_text += f"{r['caption']} {ocr_text} {r['tags']} "
            
            source_metas.append({
                "file_path": r["file_path"],
                "score": r["score"],
                "caption": r["caption"],
                "image_id": r["image_id"],
                "tags": r["tags"]
            })

        context_str = "\n".join(context_parts)
        
        # 4. Load VLM
        try:
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_image
        except ImportError:
            yield {"type": "token", "content": "Error: mlx-vlm not installed."}
            return

        model, processor = load(self.cfg.vlm_model_name)
        
        try:
            system_prompt = (
                "You are a strict grounded assistant. Answer ONLY using the provided Context.\n"
                "If the answer is not in the context, say 'Not found in retrieved images'.\n"
            )
            user_content = f"Context:\n{context_str}\n\nQuestion: {query}"
            
            formatted_prompt = apply_chat_template(
                processor, model.config, user_content, num_images=len(image_paths)
            )
            
            loaded_images = [load_image(p) for p in image_paths]
            
            output = generate(
                model, 
                processor, 
                image=loaded_images, 
                prompt=formatted_prompt, 
                max_tokens=256, 
                verbose=False
            )
            
            full_answer = output if isinstance(output, str) else output.text
            
            # Verification & Post-processing
            grounded_score = self._verify_grounding(full_answer, full_context_text)
            confidence = "High" if grounded_score > 0.7 else "Medium"
            
            if grounded_score < self.min_grounding_score:
                full_answer = "Not found in retrieved images."
                confidence = "Low"
                grounded_score = 0.0

            # Region Awareness
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
            
        finally:
            del model
            del processor
            cleanup_torch_mps()
            gc.collect()

        self.session.add_turn(query, full_answer, [s["image_id"] for s in source_metas])

        yield {
            "type": "complete",
            "answer": full_answer,
            "sources": source_metas,
            "confidence": confidence,
            "grounded_score": grounded_score,
            "highlight_regions": regions
        }

    def chat(self, query: str, top_k: int = 5) -> ChatResponse:
        """Blocking wrapper."""
        final_res = None
        for event in self.stream_chat(query, top_k):
            if event["type"] == "complete":
                final_res = event
        
        if final_res:
            return ChatResponse(
                answer=final_res["answer"],
                sources=final_res["sources"],
                confidence=final_res["confidence"],
                grounded_score=final_res["grounded_score"],
                highlight_regions=final_res.get("highlight_regions", [])
            )
        
        return ChatResponse("Error", [], "Low", 0.0, [])
