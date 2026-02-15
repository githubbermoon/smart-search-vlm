from __future__ import annotations

import json
import re
from pathlib import Path

from .models import OCRBlock, VLMOutput
from .utils import cleanup_torch_mps


def _parse_json_like(text: str) -> tuple[str, str, str, list[str]]:
    candidate = (text or "").strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        snippet = candidate[start : end + 1]
        try:
            obj = json.loads(snippet)
            caption = str(obj.get("caption", "")).strip()
            summary = str(obj.get("summary", "")).strip()
            category = str(obj.get("category", "Other")).strip()
            tags_raw = obj.get("tags", [])
            if isinstance(tags_raw, list):
                tags = [str(x).strip().lower() for x in tags_raw if str(x).strip()]
            elif isinstance(tags_raw, str):
                tags = [t.strip().lower() for t in re.split(r"[,;|]", tags_raw) if t.strip()]
            else:
                tags = []
            return caption, summary, category, tags[:8]
        except Exception:
            pass

    lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
    caption = lines[0][:220] if lines else ""
    summary = lines[1][:400] if len(lines) > 1 else caption
    tags = [t for t in re.split(r"[\s,;|]+", caption.lower()) if len(t) > 3][:6]
    return caption, summary, "Other", tags


class VLMAnalyzer:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self) -> None:
        try:
            from mlx_vlm import load
        except Exception as exc:
            raise RuntimeError("mlx-vlm is required for VLM analysis") from exc
        self.model, self.processor = load(self.model_id)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        cleanup_torch_mps()

    def __enter__(self) -> "VLMAnalyzer":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def analyze(self, image_path: Path, ocr_blocks: list[OCRBlock]) -> VLMOutput:
        if self.model is None or self.processor is None:
            raise RuntimeError("VLM model not loaded")

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_image

        ocr_lines = [block.text for block in ocr_blocks if block.text.strip()]
        ocr_text = "\n".join(ocr_lines[:50]) or "(none)"

        user_prompt = (
            "You are a deterministic image analysis engine for local indexing.\n"
            "Return valid JSON only with this schema:\n"
            '{"caption":"","summary":"","category":"","tags":["","","","",""]}\n'
            "Rules:\n"
            "- caption: one concise factual sentence\n"
            "- summary: 2-3 short factual sentences\n"
            "- category: exactly one of [Finance, Political, Design, Academic, Personal, Technical, Other]\n"
            "- tags: lowercase keywords, no hallucinations\n"
            "- use OCR text only when present\n"
            f"OCR text:\n{ocr_text}\n"
        )

        prompt = apply_chat_template(self.processor, self.model.config, user_prompt, num_images=1)
        raw = generate(
            self.model,
            self.processor,
            image=[load_image(str(image_path))],
            prompt=prompt,
            max_tokens=320,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
        )
        text = raw.text if hasattr(raw, "text") else str(raw)
        caption, summary, category, tags = _parse_json_like(text)

        if not caption:
            caption = "image with identifiable visual content"
        if not summary:
            summary = caption
        if not tags:
            tags = ["image"]

        return VLMOutput(caption=caption[:220], summary=summary[:600], category=category, tags=tags[:8], raw_output=text)
