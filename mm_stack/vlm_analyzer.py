from __future__ import annotations

import json
import re
from pathlib import Path

from .models import OCRBlock, VLMOutput
from .utils import cleanup_torch_mps


def _name_mentions_from_text(text: str) -> list[dict]:
    mentions: list[dict] = []
    # Lightweight NER-lite for names without extra model load.
    for match in re.findall(r"\b[A-Z][a-z]{2,}\b", text or ""):
        if match.lower() in {"the", "with", "from", "this", "that"}:
            continue
        mentions.append(
            {
                "mention": match,
                "mention_type": "name",
                "confidence": 0.55,
                "source_field": "summary",
            }
        )
    uniq: list[dict] = []
    seen: set[str] = set()
    for item in mentions:
        key = str(item["mention"]).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq[:12]


def _decode_json_string(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    try:
        return str(json.loads(f"\"{token}\""))
    except Exception:
        return token


def _extract_json_string_field(blob: str, field: str) -> str:
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, blob or "", flags=re.DOTALL)
    if not match:
        return ""
    return _decode_json_string(match.group(1)).strip()


def _extract_json_list_field(blob: str, field: str) -> list[str]:
    pattern = rf'"{re.escape(field)}"\s*:\s*(\[[^\]]*\])'
    match = re.search(pattern, blob or "", flags=re.DOTALL)
    if not match:
        return []
    raw = match.group(1)
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x).strip().lower() for x in arr if str(x).strip()]
    except Exception:
        pass
    return []


def _build_payload(
    *,
    caption: str,
    summary: str,
    category: str,
    tags: list[str],
    entities: list[dict] | None = None,
    relations: list[dict] | None = None,
    mentions: list[dict] | None = None,
) -> dict:
    cap = (caption or "").strip()
    summ = (summary or "").strip()
    if not cap and summ:
        cap = summ
    if not summ and cap:
        summ = cap
    if not mentions:
        mentions = _name_mentions_from_text(f"{cap}\n{summ}")
    return {
        "caption": cap,
        "summary": summ,
        "category": (category or "Other").strip() or "Other",
        "tags": [str(x).strip().lower() for x in (tags or []) if str(x).strip()][:8],
        "entities": entities or [],
        "relations": relations or [],
        "mentions": mentions[:12],
    }


def _parse_json_like(text: str) -> dict:
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
            entities_raw = obj.get("entities", [])
            attributes_raw = obj.get("attributes", [])
            relations_raw = obj.get("relations", [])
            mentions_raw = obj.get("mentions", [])

            entities: list[dict] = []
            if isinstance(entities_raw, list):
                for entry in entities_raw:
                    if not isinstance(entry, dict):
                        continue
                    label = str(entry.get("entity_label", "")).strip().lower()
                    if not label:
                        continue
                    ent = {
                        "entity_label": label,
                        "entity_type": str(entry.get("entity_type", "unknown")).strip().lower() or "unknown",
                        "confidence": float(entry.get("confidence", 0.0) or 0.0),
                        "bbox": entry.get("bbox", []),
                        "attributes": [],
                    }
                    entities.append(ent)

            if isinstance(attributes_raw, list):
                by_label: dict[str, list[dict]] = {}
                for entry in attributes_raw:
                    if not isinstance(entry, dict):
                        continue
                    target = str(entry.get("entity_label", "")).strip().lower()
                    key = str(entry.get("attr_key", "")).strip().lower()
                    value = str(entry.get("attr_value", "")).strip().lower()
                    if not target or not key or not value:
                        continue
                    by_label.setdefault(target, []).append(
                        {
                            "attr_key": key,
                            "attr_value": value,
                            "confidence": float(entry.get("confidence", 0.0) or 0.0),
                        }
                    )
                for ent in entities:
                    ent["attributes"] = by_label.get(str(ent["entity_label"]), [])

            relations: list[dict] = []
            if isinstance(relations_raw, list):
                for entry in relations_raw:
                    if not isinstance(entry, dict):
                        continue
                    relation = str(entry.get("relation", "")).strip().lower()
                    if not relation:
                        continue
                    relations.append(
                        {
                            "subject": str(entry.get("subject", "")).strip().lower(),
                            "relation": relation,
                            "object": str(entry.get("object", "")).strip().lower(),
                            "confidence": float(entry.get("confidence", 0.0) or 0.0),
                            "evidence_text": str(entry.get("evidence_text", "")).strip(),
                        }
                    )

            mentions: list[dict] = []
            if isinstance(mentions_raw, list):
                for entry in mentions_raw:
                    if not isinstance(entry, dict):
                        continue
                    mention = str(entry.get("mention", "")).strip()
                    if not mention:
                        continue
                    mentions.append(
                        {
                            "mention": mention,
                            "mention_type": str(entry.get("mention_type", "name")),
                            "confidence": float(entry.get("confidence", 0.0) or 0.0),
                            "source_field": str(entry.get("source_field", "summary")),
                        }
                    )
            if not mentions:
                mentions = _name_mentions_from_text(f"{caption}\n{summary}")

            return _build_payload(
                caption=caption,
                summary=summary,
                category=category,
                tags=tags,
                entities=entities,
                relations=relations,
                mentions=mentions,
            )
        except Exception:
            pass

    # Recover partially emitted JSON payloads:
    # e.g. caption='{' and summary starts with '"caption": "..."'
    recovered_caption = _extract_json_string_field(candidate, "caption")
    recovered_summary = _extract_json_string_field(candidate, "summary")
    recovered_category = _extract_json_string_field(candidate, "category") or "Other"
    recovered_tags = _extract_json_list_field(candidate, "tags")
    if recovered_caption or recovered_summary or recovered_tags:
        return _build_payload(
            caption=recovered_caption,
            summary=recovered_summary,
            category=recovered_category,
            tags=recovered_tags,
        )

    lines = [
        ln.strip().strip(",")
        for ln in candidate.splitlines()
        if ln.strip() and ln.strip() not in {"{", "}", "[", "]"}
    ]
    caption = lines[0][:220] if lines else ""
    summary = lines[1][:400] if len(lines) > 1 else caption
    tags = [t for t in re.split(r"[\s,;|]+", caption.lower()) if len(t) > 3][:6]
    return _build_payload(
        caption=caption,
        summary=summary,
        category="Other",
        tags=tags,
    )


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
            '{"caption":"","summary":"","category":"","tags":[""],'
            '"entities":[{"entity_label":"","entity_type":"","confidence":0.0,"bbox":[0,0,0,0]}],'
            '"attributes":[{"entity_label":"","attr_key":"","attr_value":"","confidence":0.0}],'
            '"relations":[{"subject":"","relation":"","object":"","confidence":0.0,"evidence_text":""}],'
            '"mentions":[{"mention":"","mention_type":"name","confidence":0.0,"source_field":"summary"}]}\n'
            "Rules:\n"
            "- caption: one concise factual sentence\n"
            "- summary: 2-3 short factual sentences\n"
            "- category: exactly one of [Finance, Political, Design, Academic, Personal, Technical, Other]\n"
            "- tags: lowercase keywords, no hallucinations\n"
            "- entities: generic object/person rows with confidence\n"
            "- attributes: color/pattern/age_group/clothing_type/activity when available\n"
            "- relations: subject-relation-object triples when visible\n"
            "- mentions: person/location/org names from visible text/caption context\n"
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
        parsed = _parse_json_like(text)
        caption = str(parsed.get("caption", ""))
        summary = str(parsed.get("summary", ""))
        category = str(parsed.get("category", "Other"))
        tags = parsed.get("tags", [])
        entities = parsed.get("entities", [])
        relations = parsed.get("relations", [])
        mentions = parsed.get("mentions", [])

        if not caption:
            caption = "image with identifiable visual content"
        if not summary:
            summary = caption
        if not tags:
            tags = ["image"]

        return VLMOutput(
            caption=caption[:220],
            summary=summary[:600],
            category=category,
            tags=tags[:8],
            raw_output=text,
            entities=entities if isinstance(entities, list) else [],
            relations=relations if isinstance(relations, list) else [],
            mentions=mentions if isinstance(mentions, list) else [],
        )

    def generate_text(self, image_path: Path, prompt: str) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("VLM model not loaded")

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_image
        
        formatted_prompt = apply_chat_template(self.processor, self.model.config, prompt, num_images=1)
        raw = generate(
            self.model,
            self.processor,
            image=[load_image(str(image_path))],
            prompt=formatted_prompt,
            max_tokens=60,
            temperature=0.1,
        )
        return raw.text if hasattr(raw, "text") else str(raw)
