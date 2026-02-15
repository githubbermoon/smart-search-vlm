from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StackConfig:
    stack_root: Path = Path("/Users/pranjal/garage/smart_stack")
    vault_root: Path = Path("/Users/pranjal/Pranjal-Obs/clawd")

    sqlite_path: Path = Path("/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db")
    lancedb_path: Path = Path("/Users/pranjal/Pranjal-Obs/clawd/vectors.lance")

    inbox_dir: Path = Path("/Users/pranjal/garage/smart_stack/inbox")
    processed_dir: Path = Path("/Users/pranjal/garage/smart_stack/processed")
    failed_dir: Path = Path("/Users/pranjal/garage/smart_stack/failed")
    media_dir: Path = Path("/Users/pranjal/Pranjal-Obs/clawd/Media")
    preprocessed_dir: Path = Path("/Users/pranjal/garage/smart_stack/.cache/preprocessed")

    clip_index_name: str = "clip_index"
    text_index_name: str = "text_index"

    schema_version: str = "mm-v1"
    clip_schema_version: str = "clip-v1"
    text_schema_version: str = "text-v1"

    clip_model_name: str = os.getenv("SMART_STACK_CLIP_MODEL", "open_clip:ViT-B-32/laion2b_s34b_b79k")
    text_model_name: str = os.getenv("SMART_STACK_TEXT_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    vlm_model_name: str = os.getenv("SMART_STACK_VLM_MODEL", "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit")

    clip_dimension: int = 512
    text_dimension: int = 768

    max_image_dim: int = 1024
    supported_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp", ".tiff")


OCR_INTENT_KEYWORDS: tuple[str, ...] = (
    "invoice",
    "receipt",
    "extract text",
    "amount",
    "convert to latex",
    "number",
    "total",
    "document",
    "bill",
)

VISUAL_INTENT_KEYWORDS: tuple[str, ...] = (
    "similar",
    "looks like",
    "style",
    "layout",
    "poster",
    "design",
    "hoarding",
    "diagram",
)
