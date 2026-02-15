from __future__ import annotations

import shutil
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clip_embedder import OpenCLIPEmbedder
from .config import StackConfig
from .db import (
    connect_sqlite,
    ensure_schema,
    get_image_by_hash,
    upsert_image_metadata,
    upsert_vector_metadata,
)
from .lancedb_store import LanceStore
from .models import OCRBlock, PreparedImage, VLMOutput
from .ocr import extract_ocr_structured
from .preprocess import preprocess_image
from .text_embedder import TextEmbedder
from .utils import sha256_text, utc_now_iso
from .vlm_analyzer import VLMAnalyzer


@dataclass
class Candidate:
    image_id: str
    prepared: PreparedImage
    ocr_blocks: list[OCRBlock]
    ocr_conf_avg: float
    existing_file_path: str | None = None
    vlm: VLMOutput | None = None
    clip_vec: list[float] | None = None
    text_vec: list[float] | None = None
    text_payload_hash: str = ""


def _unique_dest(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        alt = directory / f"{stem}_{i}{suffix}"
        if not alt.exists():
            return alt
        i += 1


def _copy_to_media(src: Path, media_dir: Path) -> Path:
    media_dir.mkdir(parents=True, exist_ok=True)
    dst = _unique_dest(media_dir, src.name)
    shutil.copy2(str(src), str(dst))
    return dst


def _move_to_processed(src: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dst = _unique_dest(processed_dir, src.name)
    shutil.move(str(src), str(dst))
    return dst


def _ocr_text_from_blocks(blocks: list[OCRBlock]) -> str:
    return "\n".join(block.text for block in blocks if block.text.strip())


def _ocr_to_json(blocks: list[OCRBlock]) -> list[dict[str, Any]]:
    return [
        {
            "type": b.block_type,
            "text": b.text,
            "bbox": b.bbox,
            "confidence": b.confidence,
        }
        for b in blocks
    ]


def _build_text_payload(c: Candidate) -> str:
    ocr_text = _ocr_text_from_blocks(c.ocr_blocks)
    caption = c.vlm.caption if c.vlm else ""
    summary = c.vlm.summary if c.vlm else ""
    return "\n".join(x for x in (caption, summary, ocr_text) if x.strip())


class MultimodalIngestor:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()
        self.cfg.preprocessed_dir.mkdir(parents=True, exist_ok=True)

    def ingest_image(self, image_path: str | Path, *, safe_reprocess: bool = False) -> dict[str, Any]:
        return self.ingest_batch([Path(image_path)], safe_reprocess=safe_reprocess)

    def ingest_inbox(self, *, limit: int = 0, safe_reprocess: bool = False) -> dict[str, Any]:
        source = self.cfg.processed_dir if safe_reprocess else self.cfg.inbox_dir
        files = [
            p
            for p in sorted(source.iterdir())
            if p.is_file() and p.suffix.lower() in self.cfg.supported_exts
        ]
        if limit > 0:
            files = files[:limit]
        return self.ingest_batch(files, safe_reprocess=safe_reprocess)

    def ingest_batch(self, image_paths: list[Path], *, safe_reprocess: bool = False) -> dict[str, Any]:
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        store = LanceStore(self.cfg)

        candidates: list[Candidate] = []
        skipped_duplicates = 0
        failures: list[str] = []

        # Stage 1: preprocess + OCR + dedupe check (no CLIP/VLM/text model loaded).
        for image_path in image_paths:
            try:
                prepared = preprocess_image(Path(image_path), self.cfg)
                existing = get_image_by_hash(conn, prepared.sha256_hash)
                if existing and not safe_reprocess:
                    skipped_duplicates += 1
                    continue
                image_id = str(existing["id"]) if existing else str(uuid.uuid4())
                existing_file_path = str(existing["file_path"]) if existing else None

                ocr_blocks, ocr_conf = extract_ocr_structured(prepared.normalized_path, prepared.width, prepared.height)
                candidates.append(
                    Candidate(
                        image_id=image_id,
                        prepared=prepared,
                        ocr_blocks=ocr_blocks,
                        ocr_conf_avg=ocr_conf,
                        existing_file_path=existing_file_path,
                    )
                )
            except Exception as exc:
                failures.append(f"{image_path}: {exc}")
                traceback.print_exc()

        if not candidates:
            conn.close()
            return {
                "ingested": 0,
                "skipped_duplicates": skipped_duplicates,
                "failed": failures,
            }

        # Stage 2: CLIP embeddings.
        with OpenCLIPEmbedder(self.cfg.clip_model_name) as clip:
            clip_vectors = clip.encode_images([c.prepared.normalized_path for c in candidates])
            for candidate, vec in zip(candidates, clip_vectors, strict=True):
                candidate.clip_vec = vec

        # Stage 3: VLM analysis.
        with VLMAnalyzer(self.cfg.vlm_model_name) as vlm:
            for candidate in candidates:
                candidate.vlm = vlm.analyze(candidate.prepared.normalized_path, candidate.ocr_blocks)

        # Stage 4: text embeddings from caption + summary + OCR text.
        with TextEmbedder(self.cfg.text_model_name) as text_embedder:
            payloads = []
            for candidate in candidates:
                payload = _build_text_payload(candidate)
                candidate.text_payload_hash = sha256_text(payload)
                payloads.append(payload)
            vectors = text_embedder.encode(payloads, is_query=False)
            for candidate, vec in zip(candidates, vectors, strict=True):
                candidate.text_vec = vec

        # Stage 5: persist metadata + vectors.
        ingested_count = 0
        now = utc_now_iso()
        for candidate in candidates:
            try:
                source_path = candidate.prepared.source_path
                if safe_reprocess and candidate.existing_file_path:
                    media_path = Path(candidate.existing_file_path)
                else:
                    media_path = _copy_to_media(source_path, self.cfg.media_dir)
                if not safe_reprocess and source_path.exists():
                    _move_to_processed(source_path, self.cfg.processed_dir)

                ocr_json = _ocr_to_json(candidate.ocr_blocks)
                tags = candidate.vlm.tags if candidate.vlm else []
                caption = candidate.vlm.caption if candidate.vlm else ""
                summary = candidate.vlm.summary if candidate.vlm else ""

                upsert_image_metadata(
                    conn,
                    {
                        "id": candidate.image_id,
                        "file_path": str(media_path),
                        "sha256_hash": candidate.prepared.sha256_hash,
                        "width": candidate.prepared.width,
                        "height": candidate.prepared.height,
                        "caption": caption,
                        "summary": summary,
                        "tags": tags,
                        "ocr_structured": ocr_json,
                        "ocr_confidence_avg": candidate.ocr_conf_avg,
                        "schema_version": self.cfg.schema_version,
                        "embedding_model_clip": self.cfg.clip_model_name,
                        "embedding_model_text": self.cfg.text_model_name,
                        "embedding_dimension_clip": self.cfg.clip_dimension,
                        "embedding_dimension_text": self.cfg.text_dimension,
                        "embedding_schema_version_clip": self.cfg.clip_schema_version,
                        "embedding_schema_version_text": self.cfg.text_schema_version,
                        "text_payload_hash": candidate.text_payload_hash,
                        "clip_content_hash": candidate.prepared.sha256_hash,
                        "is_stale": 0,
                        "created_at": now,
                    },
                )

                if not candidate.clip_vec or not candidate.text_vec:
                    raise RuntimeError("Missing embedding vectors during persist")

                store.upsert_clip_vector(
                    image_id=candidate.image_id,
                    vector=candidate.clip_vec,
                    model_name=self.cfg.clip_model_name,
                    schema_version=self.cfg.clip_schema_version,
                    created_at=now,
                )
                upsert_vector_metadata(
                    conn,
                    table_name="clip_vectors",
                    image_id=candidate.image_id,
                    vector_id=f"clip:{candidate.image_id}",
                    model_name=self.cfg.clip_model_name,
                    dimension=self.cfg.clip_dimension,
                    schema_version=self.cfg.clip_schema_version,
                )

                store.upsert_text_vector(
                    image_id=candidate.image_id,
                    vector=candidate.text_vec,
                    model_name=self.cfg.text_model_name,
                    schema_version=self.cfg.text_schema_version,
                    created_at=now,
                )
                upsert_vector_metadata(
                    conn,
                    table_name="text_vectors",
                    image_id=candidate.image_id,
                    vector_id=f"text:{candidate.image_id}",
                    model_name=self.cfg.text_model_name,
                    dimension=self.cfg.text_dimension,
                    schema_version=self.cfg.text_schema_version,
                )

                conn.commit()
                ingested_count += 1
            except Exception as exc:
                conn.rollback()
                failures.append(f"{candidate.prepared.source_path}: {exc}")
                traceback.print_exc()

        conn.close()
        return {
            "ingested": ingested_count,
            "skipped_duplicates": skipped_duplicates,
            "failed": failures,
        }
