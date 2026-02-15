from __future__ import annotations

import json
from pathlib import Path

from .clip_embedder import OpenCLIPEmbedder
from .config import StackConfig
from .db import (
    connect_sqlite,
    ensure_schema,
    list_all_images,
    mark_stale_if_versions_mismatch,
    upsert_image_metadata,
    upsert_vector_metadata,
)
from .lancedb_store import LanceStore
from .preprocess import preprocess_image
from .text_embedder import TextEmbedder
from .utils import sha256_text, utc_now_iso


def _ocr_text(ocr_structured_json: str) -> str:
    try:
        blocks = json.loads(ocr_structured_json or "[]")
        if isinstance(blocks, list):
            return "\n".join(str(b.get("text", "")).strip() for b in blocks if isinstance(b, dict))
    except Exception:
        pass
    return ""


def _text_payload(row) -> str:
    caption = str(row["caption"] or "")
    summary = str(row["summary"] or "")
    ocr_text = _ocr_text(str(row["ocr_structured"] or "[]"))
    return "\n".join(x for x in (caption, summary, ocr_text) if x.strip())


def reembed_all(cfg: StackConfig | None = None) -> dict[str, int]:
    cfg = cfg or StackConfig()
    conn = connect_sqlite(cfg)
    ensure_schema(conn)
    store = LanceStore(cfg)

    stale_marked = mark_stale_if_versions_mismatch(conn, cfg)
    rows = list_all_images(conn)

    clip_targets = []
    text_targets = []

    for row in rows:
        image_id = str(row["id"])
        file_path = Path(str(row["file_path"]))
        payload = _text_payload(row)
        payload_hash = sha256_text(payload)

        clip_mismatch = (
            str(row["embedding_model_clip"]) != cfg.clip_model_name
            or int(row["embedding_dimension_clip"]) != cfg.clip_dimension
            or str(row["embedding_schema_version_clip"]) != cfg.clip_schema_version
        )
        text_mismatch = (
            str(row["embedding_model_text"]) != cfg.text_model_name
            or int(row["embedding_dimension_text"]) != cfg.text_dimension
            or str(row["embedding_schema_version_text"]) != cfg.text_schema_version
        )

        text_changed = payload_hash != str(row["text_payload_hash"] or "")
        clip_changed = str(row["clip_content_hash"] or "") != str(row["sha256_hash"] or "")

        if (int(row["is_stale"]) == 1 or clip_mismatch or clip_changed) and file_path.exists():
            clip_targets.append((image_id, file_path))

        if int(row["is_stale"]) == 1 or text_mismatch or text_changed:
            text_targets.append((image_id, payload, payload_hash))

    clip_done = 0
    text_done = 0

    if clip_targets:
        prepared = []
        valid_ids = []
        for image_id, path in clip_targets:
            try:
                p = preprocess_image(path, cfg)
                prepared.append(p)
                valid_ids.append(image_id)
            except Exception:
                continue

        if prepared:
            with OpenCLIPEmbedder(cfg.clip_model_name) as clip:
                vecs = clip.encode_images([p.normalized_path for p in prepared])
            now = utc_now_iso()
            for image_id, p, vec in zip(valid_ids, prepared, vecs, strict=True):
                store.upsert_clip_vector(
                    image_id=image_id,
                    vector=vec,
                    model_name=cfg.clip_model_name,
                    schema_version=cfg.clip_schema_version,
                    created_at=now,
                )
                upsert_vector_metadata(
                    conn,
                    table_name="clip_vectors",
                    image_id=image_id,
                    vector_id=f"clip:{image_id}",
                    model_name=cfg.clip_model_name,
                    dimension=cfg.clip_dimension,
                    schema_version=cfg.clip_schema_version,
                )
                row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
                if row:
                    upsert_image_metadata(
                        conn,
                        {
                            "id": image_id,
                            "file_path": str(row["file_path"]),
                            "sha256_hash": str(row["sha256_hash"]),
                            "width": int(row["width"]),
                            "height": int(row["height"]),
                            "caption": str(row["caption"]),
                            "summary": str(row["summary"]),
                            "tags": json.loads(str(row["tags"])),
                            "ocr_structured": json.loads(str(row["ocr_structured"])),
                            "ocr_confidence_avg": float(row["ocr_confidence_avg"]),
                            "schema_version": cfg.schema_version,
                            "embedding_model_clip": cfg.clip_model_name,
                            "embedding_model_text": str(row["embedding_model_text"]),
                            "embedding_dimension_clip": cfg.clip_dimension,
                            "embedding_dimension_text": int(row["embedding_dimension_text"]),
                            "embedding_schema_version_clip": cfg.clip_schema_version,
                            "embedding_schema_version_text": str(row["embedding_schema_version_text"]),
                            "text_payload_hash": str(row["text_payload_hash"]),
                            "clip_content_hash": str(row["sha256_hash"]),
                            "is_stale": 0,
                            "created_at": str(row["created_at"]),
                        },
                    )
                clip_done += 1
            conn.commit()

    if text_targets:
        with TextEmbedder(cfg.text_model_name) as text_model:
            vecs = text_model.encode([t[1] for t in text_targets], is_query=False)
        now = utc_now_iso()
        for (image_id, payload, payload_hash), vec in zip(text_targets, vecs, strict=True):
            store.upsert_text_vector(
                image_id=image_id,
                vector=vec,
                model_name=cfg.text_model_name,
                schema_version=cfg.text_schema_version,
                created_at=now,
            )
            upsert_vector_metadata(
                conn,
                table_name="text_vectors",
                image_id=image_id,
                vector_id=f"text:{image_id}",
                model_name=cfg.text_model_name,
                dimension=cfg.text_dimension,
                schema_version=cfg.text_schema_version,
            )
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if row:
                upsert_image_metadata(
                    conn,
                    {
                        "id": image_id,
                        "file_path": str(row["file_path"]),
                        "sha256_hash": str(row["sha256_hash"]),
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "caption": str(row["caption"]),
                        "summary": str(row["summary"]),
                        "tags": json.loads(str(row["tags"])),
                        "ocr_structured": json.loads(str(row["ocr_structured"])),
                        "ocr_confidence_avg": float(row["ocr_confidence_avg"]),
                        "schema_version": cfg.schema_version,
                        "embedding_model_clip": str(row["embedding_model_clip"]),
                        "embedding_model_text": cfg.text_model_name,
                        "embedding_dimension_clip": int(row["embedding_dimension_clip"]),
                        "embedding_dimension_text": cfg.text_dimension,
                        "embedding_schema_version_clip": str(row["embedding_schema_version_clip"]),
                        "embedding_schema_version_text": cfg.text_schema_version,
                        "text_payload_hash": payload_hash,
                        "clip_content_hash": str(row["clip_content_hash"]),
                        "is_stale": 0,
                        "created_at": str(row["created_at"]),
                    },
                )
            text_done += 1
        conn.commit()

    conn.close()
    return {
        "stale_marked": stale_marked,
        "clip_reembedded": clip_done,
        "text_reembedded": text_done,
    }
