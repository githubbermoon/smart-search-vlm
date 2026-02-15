from __future__ import annotations

import sqlite3
from typing import Any

from .config import StackConfig
from .utils import json_dumps, utc_now_iso


def connect_sqlite(cfg: StackConfig) -> sqlite3.Connection:
    cfg.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            sha256_hash TEXT NOT NULL UNIQUE,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            caption TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '[]',
            ocr_structured TEXT NOT NULL DEFAULT '[]',
            ocr_confidence_avg REAL NOT NULL DEFAULT 0.0,
            schema_version TEXT NOT NULL,
            embedding_model_clip TEXT NOT NULL,
            embedding_model_text TEXT NOT NULL,
            embedding_dimension_clip INTEGER NOT NULL,
            embedding_dimension_text INTEGER NOT NULL,
            embedding_schema_version_clip TEXT NOT NULL,
            embedding_schema_version_text TEXT NOT NULL,
            text_payload_hash TEXT NOT NULL DEFAULT '',
            clip_content_hash TEXT NOT NULL DEFAULT '',
            is_stale INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS clip_vectors (
            id TEXT PRIMARY KEY,
            image_id TEXT NOT NULL UNIQUE,
            embedding_model_name TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            embedding_schema_version TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(id)
        );

        CREATE TABLE IF NOT EXISTS text_vectors (
            id TEXT PRIMARY KEY,
            image_id TEXT NOT NULL UNIQUE,
            embedding_model_name TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            embedding_schema_version TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(id)
        );

        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            routing_decision TEXT NOT NULL,
            latency_ms INTEGER NOT NULL,
            result_ids TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )
    conn.commit()


def get_image_by_hash(conn: sqlite3.Connection, sha256_hash: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM images WHERE sha256_hash = ? LIMIT 1",
        (sha256_hash,),
    ).fetchone()


def get_image_by_id(conn: sqlite3.Connection, image_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM images WHERE id = ? LIMIT 1", (image_id,)).fetchone()


def get_images_by_ids(conn: sqlite3.Connection, ids: list[str]) -> dict[str, sqlite3.Row]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT * FROM images WHERE id IN ({placeholders})", ids).fetchall()
    return {str(row["id"]): row for row in rows}


def upsert_image_metadata(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    now = utc_now_iso()
    payload = {
        "id": row["id"],
        "file_path": row["file_path"],
        "sha256_hash": row["sha256_hash"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "caption": row.get("caption", ""),
        "summary": row.get("summary", ""),
        "tags": json_dumps(row.get("tags", [])),
        "ocr_structured": json_dumps(row.get("ocr_structured", [])),
        "ocr_confidence_avg": float(row.get("ocr_confidence_avg", 0.0)),
        "schema_version": row["schema_version"],
        "embedding_model_clip": row["embedding_model_clip"],
        "embedding_model_text": row["embedding_model_text"],
        "embedding_dimension_clip": int(row["embedding_dimension_clip"]),
        "embedding_dimension_text": int(row["embedding_dimension_text"]),
        "embedding_schema_version_clip": row["embedding_schema_version_clip"],
        "embedding_schema_version_text": row["embedding_schema_version_text"],
        "text_payload_hash": row.get("text_payload_hash", ""),
        "clip_content_hash": row.get("clip_content_hash", ""),
        "is_stale": int(row.get("is_stale", 0)),
        "created_at": row.get("created_at", now),
        "updated_at": now,
    }
    conn.execute(
        """
        INSERT INTO images (
            id,file_path,sha256_hash,width,height,caption,summary,tags,ocr_structured,ocr_confidence_avg,
            schema_version,embedding_model_clip,embedding_model_text,embedding_dimension_clip,
            embedding_dimension_text,embedding_schema_version_clip,embedding_schema_version_text,
            text_payload_hash,clip_content_hash,is_stale,created_at,updated_at
        ) VALUES (
            :id,:file_path,:sha256_hash,:width,:height,:caption,:summary,:tags,:ocr_structured,:ocr_confidence_avg,
            :schema_version,:embedding_model_clip,:embedding_model_text,:embedding_dimension_clip,
            :embedding_dimension_text,:embedding_schema_version_clip,:embedding_schema_version_text,
            :text_payload_hash,:clip_content_hash,:is_stale,:created_at,:updated_at
        )
        ON CONFLICT(id) DO UPDATE SET
            file_path=excluded.file_path,
            sha256_hash=excluded.sha256_hash,
            width=excluded.width,
            height=excluded.height,
            caption=excluded.caption,
            summary=excluded.summary,
            tags=excluded.tags,
            ocr_structured=excluded.ocr_structured,
            ocr_confidence_avg=excluded.ocr_confidence_avg,
            schema_version=excluded.schema_version,
            embedding_model_clip=excluded.embedding_model_clip,
            embedding_model_text=excluded.embedding_model_text,
            embedding_dimension_clip=excluded.embedding_dimension_clip,
            embedding_dimension_text=excluded.embedding_dimension_text,
            embedding_schema_version_clip=excluded.embedding_schema_version_clip,
            embedding_schema_version_text=excluded.embedding_schema_version_text,
            text_payload_hash=excluded.text_payload_hash,
            clip_content_hash=excluded.clip_content_hash,
            is_stale=excluded.is_stale,
            updated_at=excluded.updated_at
        """,
        payload,
    )


def upsert_vector_metadata(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    image_id: str,
    vector_id: str,
    model_name: str,
    dimension: int,
    schema_version: str,
) -> None:
    now = utc_now_iso()
    conn.execute(
        f"""
        INSERT INTO {table_name} (id,image_id,embedding_model_name,embedding_dimension,embedding_schema_version,updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(image_id) DO UPDATE SET
            id=excluded.id,
            embedding_model_name=excluded.embedding_model_name,
            embedding_dimension=excluded.embedding_dimension,
            embedding_schema_version=excluded.embedding_schema_version,
            updated_at=excluded.updated_at
        """,
        (vector_id, image_id, model_name, int(dimension), schema_version, now),
    )


def log_search(conn: sqlite3.Connection, query: str, routing_decision: str, latency_ms: int, result_ids: list[str]) -> None:
    conn.execute(
        "INSERT INTO search_logs (query,routing_decision,latency_ms,result_ids,timestamp) VALUES (?,?,?,?,?)",
        (query, routing_decision, int(latency_ms), json_dumps(result_ids), utc_now_iso()),
    )
    conn.commit()


def list_all_images(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM images ORDER BY created_at ASC").fetchall()


def list_stale_images(conn: sqlite3.Connection, cfg: StackConfig) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM images
        WHERE is_stale = 1
           OR embedding_model_clip != ?
           OR embedding_model_text != ?
           OR embedding_dimension_clip != ?
           OR embedding_dimension_text != ?
           OR embedding_schema_version_clip != ?
           OR embedding_schema_version_text != ?
        ORDER BY created_at ASC
        """,
        (
            cfg.clip_model_name,
            cfg.text_model_name,
            cfg.clip_dimension,
            cfg.text_dimension,
            cfg.clip_schema_version,
            cfg.text_schema_version,
        ),
    ).fetchall()


def mark_stale_if_versions_mismatch(conn: sqlite3.Connection, cfg: StackConfig) -> int:
    cur = conn.execute(
        """
        UPDATE images
        SET is_stale = 1, updated_at = ?
        WHERE embedding_model_clip != ?
           OR embedding_model_text != ?
           OR embedding_dimension_clip != ?
           OR embedding_dimension_text != ?
           OR embedding_schema_version_clip != ?
           OR embedding_schema_version_text != ?
        """,
        (
            utc_now_iso(),
            cfg.clip_model_name,
            cfg.text_model_name,
            cfg.clip_dimension,
            cfg.text_dimension,
            cfg.clip_schema_version,
            cfg.text_schema_version,
        ),
    )
    conn.commit()
    return cur.rowcount
