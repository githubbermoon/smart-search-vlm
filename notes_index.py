#!/Users/pranjal/garage/smart_stack/.venv/bin/python3
"""Index Obsidian markdown notes into LanceDB for semantic search."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HOME = Path.home()
VAULT_PATH = HOME / "Pranjal-Obs" / "clawd"
LANCEDB_PATH = VAULT_PATH / "vectors.lance"
SQLITE_PATH = VAULT_PATH / "smart_stack.db"

DEFAULT_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_ID = os.getenv("SMART_STACK_EMBED_MODEL", DEFAULT_EMBED_MODEL_ID)
NOTE_LANCEDB_TABLE = "note_embeddings"
SKIP_DIRS = {".obsidian", ".trash", ".git", ".smartenv", ".venv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index markdown notes into LanceDB.")
    p.add_argument("--vault", default=str(VAULT_PATH), help=f"Vault root directory (default: {VAULT_PATH})")
    p.add_argument(
        "--embed-model",
        default=EMBED_MODEL_ID,
        help=f"Embedding model id for notes (default: {EMBED_MODEL_ID})",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional max number of notes to process (0 = no limit)")
    p.add_argument("--chunk-chars", type=int, default=1400, help="Chunk size in characters")
    p.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    p.add_argument("--force", action="store_true", help="Re-index even if note hash has not changed")
    return p.parse_args()


def requires_trust_remote_code(model_id: str) -> bool:
    return "nomic-ai/nomic-embed-text" in (model_id or "").lower()


def prepare_embed_input(text: str, model_id: str, is_query: bool) -> str:
    payload = text.strip()
    lowered = (model_id or "").lower()
    if "nomic-embed-text" in lowered:
        return f"{'search_query' if is_query else 'search_document'}: {payload}"
    if "e5" in lowered:
        return f"{'query' if is_query else 'passage'}: {payload}"
    return payload


def table_for_embed_model(base_table: str, model_id: str) -> str:
    if model_id == DEFAULT_EMBED_MODEL_ID:
        return base_table
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    slug = slug[:64] if slug else "custom"
    return f"{base_table}__{slug}"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_markdown_files(vault: Path, limit: int = 0) -> list[Path]:
    out: list[Path] = []
    for path in sorted(vault.rglob("*.md")):
        rel_parts = set(path.relative_to(vault).parts)
        if rel_parts.intersection(SKIP_DIRS):
            continue
        out.append(path)
        if limit > 0 and len(out) >= limit:
            break
    return out


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) <= chunk_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_chars:
            current = para
            continue
        start = 0
        step = max(1, chunk_chars - max(0, overlap))
        while start < len(para):
            piece = para[start : start + chunk_chars].strip()
            if piece:
                chunks.append(piece)
            if start + chunk_chars >= len(para):
                break
            start += step
        current = ""
    if current:
        chunks.append(current)
    return chunks


def build_sqlite() -> sqlite3.Connection:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_notes (
            note_path TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            embed_model TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_indexed_notes_hash ON indexed_notes(file_hash)")
    conn.commit()
    return conn


def read_indexed_hash(conn: sqlite3.Connection, note_path: str, embed_model: str) -> str | None:
    row = conn.execute(
        "SELECT file_hash FROM indexed_notes WHERE note_path = ? AND embed_model = ? LIMIT 1",
        (note_path, embed_model),
    ).fetchone()
    if not row:
        return None
    return str(row[0])


def upsert_indexed_note(
    conn: sqlite3.Connection,
    note_path: str,
    file_hash: str,
    chunk_count: int,
    embed_model: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO indexed_notes (note_path, file_hash, chunk_count, embed_model, indexed_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(note_path) DO UPDATE SET
            file_hash=excluded.file_hash,
            chunk_count=excluded.chunk_count,
            embed_model=excluded.embed_model,
            indexed_at=excluded.indexed_at
        """,
        (note_path, file_hash, chunk_count, embed_model, now),
    )
    conn.commit()


def embed_text(text: str, embedder: Any, embed_model_id: str) -> list[float]:
    payload = prepare_embed_input(text, embed_model_id, is_query=False)
    vector = embedder.encode(payload, normalize_embeddings=True)
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if isinstance(vector, list) and vector and isinstance(vector[0], list):
        vector = vector[0]
    return [float(x) for x in vector]


def index_notes(
    vault: Path,
    embed_model_id: str,
    limit: int,
    chunk_chars: int,
    overlap: int,
    force: bool,
) -> None:
    notes = collect_markdown_files(vault, limit=limit)
    if not notes:
        print(f"[INFO] No markdown notes found under {vault}")
        return

    from sentence_transformers import SentenceTransformer
    import lancedb

    conn = build_sqlite()
    db = lancedb.connect(str(LANCEDB_PATH))
    table_name = table_for_embed_model(NOTE_LANCEDB_TABLE, embed_model_id)
    try:
        table = db.open_table(table_name)
    except Exception:
        table = None
    embedder = SentenceTransformer(
        embed_model_id,
        trust_remote_code=requires_trust_remote_code(embed_model_id),
    )

    print(f"[INFO] Note table: {table_name}")
    print(f"[INFO] Embedding model: {embed_model_id}")
    print(f"[INFO] Notes discovered: {len(notes)}")

    processed = 0
    skipped = 0
    failed = 0
    try:
        for note_path in notes:
            try:
                rel_path = str(note_path.relative_to(vault))
                file_hash = sha256_file(note_path)
                existing_hash = read_indexed_hash(conn, rel_path, embed_model_id)
                if (not force) and existing_hash == file_hash:
                    skipped += 1
                    print(f"[SKIP] unchanged {rel_path}")
                    continue

                text = note_path.read_text(encoding="utf-8", errors="ignore")
                chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
                if not chunks:
                    skipped += 1
                    print(f"[SKIP] empty {rel_path}")
                    continue

                rows: list[dict[str, Any]] = []
                now = datetime.now(timezone.utc).isoformat()
                note_title = note_path.stem
                for idx, chunk in enumerate(chunks):
                    embed_payload = "\n".join([note_title, rel_path, chunk])
                    rows.append(
                        {
                            "id": f"{file_hash}:{idx}",
                            "file_hash": file_hash,
                            "note_path": rel_path,
                            "note_title": note_title,
                            "chunk_index": idx,
                            "chunk_text": chunk,
                            "embedding": embed_text(embed_payload, embedder, embed_model_id),
                            "processed_at": now,
                        }
                    )

                if table is not None:
                    escaped_path = rel_path.replace("'", "''")
                    try:
                        table.delete(f"note_path = '{escaped_path}'")
                    except Exception:
                        pass

                if table is None:
                    table = db.create_table(table_name, data=rows)
                else:
                    try:
                        table.add(rows)
                    except Exception as exc:
                        # Recover from older/bad schema (wrong vector width) by rebuilding table.
                        if "FixedSizeListType" in str(exc):
                            print(f"[WARN] Rebuilding note table schema: {table_name}")
                            try:
                                db.drop_table(table_name)
                            except Exception:
                                pass
                            table = db.create_table(table_name, data=rows)
                        else:
                            raise
                upsert_indexed_note(conn, rel_path, file_hash, len(rows), embed_model_id)
                processed += 1
                print(f"[OK] {rel_path} chunks={len(rows)}")
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {note_path}: {exc}")
    finally:
        conn.close()
        del embedder
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    print(f"[DONE] processed={processed} skipped={skipped} failed={failed}")


def main() -> None:
    args = parse_args()
    vault = Path(args.vault).expanduser()
    if not vault.exists():
        raise SystemExit(f"Vault not found: {vault}")
    LANCEDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    index_notes(
        vault=vault,
        embed_model_id=args.embed_model,
        limit=max(0, args.limit),
        chunk_chars=max(256, args.chunk_chars),
        overlap=max(0, min(args.chunk_overlap, args.chunk_chars - 64)),
        force=args.force,
    )


if __name__ == "__main__":
    main()
