#!/Users/pranjal/garage/smart_stack/.venv/bin/python3
"""Smart Search CLI for the Level 2 Local Intelligence Stack."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

HOME = Path.home()
VAULT_PATH = HOME / "Pranjal-Obs" / "clawd"
SQLITE_PATH = VAULT_PATH / "smart_stack.db"
LANCEDB_PATH = VAULT_PATH / "vectors.lance"
DEFAULT_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_ID = os.getenv("SMART_STACK_EMBED_MODEL", DEFAULT_EMBED_MODEL_ID)
IMAGE_LANCEDB_TABLE = "image_embeddings"
NOTE_LANCEDB_TABLE = "note_embeddings"

QUERY_EXPANSIONS: dict[str, list[str]] = {
    "receipt": ["invoice", "bill", "payment", "transaction", "purchase"],
    "receipts": ["invoice", "bill", "payment", "transaction", "purchase"],
    "restaurant": ["meal", "food", "cafe", "dining"],
    "coffee": ["cafe", "latte", "drink", "beverage"],
    "document": ["paper", "note", "page", "text"],
    "screenshot": ["screen capture", "ui", "app", "display"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local Smart Stack embeddings.")
    parser.add_argument("query", nargs="+", help="Search query text")
    parser.add_argument("-n", "--top-k", type=int, default=5, help="Top N matches to return (default: 5)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum similarity score in [-1.0, 1.0] to keep (default: 0.0)",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL_ID,
        help=f"Embedding model id for query vector (default: {EMBED_MODEL_ID})",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Disable simple rule-based query expansion",
    )
    parser.add_argument(
        "--no-notes",
        action="store_true",
        help="Search only image embeddings and skip note embeddings",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON for integrations",
    )
    parser.add_argument("--open", action="store_true", help="Open the top result after search")
    parser.add_argument(
        "--open-app",
        choices=["obsidian", "finder"],
        default="obsidian",
        help="When using --open, open with Obsidian (default) or reveal in Finder",
    )
    return parser.parse_args()


def expand_query(query: str) -> str:
    lowered = query.lower()
    extra_terms: list[str] = []
    for keyword, expansions in QUERY_EXPANSIONS.items():
        if keyword in lowered:
            for term in expansions:
                if term not in extra_terms:
                    extra_terms.append(term)
    if not extra_terms:
        return query
    return f"{query}\nrelated terms: {', '.join(extra_terms)}"


def requires_trust_remote_code(model_id: str) -> bool:
    lowered = (model_id or "").lower()
    return "nomic-ai/nomic-embed-text" in lowered


def prepare_embed_input(text: str, model_id: str, is_query: bool) -> str:
    payload = text.strip()
    lowered = (model_id or "").lower()
    if "nomic-embed-text" in lowered:
        prefix = "search_query: " if is_query else "search_document: "
        return f"{prefix}{payload}"
    if "e5" in lowered:
        prefix = "query: " if is_query else "passage: "
        return f"{prefix}{payload}"
    return payload


def table_for_embed_model(base_table: str, model_id: str) -> str:
    if model_id == DEFAULT_EMBED_MODEL_ID:
        return base_table
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    slug = slug[:64] if slug else "custom"
    return f"{base_table}__{slug}"


def embed_text(text: str, embed_model_id: str) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = None
    try:
        model = SentenceTransformer(
            embed_model_id,
            trust_remote_code=requires_trust_remote_code(embed_model_id),
        )
        payload = prepare_embed_input(text, embed_model_id, is_query=True)
        vector = model.encode(payload, normalize_embeddings=True)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]
        return [float(x) for x in vector]
    finally:
        if model is not None:
            del model
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()


def list_table_names(db: Any) -> list[str]:
    try:
        raw = db.list_tables()
    except Exception:
        try:
            return [str(x) for x in db.table_names()]
        except Exception:
            return []

    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        tables = raw.get("tables")
        if isinstance(tables, list):
            return [str(x) for x in tables]
    if hasattr(raw, "tables"):
        tables = getattr(raw, "tables")
        if isinstance(tables, list):
            return [str(x) for x in tables]
    try:
        return [str(x) for x in raw]
    except Exception:
        return []


def search_table(db: Any, table_name: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    table = db.open_table(table_name)
    search = table.search(query_vector)
    try:
        search = search.metric("cosine")
    except Exception:
        pass
    search = search.limit(top_k)

    if hasattr(search, "to_list"):
        rows = search.to_list()
    else:
        rows = search.to_pandas().to_dict(orient="records")

    return [dict(row) for row in rows]


def fetch_image_metadata(file_hashes: list[str]) -> dict[str, dict[str, Any]]:
    if not file_hashes:
        return {}

    placeholders = ",".join("?" for _ in file_hashes)
    sql = (
        "SELECT file_hash, filename, caption, tags, obsidian_path "
        f"FROM processed_images WHERE file_hash IN ({placeholders})"
    )
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, file_hashes).fetchall()

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        tags_raw = row["tags"] or "[]"
        try:
            parsed = json.loads(tags_raw)
            if isinstance(parsed, list):
                tags = [str(x) for x in parsed]
            else:
                tags = [str(parsed)]
        except json.JSONDecodeError:
            tags = [t.strip() for t in str(tags_raw).split(",") if t.strip()]

        out[str(row["file_hash"])] = {
            "filename": row["filename"] or "",
            "caption": row["caption"] or "",
            "tags": tags,
            "obsidian_path": row["obsidian_path"] or "",
        }
    return out


def distance_to_similarity(distance: Any) -> str:
    try:
        d = float(distance)
    except Exception:
        return "-"
    sim = max(-1.0, min(1.0, 1.0 - d))
    return f"{sim:.4f}"


def similarity_value(distance: Any) -> float:
    try:
        d = float(distance)
    except Exception:
        return -1.0
    return max(-1.0, min(1.0, 1.0 - d))


def snippet(text: str, limit: int = 110) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def render_results(rows: list[dict[str, Any]]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Smart Search Results")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Source", style="magenta")
        table.add_column("Filename", style="bold")
        table.add_column("Caption / Snippet")
        table.add_column("Tags")
        table.add_column("Score", justify="right")
        for i, row in enumerate(rows, start=1):
            table.add_row(
                str(i),
                row["source"],
                row["filename"],
                snippet(row["caption"]),
                ", ".join(row["tags"]) if row["tags"] else "-",
                row["score"],
            )
        console.print(table)
        return
    except Exception:
        pass

    print("\nSmart Search Results")
    print("=" * 80)
    for i, row in enumerate(rows, start=1):
        print(f"{i}. [{row['source']}] {row['filename']} | score={row['score']}")
        print(f"   caption: {snippet(row['caption'])}")
        print(f"   tags: {', '.join(row['tags']) if row['tags'] else '-'}")
        print()


def open_top_result(path: str, app: str) -> None:
    target = Path(path).expanduser()
    if not target.exists():
        print(f"[WARN] Top result path not found: {target}")
        return

    if app == "finder":
        subprocess.run(["open", "-R", str(target)], check=False)
        print(f"[OPEN] Revealed in Finder: {target}")
        return

    proc = subprocess.run(["open", "-a", "Obsidian", str(target)], check=False)
    if proc.returncode != 0:
        subprocess.run(["open", str(target)], check=False)
    print(f"[OPEN] Opened: {target}")


def as_distance(row: dict[str, Any]) -> float:
    try:
        return float(row.get("_distance"))
    except Exception:
        return math.inf


def main() -> None:
    args = parse_args()

    query = " ".join(args.query).strip()
    if not query:
        raise SystemExit("Query cannot be empty.")

    query_payload = query if args.no_expand else expand_query(query)
    vector = embed_text(query_payload, args.embed_model)
    min_score = max(-1.0, min(1.0, float(args.min_score)))

    import lancedb

    db = lancedb.connect(str(LANCEDB_PATH))
    available = set(list_table_names(db))

    image_table = table_for_embed_model(IMAGE_LANCEDB_TABLE, args.embed_model)
    note_table = table_for_embed_model(NOTE_LANCEDB_TABLE, args.embed_model)

    merged_rows: list[dict[str, Any]] = []
    searched_sources: list[str] = []

    if image_table in available:
        rows = search_table(db, image_table, vector, max(1, args.top_k))
        for row in rows:
            row["_source"] = "image"
        merged_rows.extend(rows)
        searched_sources.append("image")

    if (not args.no_notes) and (note_table in available):
        rows = search_table(db, note_table, vector, max(1, args.top_k))
        for row in rows:
            row["_source"] = "note"
        merged_rows.extend(rows)
        searched_sources.append("note")

    if not merged_rows:
        if not searched_sources:
            msg = ["No searchable table found for this embedding model."]
            msg.append(f"Expected image table: {image_table}")
            if not args.no_notes:
                msg.append(f"Expected note table: {note_table}")
            raise SystemExit("\n".join(msg))
        print("No matches found.")
        return

    merged_rows.sort(key=as_distance)
    merged_rows = [row for row in merged_rows if similarity_value(row.get("_distance")) >= min_score]
    merged_rows = merged_rows[: max(1, args.top_k)]
    if not merged_rows:
        print(f"No matches found with min-score >= {min_score:.2f}.")
        return

    image_hashes: list[str] = []
    for row in merged_rows:
        if row.get("_source") != "image":
            continue
        fh = row.get("file_hash") or row.get("id")
        if fh:
            image_hashes.append(str(fh))
    image_meta = fetch_image_metadata(image_hashes)

    final_rows: list[dict[str, Any]] = []
    for row in merged_rows:
        source = str(row.get("_source") or "unknown")
        if source == "image":
            fh = str(row.get("file_hash") or row.get("id") or "")
            meta = image_meta.get(fh, {})
            final_rows.append(
                {
                    "source": "image",
                    "filename": str(meta.get("filename") or row.get("filename") or "unknown"),
                    "caption": str(meta.get("caption") or ""),
                    "tags": meta.get("tags") or [],
                    "score": distance_to_similarity(row.get("_distance")),
                    "obsidian_path": str(meta.get("obsidian_path") or row.get("obsidian_path") or ""),
                }
            )
            continue

        note_path = Path(str(row.get("note_path") or "").strip())
        if note_path.is_absolute():
            resolved = note_path
        else:
            resolved = VAULT_PATH / note_path
        note_title = str(row.get("note_title") or resolved.stem or "note")
        chunk_text = str(row.get("chunk_text") or "")
        final_rows.append(
            {
                "source": "note",
                "filename": str(note_path) if str(note_path) else note_title,
                "caption": chunk_text or note_title,
                "tags": ["note"],
                "score": distance_to_similarity(row.get("_distance")),
                "obsidian_path": str(resolved),
            }
        )

    if args.json:
        payload = json.dumps(
            {
                "query": query,
                "embed_model": args.embed_model,
                "top_k": max(1, args.top_k),
                "min_score": min_score,
                "results": final_rows,
            },
            ensure_ascii=False,
        )
        print(f"@@SMARTSTACK_JSON@@{payload}")
    else:
        render_results(final_rows)

    if args.open and final_rows:
        open_top_result(final_rows[0]["obsidian_path"], args.open_app)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
