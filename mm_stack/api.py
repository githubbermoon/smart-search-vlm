from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .config import StackConfig
from .chat import MultimodalChat
from .context_lens import ContextLensEngine
from .evaluation import evaluate as evaluate_impl
from .fs_watch import watch_live as watch_live_impl
from .ingestion import MultimodalIngestor
from .reembed import reembed_all as reembed_impl
from .search_engine import MultimodalSearchEngine
from .timeline import SemanticTimelineEngine


def ingest_image(
    image_path: str,
    *,
    safe_reprocess: bool = False,
    image_batch_size: int | None = None,
    cfg: StackConfig | None = None,
) -> dict[str, Any]:
    engine = MultimodalIngestor(cfg, image_batch_size=image_batch_size)
    return engine.ingest_image(Path(image_path), safe_reprocess=safe_reprocess)


def search(
    query: str,
    *,
    image_path: str | None = None,
    top_k: int = 10,
    mode: str | None = None,
    auto_strategy: str | None = None,
    verify: bool = False,
    semantic_fallback_threshold: int | None = None,
    cfg: StackConfig | None = None,
) -> dict[str, Any]:
    engine = MultimodalSearchEngine(cfg)
    response = engine.search(
        query=query,
        image_path=image_path,
        top_k=top_k,
        mode=mode,
        auto_strategy=auto_strategy,
        enable_verification=verify,
        semantic_fallback_threshold=semantic_fallback_threshold,
    )
    return response.to_dict()


def chat(
    query: str,
    top_k: int = 3,
    cfg: StackConfig | None = None,
    *,
    attached_image_id: str | None = None,
    attached_file_path: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chat_engine = MultimodalChat(cfg)
    response = chat_engine.chat(
        query=query,
        top_k=top_k,
        attached_image_id=attached_image_id,
        attached_file_path=attached_file_path,
        history=history,
    )
    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "grounded_score": response.grounded_score,
        "timings": response.timings,
    }

def stream_chat(
    query: str,
    top_k: int = 5,
    cfg: StackConfig | None = None,
    *,
    attached_image_id: str | None = None,
    attached_file_path: str | None = None,
    history: list[dict[str, Any]] | None = None,
):
    # Generator wrapper
    c = MultimodalChat(cfg)
    return c.stream_chat(
        query,
        top_k,
        attached_image_id=attached_image_id,
        attached_file_path=attached_file_path,
        history=history,
    )

def explain(query: str, cfg: StackConfig | None = None) -> dict[str, Any]:
    from .search_engine import MultimodalSearchEngine
    se = MultimodalSearchEngine(cfg)
    return se.explain(query)

def compare(query: str, cfg: StackConfig | None = None) -> Any:
    from .compare import Comparator
    comp = Comparator(cfg)
    return comp.compare(query)

def context_lens(
    *,
    image_id: str | None = None,
    file_path: str | None = None,
    top_k: int = 8,
    cfg: StackConfig | None = None,
) -> dict[str, Any]:
    engine = ContextLensEngine(cfg)
    return engine.context_lens(image_id=image_id, file_path=file_path, top_k=top_k)

def timeline(
    *,
    granularity: str = "month",
    query: str | None = None,
    limit: int = 240,
    cfg: StackConfig | None = None,
) -> dict[str, Any]:
    engine = SemanticTimelineEngine(cfg)
    return engine.timeline(granularity=granularity, query=query, limit=limit)


def photos_list(
    *,
    limit: int = 500,
    offset: int = 0,
    include_missing: bool = True,
    check_paths: bool = False,
    cfg: StackConfig | None = None,
) -> dict[str, Any]:
    from .db import connect_sqlite, ensure_schema

    conn = connect_sqlite(cfg or StackConfig())
    ensure_schema(conn)
    try:
        total_row = conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()
        total_indexed = int(total_row["c"] if total_row is not None else 0)
        rows = conn.execute(
            """
            SELECT id, file_path, caption, summary, tags, created_at, updated_at, is_stale
            FROM images
            ORDER BY datetime(created_at) DESC, created_at DESC
            LIMIT ? OFFSET ?
            """,
            (max(1, int(limit)), max(0, int(offset))),
        ).fetchall()

        effective_check_paths = bool(check_paths or (not include_missing))
        items: list[dict[str, Any]] = []
        for row in rows:
            file_path = str(row["file_path"] or "")
            exists_on_disk = True
            if effective_check_paths:
                exists_on_disk = Path(file_path).exists() if file_path else False
            if effective_check_paths and (not include_missing) and (not exists_on_disk):
                continue
            raw_tags = str(row["tags"] or "[]")
            tags: list[str]
            try:
                parsed_tags = json.loads(raw_tags)
                if isinstance(parsed_tags, list):
                    tags = [str(x) for x in parsed_tags if str(x).strip()]
                else:
                    tags = []
            except Exception:
                tags = [x.strip() for x in raw_tags.split(",") if x.strip()]

            items.append(
                {
                    "image_id": str(row["id"]),
                    "file_path": file_path,
                    "caption": str(row["caption"] or ""),
                    "summary": str(row["summary"] or ""),
                    "tags": tags,
                    "created_at": str(row["created_at"] or ""),
                    "updated_at": str(row["updated_at"] or ""),
                    "is_stale": bool(row["is_stale"]),
                    "exists_on_disk": exists_on_disk,
                }
            )
    finally:
        conn.close()

    return {
        "total_indexed": total_indexed,
        "returned": len(items),
        "limit": max(1, int(limit)),
        "offset": max(0, int(offset)),
        "include_missing": bool(include_missing),
        "path_checks_performed": bool(effective_check_paths),
        "items": items,
    }

def cluster_recalc(
    *,
    n_clusters: int = 20,
    auto_label: bool = False,
    cfg: StackConfig | None = None,
) -> dict[str, Any]:
    from .memory import MemoryManager

    mm = MemoryManager(cfg)
    out = mm.update_clusters(n_clusters=max(1, int(n_clusters)))
    if auto_label:
        out["labeled_count"] = mm.auto_label_clusters()
    return out


def cluster_label(cfg: StackConfig | None = None) -> dict[str, int]:
    from .memory import MemoryManager

    mm = MemoryManager(cfg)
    return {"labeled_count": mm.auto_label_clusters()}


def cluster_list(
    *,
    limit: int = 64,
    min_items: int = 1,
    cfg: StackConfig | None = None,
) -> list[dict[str, Any]]:
    from .memory import MemoryManager

    mm = MemoryManager(cfg)
    return mm.list_clusters(limit=max(1, int(limit)), min_items=max(0, int(min_items)))


def cluster_items(
    cluster_id: str,
    *,
    limit: int = 120,
    cfg: StackConfig | None = None,
) -> list[dict[str, Any]]:
    from .memory import MemoryManager

    mm = MemoryManager(cfg)
    return mm.get_cluster_items(cluster_id, limit=max(1, int(limit)))


def reembed_all(cfg: StackConfig | None = None) -> dict[str, int]:
    return reembed_impl(cfg)


def evaluate(cfg: StackConfig | None = None, fixture_path: str | None = None) -> dict[str, Any]:
    return evaluate_impl(cfg, fixture_path)


# ── Index-in-Place API ──

def ingest_path(
    target: str,
    *,
    safe_reprocess: bool = False,
    image_batch_size: int | None = None,
    cfg: StackConfig | None = None,
) -> dict[str, Any]:
    engine = MultimodalIngestor(cfg, image_batch_size=image_batch_size)
    return engine.ingest_path(target, safe_reprocess=safe_reprocess)


def rescan(*, image_batch_size: int | None = None, cfg: StackConfig | None = None) -> dict[str, Any]:
    engine = MultimodalIngestor(cfg, image_batch_size=image_batch_size)
    return engine.rescan_stale()


def rescan_watched(*, image_batch_size: int | None = None, cfg: StackConfig | None = None) -> dict[str, Any]:
    engine = MultimodalIngestor(cfg, image_batch_size=image_batch_size)
    return engine.rescan_watched()


def watch_live(
    *,
    hourly_refresh_min: int = 60,
    debounce_ms: int = 1200,
    move_grace_sec: float = 5.0,
    initial_refresh: bool = False,
    cfg: StackConfig | None = None,
) -> dict[str, Any]:
    return watch_live_impl(
        cfg=cfg,
        hourly_refresh_min=max(1, int(hourly_refresh_min)),
        debounce_ms=max(50, int(debounce_ms)),
        move_grace_sec=max(0.0, float(move_grace_sec)),
        initial_refresh=bool(initial_refresh),
    )


# ── Watched Folders API ──

def watch_add(path: str, cfg: StackConfig | None = None) -> dict[str, str]:
    from .db import connect_sqlite, ensure_schema, add_watched_folder
    conn = connect_sqlite(cfg or StackConfig())
    ensure_schema(conn)
    add_watched_folder(conn, path)
    conn.close()
    return {"status": "added", "path": path}


def watch_remove(path: str, cfg: StackConfig | None = None) -> dict[str, str]:
    from .db import connect_sqlite, ensure_schema, remove_watched_folder
    conn = connect_sqlite(cfg or StackConfig())
    ensure_schema(conn)
    remove_watched_folder(conn, path)
    conn.close()
    return {"status": "removed", "path": path}


def watch_toggle(path: str, cfg: StackConfig | None = None) -> dict[str, str]:
    from .db import connect_sqlite, ensure_schema, toggle_watched_folder
    conn = connect_sqlite(cfg or StackConfig())
    ensure_schema(conn)
    toggle_watched_folder(conn, path)
    conn.close()
    return {"status": "toggled", "path": path}


def watch_list(cfg: StackConfig | None = None) -> list[dict[str, Any]]:
    from .db import connect_sqlite, ensure_schema, list_watched_folders
    conn = connect_sqlite(cfg or StackConfig())
    ensure_schema(conn)
    result = list_watched_folders(conn)
    conn.close()
    return result


def exclude_add(pattern: str, cfg: StackConfig | None = None) -> dict[str, str]:
    from .db import connect_sqlite, ensure_schema, add_exclusion
    conn = connect_sqlite(cfg or StackConfig())
    ensure_schema(conn)
    add_exclusion(conn, pattern)
    conn.close()
    return {"status": "added", "pattern": pattern}


def exclude_remove(pattern: str, cfg: StackConfig | None = None) -> dict[str, str]:
    from .db import connect_sqlite, ensure_schema, remove_exclusion
    conn = connect_sqlite(cfg or StackConfig())
    ensure_schema(conn)
    remove_exclusion(conn, pattern)
    conn.close()
    return {"status": "removed", "pattern": pattern}


def exclude_list(cfg: StackConfig | None = None) -> list[dict[str, Any]]:
    from .db import connect_sqlite, ensure_schema, list_exclusions
    conn = connect_sqlite(cfg or StackConfig())
    ensure_schema(conn)
    result = list_exclusions(conn)
    conn.close()
    return result
