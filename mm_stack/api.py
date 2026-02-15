from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .config import StackConfig
from .chat import MultimodalChat
from .evaluation import evaluate as evaluate_impl
from .ingestion import MultimodalIngestor
from .reembed import reembed_all as reembed_impl
from .search_engine import MultimodalSearchEngine


def ingest_image(image_path: str, *, safe_reprocess: bool = False, cfg: StackConfig | None = None) -> dict[str, Any]:
    engine = MultimodalIngestor(cfg)
    return engine.ingest_image(Path(image_path), safe_reprocess=safe_reprocess)


def search(query: str, *, image_path: str | None = None, top_k: int = 10, cfg: StackConfig | None = None) -> dict[str, Any]:
    engine = MultimodalSearchEngine(cfg)
    response = engine.search(query=query, image_path=image_path, top_k=top_k)
    return response.to_dict()


def chat(query: str, top_k: int = 3, cfg: StackConfig | None = None) -> dict[str, Any]:
    chat_engine = MultimodalChat(cfg)
    response = chat_engine.chat(query=query, top_k=top_k)
    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "grounded_score": response.grounded_score,
    }

def stream_chat(query: str, top_k: int = 5, cfg: StackConfig | None = None):
    # Generator wrapper
    c = MultimodalChat(cfg)
    return c.stream_chat(query, top_k)

def explain(query: str, cfg: StackConfig | None = None) -> dict[str, Any]:
    from .search_engine import MultimodalSearchEngine
    se = MultimodalSearchEngine(cfg)
    return se.explain(query)

def compare(query: str, cfg: StackConfig | None = None) -> Any:
    from .compare import Comparator
    comp = Comparator(cfg)
    return comp.compare(query)


def reembed_all(cfg: StackConfig | None = None) -> dict[str, int]:
    return reembed_impl(cfg)


def evaluate(cfg: StackConfig | None = None, fixture_path: str | None = None) -> dict[str, Any]:
    return evaluate_impl(cfg, fixture_path)


# ── Index-in-Place API ──

def ingest_path(target: str, *, safe_reprocess: bool = False, cfg: StackConfig | None = None) -> dict[str, Any]:
    engine = MultimodalIngestor(cfg)
    return engine.ingest_path(target, safe_reprocess=safe_reprocess)


def rescan(cfg: StackConfig | None = None) -> dict[str, Any]:
    engine = MultimodalIngestor(cfg)
    return engine.rescan_stale()


def rescan_watched(cfg: StackConfig | None = None) -> dict[str, Any]:
    engine = MultimodalIngestor(cfg)
    return engine.rescan_watched()


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
