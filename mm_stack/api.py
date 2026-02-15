from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import StackConfig
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


def reembed_all(cfg: StackConfig | None = None) -> dict[str, int]:
    return reembed_impl(cfg)


def evaluate(cfg: StackConfig | None = None, fixture_path: str | None = None) -> dict[str, Any]:
    return evaluate_impl(cfg, fixture_path)
