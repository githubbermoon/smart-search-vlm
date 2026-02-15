from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .config import StackConfig
from .search_engine import MultimodalSearchEngine
from .utils import utc_now_iso


def ensure_eval_fixture(cfg: StackConfig) -> Path:
    fixture = cfg.stack_root / "mm_stack" / "evaluation" / "benchmark_cases.json"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    if fixture.exists():
        return fixture

    template = {
        "schema_version": "eval-v1",
        "created_at": utc_now_iso(),
        "note": "Fill with 20 benchmark cases. relevant_image_ids should contain image UUIDs from images table.",
        "cases": [
            {
                "id": f"case_{i:02d}",
                "query": "",
                "relevant_image_ids": [],
            }
            for i in range(1, 21)
        ],
    }
    fixture.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    return fixture


def _precision_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    top = predicted[:k]
    if not top:
        return 0.0
    hits = sum(1 for x in top if x in relevant)
    return hits / float(k)


def _recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = predicted[:k]
    hits = sum(1 for x in top if x in relevant)
    return hits / float(len(relevant))


def evaluate(cfg: StackConfig | None = None, fixture_path: str | None = None) -> dict[str, Any]:
    cfg = cfg or StackConfig()
    fixture = Path(fixture_path) if fixture_path else ensure_eval_fixture(cfg)
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])

    if len(cases) < 20:
        raise RuntimeError("Evaluation harness expects at least 20 test cases")

    engine = MultimodalSearchEngine(cfg)
    metrics: dict[str, dict[str, list[float]]] = {
        "clip": {"precision@5": [], "recall@10": [], "avg_similarity": []},
        "text": {"precision@5": [], "recall@10": [], "avg_similarity": []},
        "hybrid": {"precision@5": [], "recall@10": [], "avg_similarity": []},
    }

    for case in cases:
        query = str(case.get("query", "")).strip()
        relevant = {str(x) for x in case.get("relevant_image_ids", [])}
        if not query:
            continue

        for mode in ("clip", "text", "hybrid"):
            response = engine.search_forced_mode(query=query, mode=mode, top_k=10)
            ids = [str(r.get("image_id")) for r in response.results]
            sims = [float(r.get("score", 0.0)) for r in response.results]
            metrics[mode]["precision@5"].append(_precision_at_k(ids, relevant, 5))
            metrics[mode]["recall@10"].append(_recall_at_k(ids, relevant, 10))
            metrics[mode]["avg_similarity"].append(mean(sims) if sims else 0.0)

    summary: dict[str, dict[str, float]] = {}
    for mode, vals in metrics.items():
        summary[mode] = {
            "precision@5": round(mean(vals["precision@5"]) if vals["precision@5"] else 0.0, 4),
            "recall@10": round(mean(vals["recall@10"]) if vals["recall@10"] else 0.0, 4),
            "avg_similarity": round(mean(vals["avg_similarity"]) if vals["avg_similarity"] else 0.0, 4),
        }

    return {
        "fixture": str(fixture),
        "cases_count": len(cases),
        "summary": summary,
        "notes": [
            "Hybrid uses normalized per-index scores before weighted fusion.",
            "Future hook: add cross-modal reranking with VLM over top-N candidates.",
            "Future hook: dynamic routing weights based on intent confidence.",
            "Future hook: multilingual OCR adapters and tokenizer normalization.",
            "Future hook: swap CLIP ViT-B/32 to ViT-B/16 with re-embedding migration.",
        ],
    }
