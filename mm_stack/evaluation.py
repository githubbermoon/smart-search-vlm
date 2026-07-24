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
        "schema_version": "eval-v2",
        "created_at": utc_now_iso(),
        "note": (
            "Fill with at least 20 benchmark cases. "
            "Each case should include query + relevant_image_ids from the images table."
        ),
        "cases": [
            {
                "id": f"case_{i:02d}",
                "query": "",
                "relevant_image_ids": [],
                "notes": "",
            }
            for i in range(1, 21)
        ],
    }
    fixture.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    return fixture


def ensure_eval_thresholds(cfg: StackConfig) -> Path:
    thresholds = cfg.stack_root / "mm_stack" / "evaluation" / "thresholds.json"
    thresholds.parent.mkdir(parents=True, exist_ok=True)
    if thresholds.exists():
        return thresholds

    default_thresholds = {
        "schema_version": "eval-thresholds-v1",
        "updated_at": utc_now_iso(),
        "min_cases_with_relevance": 20,
        "min_summary": {
            "clip": {"precision@5": 0.02, "recall@10": 0.06},
            "text": {"precision@5": 0.06, "recall@10": 0.15},
            "hybrid": {"precision@5": 0.10, "recall@10": 0.30},
        },
        "min_case_hit_rate_at_10": {
            "clip": 0.20,
            "text": 0.45,
            "hybrid": 0.65,
        },
        "max_failed_cases_hybrid": 7,
    }
    thresholds.write_text(json.dumps(default_thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    return thresholds


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


def _response_results(response: Any) -> list[dict[str, Any]]:
    if hasattr(response, "results"):
        raw = getattr(response, "results")
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
    if isinstance(response, dict):
        raw = response.get("results", [])
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
    return []


def _load_thresholds(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _evaluate_regression(
    *,
    summary: dict[str, dict[str, float]],
    case_hit_rate_at_10: dict[str, float],
    cases_with_relevance: int,
    per_case: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []

    min_cases = int(thresholds.get("min_cases_with_relevance", 20) or 20)
    if cases_with_relevance < min_cases:
        failures.append(
            f"cases_with_relevance below threshold: got={cases_with_relevance} required>={min_cases}"
        )

    min_summary = thresholds.get("min_summary", {})
    if isinstance(min_summary, dict):
        for mode, mode_limits in min_summary.items():
            if not isinstance(mode_limits, dict):
                continue
            actual = summary.get(str(mode), {})
            for metric_name, metric_limit in mode_limits.items():
                try:
                    limit_val = float(metric_limit)
                except Exception:
                    continue
                actual_val = float(actual.get(str(metric_name), 0.0) or 0.0)
                if actual_val + 1e-9 < limit_val:
                    failures.append(
                        f"summary[{mode}][{metric_name}] below threshold: got={actual_val:.4f} required>={limit_val:.4f}"
                    )

    min_case_hit = thresholds.get("min_case_hit_rate_at_10", {})
    if isinstance(min_case_hit, dict):
        for mode, hit_limit in min_case_hit.items():
            try:
                limit_val = float(hit_limit)
            except Exception:
                continue
            actual_val = float(case_hit_rate_at_10.get(str(mode), 0.0) or 0.0)
            if actual_val + 1e-9 < limit_val:
                failures.append(
                    f"case_hit_rate_at_10[{mode}] below threshold: got={actual_val:.4f} required>={limit_val:.4f}"
                )

    max_failed_hybrid = int(thresholds.get("max_failed_cases_hybrid", 9999) or 9999)
    failed_hybrid = sum(1 for case in per_case if not bool(case.get("hybrid_hit@10", False)))
    if failed_hybrid > max_failed_hybrid:
        failures.append(
            f"failed_hybrid_cases above threshold: got={failed_hybrid} allowed<={max_failed_hybrid}"
        )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
    }


def evaluate(
    cfg: StackConfig | None = None,
    fixture_path: str | None = None,
    thresholds_path: str | None = None,
) -> dict[str, Any]:
    cfg = cfg or StackConfig()
    fixture = Path(fixture_path) if fixture_path else ensure_eval_fixture(cfg)
    thresholds_file = Path(thresholds_path) if thresholds_path else ensure_eval_thresholds(cfg)

    payload = json.loads(fixture.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise RuntimeError("Evaluation fixture 'cases' must be a list")
    if len(cases) < 20:
        raise RuntimeError("Evaluation harness expects at least 20 test cases")

    thresholds = _load_thresholds(thresholds_file)
    engine = MultimodalSearchEngine(cfg)
    metrics: dict[str, dict[str, list[float]]] = {
        "clip": {"precision@5": [], "recall@10": [], "avg_similarity": [], "hit@10": []},
        "text": {"precision@5": [], "recall@10": [], "avg_similarity": [], "hit@10": []},
        "hybrid": {"precision@5": [], "recall@10": [], "avg_similarity": [], "hit@10": []},
    }

    per_case: list[dict[str, Any]] = []
    evaluated_cases = 0

    for case in cases:
        query = str(case.get("query", "")).strip()
        relevant = {str(x).strip() for x in case.get("relevant_image_ids", []) if str(x).strip()}
        case_id = str(case.get("id", "")).strip() or f"case_{len(per_case) + 1:02d}"
        case_note = str(case.get("notes", "")).strip()

        if not query:
            continue

        if not relevant:
            per_case.append(
                {
                    "id": case_id,
                    "query": query,
                    "notes": case_note,
                    "skipped": True,
                    "reason": "empty_relevance",
                    "relevant_count": 0,
                }
            )
            continue

        evaluated_cases += 1
        mode_report: dict[str, Any] = {}
        hybrid_hit10 = False

        for mode in ("clip", "text", "hybrid"):
            response = engine.search_forced_mode(query=query, mode=mode, top_k=10)
            rows = _response_results(response)
            ids = [str(r.get("image_id", "")) for r in rows if str(r.get("image_id", ""))]
            sims = [float(r.get("score", 0.0) or 0.0) for r in rows]

            p5 = _precision_at_k(ids, relevant, 5)
            r10 = _recall_at_k(ids, relevant, 10)
            hit10 = any(x in relevant for x in ids[:10])
            if mode == "hybrid":
                hybrid_hit10 = hit10

            metrics[mode]["precision@5"].append(p5)
            metrics[mode]["recall@10"].append(r10)
            metrics[mode]["avg_similarity"].append(mean(sims) if sims else 0.0)
            metrics[mode]["hit@10"].append(1.0 if hit10 else 0.0)

            mode_report[mode] = {
                "precision@5": round(p5, 4),
                "recall@10": round(r10, 4),
                "avg_similarity": round(mean(sims) if sims else 0.0, 4),
                "hit@1": bool(ids[:1] and ids[0] in relevant),
                "hit@3": any(x in relevant for x in ids[:3]),
                "hit@5": any(x in relevant for x in ids[:5]),
                "hit@10": hit10,
                "top10_ids": ids[:10],
            }

        per_case.append(
            {
                "id": case_id,
                "query": query,
                "notes": case_note,
                "relevant_count": len(relevant),
                "relevant_image_ids": sorted(relevant),
                "hybrid_hit@10": hybrid_hit10,
                "modes": mode_report,
            }
        )

    summary: dict[str, dict[str, float]] = {}
    case_hit_rate_at_10: dict[str, float] = {}
    for mode, vals in metrics.items():
        summary[mode] = {
            "precision@5": round(mean(vals["precision@5"]) if vals["precision@5"] else 0.0, 4),
            "recall@10": round(mean(vals["recall@10"]) if vals["recall@10"] else 0.0, 4),
            "avg_similarity": round(mean(vals["avg_similarity"]) if vals["avg_similarity"] else 0.0, 4),
        }
        case_hit_rate_at_10[mode] = round(mean(vals["hit@10"]) if vals["hit@10"] else 0.0, 4)

    regression = _evaluate_regression(
        summary=summary,
        case_hit_rate_at_10=case_hit_rate_at_10,
        cases_with_relevance=evaluated_cases,
        per_case=[x for x in per_case if not x.get("skipped")],
        thresholds=thresholds,
    )

    return {
        "fixture": str(fixture),
        "thresholds": str(thresholds_file),
        "cases_count": len(cases),
        "evaluated_cases": evaluated_cases,
        "summary": summary,
        "case_hit_rate_at_10": case_hit_rate_at_10,
        "regression": regression,
        "per_case": per_case,
        "notes": [
            "Per-case report includes top-10 IDs and hit@k by mode.",
            "Regression section fails if metric thresholds are not met.",
            "Hybrid uses normalized per-index scores before weighted fusion.",
        ],
    }
