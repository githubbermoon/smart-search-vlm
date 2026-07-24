import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mm_stack.config import StackConfig
from mm_stack.evaluation import evaluate
from mm_stack.search_types import SearchResponse


class _DummyEngine:
    def __init__(self, _cfg: StackConfig):
        pass

    def search_forced_mode(self, query: str, mode: str, top_k: int = 10) -> SearchResponse:
        idx = query.split("_")[-1]
        relevant = f"img_{idx}"

        if mode == "hybrid":
            ids = [relevant] + [f"other_{idx}_{i}" for i in range(1, top_k)]
        elif mode == "text":
            ids = [f"other_{idx}_{i}" for i in range(1, min(3, top_k))] + [relevant]
            while len(ids) < top_k:
                ids.append(f"other_{idx}_{len(ids)+1}")
        else:
            ids = [f"other_{idx}_{i}" for i in range(1, top_k + 1)]

        results = [{"image_id": image_id, "score": max(0.0, 1.0 - (rank * 0.05))} for rank, image_id in enumerate(ids)]
        return SearchResponse(
            routing_mode=mode,
            routing_reason="dummy",
            latency_ms=1,
            results=results,
            normalization_explanation="",
            rerank_todo="",
        )


class EvaluationHarnessTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        cases = []
        for i in range(1, 21):
            idx = f"{i:02d}"
            cases.append(
                {
                    "id": f"case_{idx}",
                    "query": f"query_{idx}",
                    "relevant_image_ids": [f"img_{idx}"],
                }
            )
        payload = {
            "schema_version": "eval-v2",
            "created_at": "2026-03-17T00:00:00+00:00",
            "cases": cases,
        }
        path = root / "fixture.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_thresholds(self, root: Path, *, strict: bool) -> Path:
        payload = {
            "schema_version": "eval-thresholds-v1",
            "min_cases_with_relevance": 20,
            "min_summary": {
                "clip": {"precision@5": 0.0, "recall@10": 0.0},
                "text": {"precision@5": 0.05, "recall@10": 0.80},
                "hybrid": {"precision@5": 0.15, "recall@10": 0.95},
            },
            "min_case_hit_rate_at_10": {
                "clip": 0.0,
                "text": 0.95,
                "hybrid": 0.95,
            },
            "max_failed_cases_hybrid": 1,
        }
        if strict:
            payload["min_summary"]["hybrid"]["precision@5"] = 0.50
            payload["max_failed_cases_hybrid"] = 0

        path = root / "thresholds.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def test_evaluate_returns_per_case_and_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._write_fixture(root)
            thresholds = self._write_thresholds(root, strict=False)
            cfg = StackConfig(stack_root=root)

            with patch("mm_stack.evaluation.MultimodalSearchEngine", _DummyEngine):
                out = evaluate(cfg=cfg, fixture_path=str(fixture), thresholds_path=str(thresholds))

            self.assertEqual(out["cases_count"], 20)
            self.assertEqual(out["evaluated_cases"], 20)
            self.assertEqual(len(out["per_case"]), 20)
            self.assertTrue(out["regression"]["passed"])
            self.assertGreaterEqual(out["case_hit_rate_at_10"]["hybrid"], 0.95)

    def test_evaluate_regression_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._write_fixture(root)
            thresholds = self._write_thresholds(root, strict=True)
            cfg = StackConfig(stack_root=root)

            with patch("mm_stack.evaluation.MultimodalSearchEngine", _DummyEngine):
                out = evaluate(cfg=cfg, fixture_path=str(fixture), thresholds_path=str(thresholds))

            self.assertFalse(out["regression"]["passed"])
            failures = "\n".join(out["regression"]["failures"])
            self.assertIn("summary[hybrid][precision@5]", failures)


if __name__ == "__main__":
    unittest.main()
