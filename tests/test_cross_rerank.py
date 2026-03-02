import unittest

from mm_stack.cross_rerank import CrossEncoderReranker


class CrossRerankTests(unittest.TestCase):
    def test_deterministic_fused_ordering(self):
        reranker = CrossEncoderReranker("dummy-model")
        reranker.score_pairs = lambda query, texts, batch_size=16: [0.10, 0.95, 0.20]  # type: ignore[method-assign]
        rows = [
            {"image_id": "a", "score": 0.90, "caption": "car on road", "summary": "", "tags": []},
            {"image_id": "b", "score": 0.40, "caption": "mother and child", "summary": "", "tags": []},
            {"image_id": "c", "score": 0.30, "caption": "tree", "summary": "", "tags": []},
        ]
        out = reranker.rerank_rows("mother", rows, rerank_k=3, weight=0.5)
        self.assertTrue(out.debug["applied"])
        self.assertEqual([r["image_id"] for r in out.rows], ["b", "a", "c"])
        self.assertIn("cross_score", out.rows[0])
        self.assertIn("base_score_pre_cross", out.rows[0])
        self.assertIn("score_post_cross", out.rows[0])

    def test_rerank_k_truncation_only_touches_shortlist(self):
        reranker = CrossEncoderReranker("dummy-model")
        reranker.score_pairs = lambda query, texts, batch_size=16: [0.01, 0.01]  # type: ignore[method-assign]
        rows = [
            {"image_id": "a", "score": 0.95, "caption": "alpha", "summary": "", "tags": []},
            {"image_id": "b", "score": 0.90, "caption": "beta", "summary": "", "tags": []},
            {"image_id": "c", "score": 0.85, "caption": "gamma", "summary": "", "tags": []},
            {"image_id": "d", "score": 0.80, "caption": "delta", "summary": "", "tags": []},
        ]
        out = reranker.rerank_rows("alpha", rows, rerank_k=2, weight=1.0)
        touched = {r["image_id"] for r in out.rows if "cross_score" in r}
        self.assertEqual(touched, {"a", "b"})
        untouched = [r for r in out.rows if r["image_id"] in {"c", "d"}]
        self.assertTrue(all("cross_score" not in r for r in untouched))

    def test_missing_model_graceful_fallback(self):
        reranker = CrossEncoderReranker("broken-model")

        def _fail():
            raise RuntimeError("cannot load model")

        reranker._ensure_model = _fail  # type: ignore[method-assign]
        rows = [
            {"image_id": "a", "score": 0.8, "caption": "car", "summary": "", "tags": []},
            {"image_id": "b", "score": 0.7, "caption": "bike", "summary": "", "tags": []},
        ]
        out = reranker.rerank_rows("car", rows, rerank_k=2, weight=0.42)
        self.assertFalse(out.debug["applied"])
        self.assertEqual(out.debug["reason"], "model_error")
        self.assertEqual([r["image_id"] for r in out.rows], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
