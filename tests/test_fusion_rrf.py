import unittest

from mm_stack.fusion import weighted_rrf_fuse


class WeightedRrfFuseTests(unittest.TestCase):
    def test_deterministic_order_prefers_dual_signal_hits(self):
        keyword_rows = [
            {"image_id": "a", "score": 0.9},
            {"image_id": "b", "score": 0.8},
        ]
        semantic_rows = [
            {"image_id": "b", "score": 0.95},
            {"image_id": "c", "score": 0.6},
        ]
        fused = weighted_rrf_fuse(
            keyword_rows,
            semantic_rows,
            rrf_k=60,
            w_keyword=0.6,
            w_semantic=0.4,
        )
        self.assertEqual([row["image_id"] for row in fused], ["b", "a", "c"])

    def test_missing_side_rank_is_none_with_zero_side_score(self):
        keyword_rows = [{"image_id": "only_kw", "score": 0.7}]
        semantic_rows = [{"image_id": "only_sem", "score": 0.8}]
        fused = weighted_rrf_fuse(
            keyword_rows,
            semantic_rows,
            rrf_k=60,
            w_keyword=0.62,
            w_semantic=0.38,
        )
        row_kw = next(row for row in fused if row["image_id"] == "only_kw")
        row_sem = next(row for row in fused if row["image_id"] == "only_sem")
        self.assertEqual(row_kw["keyword_rank"], 1)
        self.assertIsNone(row_kw["semantic_rank"])
        self.assertGreater(row_kw["keyword_score"], 0.0)
        self.assertEqual(row_kw["semantic_score"], 0.0)

        self.assertEqual(row_sem["semantic_rank"], 1)
        self.assertIsNone(row_sem["keyword_rank"])
        self.assertGreater(row_sem["semantic_score"], 0.0)
        self.assertEqual(row_sem["keyword_score"], 0.0)

    def test_ties_are_stable_and_repeatable(self):
        keyword_rows = [
            {"image_id": "a", "score": 1.0},
            {"image_id": "b", "score": 1.0},
        ]
        semantic_rows = [
            {"image_id": "b", "score": 1.0},
            {"image_id": "a", "score": 1.0},
        ]
        fused_one = weighted_rrf_fuse(
            keyword_rows,
            semantic_rows,
            rrf_k=1,
            w_keyword=0.5,
            w_semantic=0.5,
        )
        fused_two = weighted_rrf_fuse(
            keyword_rows,
            semantic_rows,
            rrf_k=1,
            w_keyword=0.5,
            w_semantic=0.5,
        )
        self.assertEqual([row["image_id"] for row in fused_one], ["a", "b"])
        self.assertEqual([row["image_id"] for row in fused_one], [row["image_id"] for row in fused_two])


if __name__ == "__main__":
    unittest.main()
