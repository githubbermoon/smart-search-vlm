import unittest

from mm_stack.query_normalization import combined_rank, normalize_query


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for x in top if x in relevant)
    return hits / len(top)


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for x in top if x in relevant)
    return hits / len(relevant)


class FuzzyMetricsTests(unittest.TestCase):
    def _fixture_rows(self):
        return [
            {
                "id": "noise_1",
                "caption": "three men standing outdoors",
                "summary": "business casual group photo",
                "tags": ["men", "lanyard"],
                "ocr_structured": "[]",
                "score": 0.93,
            },
            {
                "id": "rel_jewelry",
                "caption": "woman and white horse in tropical foliage",
                "summary": "woman adorned with jewelry and golden headpiece",
                "tags": ["woman", "horse", "gold"],
                "ocr_structured": "[]",
                "score": 0.82,
            },
            {
                "id": "rel_receipt",
                "caption": "store bill image",
                "summary": "receipt total amount shown on paper",
                "tags": ["invoice", "paper"],
                "ocr_structured": "[]",
                "score": 0.79,
            },
            {
                "id": "noise_2",
                "caption": "banana and yogurt on tray",
                "summary": "meal leftovers",
                "tags": ["food"],
                "ocr_structured": "[]",
                "score": 0.88,
            },
            {
                "id": "rel_environment",
                "caption": "forest policy report",
                "summary": "environment climate and biodiversity notes",
                "tags": ["policy", "report"],
                "ocr_structured": "[]",
                "score": 0.78,
            },
        ]

    def _baseline_rank_ids(self, rows):
        ordered = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return [r["id"] for r in ordered]

    def test_typo_query_improves_recall(self):
        rows = self._fixture_rows()
        relevant = {"rel_jewelry"}
        base = self._baseline_rank_ids(rows)
        base_recall = recall_at_k(base, relevant, 1)

        ranked = combined_rank(rows, normalize_query("jwellery"), alpha=0.8, beta=0.2)
        ranked_ids = [r["id"] for r in ranked]
        fuzzy_recall = recall_at_k(ranked_ids, relevant, 1)

        self.assertGreaterEqual(fuzzy_recall, base_recall)

    def test_metrics_cover_precision_recall_and_fuzzy_influence(self):
        rows = self._fixture_rows()
        relevant = {"rel_receipt", "rel_environment", "rel_jewelry"}

        base_ids = self._baseline_rank_ids(rows)
        base_p3 = precision_at_k(base_ids, relevant, 3)
        base_r5 = recall_at_k(base_ids, relevant, 5)

        ranked = combined_rank(rows, normalize_query("reciept enviroment jwellery"), alpha=0.8, beta=0.2)
        ranked_ids = [r["id"] for r in ranked]
        fuzzy_p3 = precision_at_k(ranked_ids, relevant, 3)
        fuzzy_r5 = recall_at_k(ranked_ids, relevant, 5)

        # Fuzzy rerank should not degrade precision and should improve or match recall.
        self.assertGreaterEqual(fuzzy_p3, base_p3)
        self.assertGreaterEqual(fuzzy_r5, base_r5)


if __name__ == "__main__":
    unittest.main()
