import unittest

from mm_stack.intent_reranker import rerank_with_intent
from mm_stack.query_planner import parse_query


class SearchRerankTests(unittest.TestCase):
    def _rerank(self, query: str, rows: list[dict]):
        intent = parse_query(query)
        return rerank_with_intent(
            rows,
            intent,
            retrieval_weight=0.6,
            attribute_weight=0.25,
            relation_weight=0.15,
            required_entity_penalty=0.35,
            activity_boost=0.12,
            color_boost=0.2,
            pattern_boost=0.2,
            presence_required=True,
        )

    def test_presence_penalty_for_missing_person(self):
        rows = [
            {
                "image_id": "tray",
                "caption": "banana peel and yogurt on tray",
                "summary": "meal on table",
                "tags": ["food", "tray"],
                "score": 0.95,
            },
            {
                "image_id": "shirt_meal",
                "caption": "old man in white check shirt having meal",
                "summary": "person eating in restaurant",
                "tags": ["man", "white", "checkered", "shirt", "meal"],
                "score": 0.88,
            },
        ]
        ranked = self._rerank("white check shirt having meal", rows)
        self.assertEqual(ranked[0]["image_id"], "shirt_meal")

    def test_relation_boost(self):
        rows = [
            {
                "image_id": "independent",
                "caption": "red car on road and bike parked separately",
                "summary": "street scene with car and bike",
                "tags": ["car", "bike"],
                "score": 0.9,
                "relation_evidence": [],
            },
            {
                "image_id": "related",
                "caption": "blue car next to a bike",
                "summary": "vehicle relation clearly visible",
                "tags": ["car", "bike", "next to"],
                "score": 0.85,
                "relation_evidence": [{"relation": "next_to", "confidence": 0.9}],
            },
        ]
        ranked = self._rerank("car next to bike", rows)
        self.assertEqual(ranked[0]["image_id"], "related")

    def test_no_colorize_hijack(self):
        rows = [
            {
                "image_id": "noise",
                "caption": "sar image colorization model output",
                "summary": "paper on image colorize process",
                "tags": ["colorization", "remote sensing"],
                "score": 0.92,
            },
            {
                "image_id": "true_car",
                "caption": "blue car next to bike in parking lot",
                "summary": "car and bike together",
                "tags": ["blue", "car", "bike"],
                "score": 0.86,
                "relation_evidence": [{"relation": "next_to", "confidence": 0.8}],
            },
        ]
        ranked = self._rerank("car next to bike what color is car", rows)
        self.assertEqual(ranked[0]["image_id"], "true_car")


if __name__ == "__main__":
    unittest.main()
