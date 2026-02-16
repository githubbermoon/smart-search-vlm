import unittest

from mm_stack.query_normalization import (
    combined_rank,
    fuzzy_match_score,
    normalize_query,
)


class QueryNormalizationTests(unittest.TestCase):
    def test_normalize_query_shape(self):
        q = normalize_query("  Jwellery, Reciept!! ")
        self.assertEqual(q.raw_query, "  Jwellery, Reciept!! ")
        self.assertEqual(q.normalized_query, "jwellery, reciept!!")
        self.assertIn("jwellery", q.tokens_normalized)
        self.assertIn("reciept", q.tokens_normalized)

    def test_fuzzy_match_jewelry_typo(self):
        s = fuzzy_match_score(["jwellery"], "woman adorned with jewelry and gold necklace")
        self.assertGreaterEqual(s, 0.84)

    def test_fuzzy_match_receipt_typo(self):
        s = fuzzy_match_score(["reciept"], "receipt total amount 120")
        self.assertGreaterEqual(s, 0.84)

    def test_fuzzy_match_environment_typo(self):
        s = fuzzy_match_score(["enviroment"], "environment policy and climate report")
        self.assertGreaterEqual(s, 0.84)

    def test_combined_rank_promotes_fuzzy_relevant_candidate(self):
        query = normalize_query("jwellery")
        rows = [
            {
                "file_path": "/tmp/a.jpg",
                "caption": "three men standing outdoors",
                "summary": "business casual photo",
                "tags": ["men", "lanyard"],
                "ocr_structured": "[]",
                "score": 0.90,
            },
            {
                "file_path": "/tmp/b.jpg",
                "caption": "woman with horse",
                "summary": "woman adorned with jewelry and golden headpiece",
                "tags": ["woman", "horse"],
                "ocr_structured": "[]",
                "score": 0.80,
            },
        ]

        ranked = combined_rank(rows, query, alpha=0.8, beta=0.2, fuzzy_threshold=0.84)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["file_path"], "/tmp/b.jpg")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

    def test_soundex_guard_avoids_false_check_couch_match(self):
        s = fuzzy_match_score(
            ["white", "check", "shirt"],
            "a man sitting on a couch in a white shirt",
        )
        self.assertLess(s, 0.9)


if __name__ == "__main__":
    unittest.main()
