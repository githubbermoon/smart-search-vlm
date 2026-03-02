import unittest

from mm_stack.config import StackConfig
from mm_stack.retrieval_confidence import compute_confidence


class RetrievalConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = StackConfig()

    def test_high_confidence_case(self):
        rows = [
            {"score": 0.92, "caption": "mother and child portrait", "summary": "", "tags": ["mother", "family"]},
            {"score": 0.40, "caption": "random vehicle", "summary": "", "tags": ["car"]},
        ]
        out = compute_confidence("mother portrait", rows, rerank_applied=True, pre_rerank_top1=0.70, cfg=self.cfg)
        self.assertEqual(out["confidence_band"], "high")
        self.assertFalse(out["abstain_recommended"])

    def test_medium_confidence_verify_band(self):
        rows = [
            {"score": 0.70, "caption": "mother at table", "summary": "family dinner", "tags": ["family"]},
            {"score": 0.58, "caption": "people in restaurant", "summary": "", "tags": ["dining"]},
        ]
        out = compute_confidence("mother", rows, rerank_applied=False, pre_rerank_top1=0.70, cfg=self.cfg)
        self.assertEqual(out["confidence_band"], "medium")
        self.assertFalse(out["abstain_recommended"])

    def test_low_confidence_abstain_case(self):
        rows = [
            {"score": 0.44, "caption": "red hyundai suv", "summary": "vehicle parked", "tags": ["car"]},
            {"score": 0.42, "caption": "motorbike near road", "summary": "", "tags": ["bike"]},
        ]
        out = compute_confidence("mother", rows, rerank_applied=False, pre_rerank_top1=0.44, cfg=self.cfg)
        self.assertEqual(out["confidence_band"], "low")
        self.assertTrue(out["abstain_recommended"])
        self.assertTrue(bool(out["abstain_reason"]))


if __name__ == "__main__":
    unittest.main()
