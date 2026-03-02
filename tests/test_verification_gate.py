import unittest

from mm_stack.intent_types import QueryIntent
from mm_stack.verification import should_verify


class VerificationGateTests(unittest.TestCase):
    def _intent(self, *, constrained: bool) -> QueryIntent:
        return QueryIntent(
            raw_query="red car next to bike",
            normalized_query="red car next to bike",
            tokens_raw=["red", "car", "next", "to", "bike"],
            tokens_normalized=["red", "car", "next", "to", "bike"],
            query_type="constrained" if constrained else "generic",
            policy_confidence_score=0.9,
            retrieval_terms=["red", "car", "bike"],
            relation_terms=["next to"] if constrained else [],
            attribute_terms=[],
            presence_terms=[],
            require_person=False,
            require_presence=False,
        )

    def test_verify_only_for_medium_band(self):
        intent = self._intent(constrained=True)
        ok, reason = should_verify(
            enabled=True,
            query_intent=intent,
            confidence_score=0.60,
            abstain_threshold=0.46,
            verify_threshold=0.70,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "medium_confidence")

    def test_skip_when_low_confidence(self):
        intent = self._intent(constrained=True)
        ok, reason = should_verify(
            enabled=True,
            query_intent=intent,
            confidence_score=0.30,
            abstain_threshold=0.46,
            verify_threshold=0.70,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "below_abstain_threshold")

    def test_skip_when_no_constraints(self):
        intent = self._intent(constrained=False)
        ok, reason = should_verify(
            enabled=True,
            query_intent=intent,
            confidence_score=0.60,
            abstain_threshold=0.46,
            verify_threshold=0.70,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "no_constraints")


if __name__ == "__main__":
    unittest.main()
