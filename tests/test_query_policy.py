import unittest

from mm_stack.config import StackConfig
from mm_stack.query_planner import parse_query
from mm_stack.query_policy import build_query_policy


class QueryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = StackConfig()

    def test_generic_policy_bounds(self):
        intent = parse_query("code")
        policy = build_query_policy(intent, self.cfg, requested_top_k=8)
        self.assertEqual(policy.query_type, "generic")
        self.assertLessEqual(abs(policy.similarity_gate - self.cfg.policy_base_similarity_gate), 0.07 + 1e-9)
        self.assertLessEqual(policy.top_k_multiplier, 2.0)
        self.assertEqual(policy.lexical_mode, "boost")

    def test_constrained_policy_enforces_lexical_and_presence(self):
        intent = parse_query("car next to bike")
        policy = build_query_policy(intent, self.cfg, requested_top_k=8)
        self.assertEqual(policy.query_type, "constrained")
        self.assertEqual(policy.lexical_mode, "enforce")
        self.assertTrue(policy.presence_required)

    def test_low_confidence_falls_back_to_generic(self):
        strict_cfg = StackConfig(policy_confidence_fallback_threshold=0.95)
        ambiguous = parse_query("white and old")
        policy = build_query_policy(ambiguous, strict_cfg, requested_top_k=8)
        self.assertTrue(policy.fallback_to_generic)
        self.assertEqual(policy.query_type, "generic")

    def test_adaptive_off_uses_fixed_baseline(self):
        cfg = StackConfig(adaptive_policy_enabled=False)
        intent = parse_query("code")
        policy = build_query_policy(intent, cfg, requested_top_k=8)
        self.assertEqual(policy.similarity_gate, cfg.policy_base_similarity_gate)
        self.assertEqual(policy.top_k_multiplier, 1.0)


if __name__ == "__main__":
    unittest.main()
