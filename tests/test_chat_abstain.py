import unittest

from mm_stack.chat import MultimodalChat
from mm_stack.config import StackConfig
from mm_stack.search_types import SearchResponse


class ChatAbstainTests(unittest.TestCase):
    def test_low_confidence_abstains_before_vlm_load(self):
        chat_engine = MultimodalChat(StackConfig(text_embed_daemon_autostart=False))
        fake_response = SearchResponse(
            routing_mode="keyword+semantic_fallback",
            routing_reason="test",
            latency_ms=10,
            results=[
                {
                    "image_id": "img-1",
                    "file_path": "/tmp/car.jpg",
                    "caption": "red suv parked on road",
                    "summary": "vehicle scene",
                    "tags": ["car"],
                    "score": 0.44,
                }
            ],
            normalization_explanation="test",
            rerank_todo="",
            query_intent=None,
            policy_applied={"similarity_gate": 0.2},
            confidence_explanation="test",
            verification=None,
            timings={"total_ms": 10},
            retrieval_debug={"confidence_band": "low"},
            confidence_debug={"confidence_band": "low", "confidence_score": 0.30},
            abstain_recommended=True,
            abstain_reason="weak_lexical_support",
        )
        chat_engine.search_engine.search = lambda **kwargs: fake_response  # type: ignore[method-assign]

        events = list(chat_engine.stream_chat("mother", top_k=3))
        self.assertTrue(events)
        final = events[-1]
        self.assertEqual(final["type"], "complete")
        self.assertEqual(final["confidence"], "Low")
        self.assertIn("could not find reliable evidence", final["answer"].lower())
        self.assertEqual(final["grounded_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
