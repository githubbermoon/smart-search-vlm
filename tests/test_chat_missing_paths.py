import time
import unittest

from mm_stack.chat import MultimodalChat
from mm_stack.config import StackConfig
from mm_stack.search_types import SearchResponse


class ChatMissingPathsTests(unittest.TestCase):
    def test_missing_indexed_files_do_not_crash_chat(self):
        chat_engine = MultimodalChat(StackConfig(text_embed_daemon_autostart=False))
        missing_path = f"/tmp/does_not_exist_{int(time.time() * 1000)}.jpg"
        fake_response = SearchResponse(
            routing_mode="semantic",
            routing_reason="test",
            latency_ms=10,
            results=[
                {
                    "image_id": "img-missing",
                    "file_path": missing_path,
                    "caption": "group of students",
                    "summary": "students in classroom",
                    "tags": ["students"],
                    "score": 0.92,
                    "ocr_structured": "[]",
                }
            ],
            normalization_explanation="test",
            rerank_todo="",
            query_intent=None,
            policy_applied={"similarity_gate": 0.2},
            confidence_explanation="test",
            verification=None,
            timings={"total_ms": 10},
        )
        chat_engine.search_engine.search = lambda **kwargs: fake_response  # type: ignore[method-assign]

        events = list(chat_engine.stream_chat("student", top_k=3))
        self.assertTrue(events)
        final = events[-1]
        self.assertEqual(final["type"], "complete")
        self.assertEqual(final["confidence"], "Low")
        self.assertIn("missing from disk", final["answer"])
        self.assertEqual(final["grounded_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
