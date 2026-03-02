import unittest

from mm_stack.config import StackConfig
from mm_stack.search_engine import MultimodalSearchEngine
from mm_stack.vlm_analyzer import _parse_json_like


class VLMMetadataRepairTests(unittest.TestCase):
    def test_parse_partial_json_like_payload(self):
        raw = """
{
  "caption": "A vast green valley with grazing sheep, flanked by towering snow-capped mountains under a clear blue sky.",
  "summary": "A scenic alpine valley with sheep in foreground and mountains in background.",
  "tags": ["valley", "mountain", "sheep"],
"""
        parsed = _parse_json_like(raw)
        self.assertIn("vast green valley", parsed["caption"].lower())
        self.assertIn("alpine valley", parsed["summary"].lower())
        self.assertIn("mountain", parsed["tags"])

    def test_search_engine_repairs_legacy_caption_summary(self):
        engine = MultimodalSearchEngine(StackConfig(text_embed_daemon_autostart=False))
        caption, summary = engine._repair_legacy_metadata_text(
            "{",
            "\"caption\": \"A vast green valley with grazing sheep and snow-capped mountains.\",",
        )
        self.assertIn("vast green valley", caption.lower())
        self.assertEqual(summary, caption)


if __name__ == "__main__":
    unittest.main()
