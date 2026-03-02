import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mm_stack.config import StackConfig
from mm_stack.db import connect_sqlite, ensure_schema, upsert_image_metadata
from mm_stack.search_engine import MultimodalSearchEngine
import mm_stack.search_engine as search_engine_module


class KeywordSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cfg = StackConfig(
            stack_root=root,
            vault_root=root,
            sqlite_path=root / "smart_stack.db",
            lancedb_path=root / "vectors.lance",
            inbox_dir=root / "inbox",
            processed_dir=root / "processed",
            failed_dir=root / "failed",
            media_dir=root / "media",
            preprocessed_dir=root / ".cache/preprocessed",
            text_embed_daemon_autostart=False,
            search_cross_rerank_enabled=False,
        )
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        upsert_image_metadata(
            conn,
            {
                "id": "img-1",
                "file_path": "/tmp/student_photo.jpg",
                "sha256_hash": "hash-img-1",
                "width": 100,
                "height": 100,
                "caption": "Three men wearing lanyards pose outdoors in front of a tree and vehicle.",
                "summary": "Three men stand together, each wearing a lanyard with an ID badge.",
                "tags": ["men", "lanyard", "id badge"],
                "ocr_structured": [],
                "schema_version": self.cfg.schema_version,
                "embedding_model_clip": self.cfg.clip_model_name,
                "embedding_model_text": self.cfg.text_model_name,
                "embedding_dimension_clip": self.cfg.clip_dimension,
                "embedding_dimension_text": self.cfg.text_dimension,
                "embedding_schema_version_clip": self.cfg.clip_schema_version,
                "embedding_schema_version_text": self.cfg.text_schema_version,
            },
        )
        upsert_image_metadata(
            conn,
            {
                "id": "img-2",
                "file_path": "/tmp/classroom.jpg",
                "sha256_hash": "hash-img-2",
                "width": 100,
                "height": 100,
                "caption": "Two students in a classroom with notebooks.",
                "summary": "Students attending a lecture.",
                "tags": ["students", "classroom"],
                "ocr_structured": [],
                "schema_version": self.cfg.schema_version,
                "embedding_model_clip": self.cfg.clip_model_name,
                "embedding_model_text": self.cfg.text_model_name,
                "embedding_dimension_clip": self.cfg.clip_dimension,
                "embedding_dimension_text": self.cfg.text_dimension,
                "embedding_schema_version_clip": self.cfg.clip_schema_version,
                "embedding_schema_version_text": self.cfg.text_schema_version,
            },
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_keyword_fts_matches_literal_term(self):
        engine = MultimodalSearchEngine(self.cfg)
        rows = engine._search_keyword("lanyard", top_k=5)
        self.assertTrue(any(r["image_id"] == "img-1" for r in rows))

    def test_keyword_fts_matches_stemmed_variant(self):
        engine = MultimodalSearchEngine(self.cfg)
        rows = engine._search_keyword("student", top_k=5)
        self.assertTrue(any(r["image_id"] == "img-2" for r in rows))

    def test_auto_mode_prefers_keyword_hits(self):
        engine = MultimodalSearchEngine(self.cfg)
        engine._search_text = lambda query, store, top_k: []  # type: ignore[method-assign]
        resp = engine.search(
            query="lanyard",
            top_k=3,
            mode="auto",
            enable_verification=False,
        )
        self.assertEqual(resp.routing_mode, "keyword+semantic_hybrid")
        self.assertIsNotNone(resp.retrieval_debug)
        self.assertEqual(resp.retrieval_debug["auto_strategy"], "hybrid")
        self.assertEqual(resp.retrieval_debug["keyword_hits"], 1)
        self.assertEqual(resp.retrieval_debug["semantic_hits"], 0)
        self.assertFalse(resp.retrieval_debug["hard_cutoff_applied"])
        self.assertIsNotNone(resp.confidence_debug)
        self.assertIn("confidence_score", resp.confidence_debug)
        self.assertFalse(resp.abstain_recommended)
        self.assertGreaterEqual(len(resp.results), 1)
        self.assertEqual(resp.results[0]["image_id"], "img-1")

    def test_auto_mode_legacy_keeps_old_fallback_behavior(self):
        engine = MultimodalSearchEngine(self.cfg)
        engine._search_text = lambda query, store, top_k: []  # type: ignore[method-assign]
        resp = engine.search(
            query="lanyard",
            top_k=3,
            mode="auto",
            auto_strategy="legacy",
            semantic_fallback_threshold=0,
            enable_verification=False,
        )
        self.assertEqual(resp.routing_mode, "keyword")
        self.assertIsNotNone(resp.retrieval_debug)
        self.assertEqual(resp.retrieval_debug["auto_strategy"], "legacy")
        self.assertEqual(resp.retrieval_debug["keyword_hits"], 1)
        self.assertEqual(resp.retrieval_debug["semantic_hits"], 0)
        self.assertFalse(resp.retrieval_debug["hard_cutoff_applied"])
        self.assertIsNotNone(resp.confidence_debug)
        self.assertIn("confidence_score", resp.confidence_debug)

    def test_auto_mode_zero_keyword_hits_uses_semantic_fallback(self):
        engine = MultimodalSearchEngine(self.cfg)
        engine._search_text = lambda query, store, top_k: [  # type: ignore[method-assign]
            {"image_id": "img-2", "score": 0.88, "source": "text"}
        ]
        resp = engine.search(
            query="astronaut helmet",
            top_k=3,
            mode="auto",
            enable_verification=False,
        )
        self.assertEqual(resp.routing_mode, "keyword+semantic_fallback")
        self.assertEqual(resp.results[0]["image_id"], "img-2")
        self.assertIsNotNone(resp.retrieval_debug)
        self.assertEqual(resp.retrieval_debug["keyword_hits"], 0)
        self.assertEqual(resp.retrieval_debug["semantic_hits"], 1)

    def test_auto_mode_high_keyword_hits_applies_hard_cutoff(self):
        engine = MultimodalSearchEngine(self.cfg)
        engine._search_keyword = lambda query, top_k: [  # type: ignore[method-assign]
            {"image_id": "img-1", "score": 0.9, "source": "keyword", "bm25_score": 0.1}
            for _ in range(151)
        ]
        engine._search_text = lambda query, store, top_k: [  # type: ignore[method-assign]
            {"image_id": "img-2", "score": 0.95, "source": "text"}
        ]
        resp = engine.search(
            query="lanyard",
            top_k=3,
            mode="auto",
            enable_verification=False,
        )
        self.assertEqual(resp.routing_mode, "keyword")
        self.assertIsNotNone(resp.retrieval_debug)
        self.assertTrue(resp.retrieval_debug["hard_cutoff_applied"])
        self.assertEqual(resp.retrieval_debug["semantic_hits"], 0)

    def test_semantic_mode_path_unchanged(self):
        engine = MultimodalSearchEngine(self.cfg)
        engine._search_text = lambda query, store, top_k: [  # type: ignore[method-assign]
            {"image_id": "img-2", "score": 0.82, "source": "text"}
        ]
        original_route_query = search_engine_module.route_query
        search_engine_module.route_query = lambda query, image_path: SimpleNamespace(  # type: ignore[assignment]
            mode="text",
            reason="forced text route for test",
        )
        try:
            resp = engine.search(
                query="students lecture",
                top_k=3,
                mode="semantic",
                enable_verification=False,
            )
        finally:
            search_engine_module.route_query = original_route_query
        self.assertEqual(resp.routing_mode, "text")
        self.assertGreaterEqual(len(resp.results), 1)
        self.assertEqual(resp.results[0]["image_id"], "img-2")


if __name__ == "__main__":
    unittest.main()
