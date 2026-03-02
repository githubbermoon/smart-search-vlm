import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mm_stack.config import StackConfig
from mm_stack.db import connect_sqlite, ensure_schema, upsert_image_metadata
from mm_stack.fs_watch import LivePathWatcher
from mm_stack.ingestion import MultimodalIngestor
from mm_stack.preprocess import preprocess_image


class PathRelinkTests(unittest.TestCase):
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
        (root / "images").mkdir(parents=True, exist_ok=True)
        self.old_path = root / "images/old.png"
        self.new_path = root / "images/new.png"
        img = Image.new("RGB", (32, 24), color=(20, 80, 160))
        img.save(self.old_path)
        img.save(self.new_path)  # same pixels -> same hash

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_hash_relinks_file_path(self):
        prepared_old = preprocess_image(self.old_path, self.cfg)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        upsert_image_metadata(
            conn,
            {
                "id": "img-existing",
                "file_path": str(self.old_path),
                "sha256_hash": prepared_old.sha256_hash,
                "width": prepared_old.width,
                "height": prepared_old.height,
                "caption": "old",
                "summary": "old",
                "tags": [],
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

        ingestor = MultimodalIngestor(self.cfg)
        out = ingestor._ingest_images([self.new_path], safe_reprocess=False)

        self.assertEqual(out["ingested"], 0)
        self.assertEqual(out["skipped_duplicates"], 1)
        self.assertEqual(out.get("relinked_paths", 0), 1)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row = conn.execute("SELECT file_path, is_stale FROM images WHERE id = ?", ("img-existing",)).fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(str(row["file_path"]), str(self.new_path))
        self.assertEqual(int(row["is_stale"]), 0)

    def test_live_move_event_relinks_without_hash_fallback(self):
        prepared_old = preprocess_image(self.old_path, self.cfg)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        upsert_image_metadata(
            conn,
            {
                "id": "img-existing",
                "file_path": str(self.old_path),
                "sha256_hash": prepared_old.sha256_hash,
                "width": prepared_old.width,
                "height": prepared_old.height,
                "caption": "old",
                "summary": "old",
                "tags": [],
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

        hash_calls = {"n": 0}

        def _hash(_path: Path) -> str:
            hash_calls["n"] += 1
            return "unused"

        watcher = LivePathWatcher(self.cfg, hash_resolver=_hash)
        watcher.handle_moved(self.old_path, self.new_path)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row = conn.execute("SELECT file_path, is_stale FROM images WHERE id = ?", ("img-existing",)).fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(str(row["file_path"]), str(self.new_path))
        self.assertEqual(int(row["is_stale"]), 0)
        self.assertEqual(hash_calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
