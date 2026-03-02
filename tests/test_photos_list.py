import tempfile
import unittest
from pathlib import Path

from mm_stack.api import photos_list
from mm_stack.config import StackConfig
from mm_stack.db import connect_sqlite, ensure_schema, upsert_image_metadata


def _seed_image(conn, cfg: StackConfig, image_id: str, file_path: str, caption: str, tags: list[str]) -> None:
    upsert_image_metadata(
        conn,
        {
            "id": image_id,
            "file_path": file_path,
            "sha256_hash": f"hash-{image_id}",
            "width": 100,
            "height": 100,
            "caption": caption,
            "summary": caption,
            "tags": tags,
            "ocr_structured": [],
            "schema_version": cfg.schema_version,
            "embedding_model_clip": cfg.clip_model_name,
            "embedding_model_text": cfg.text_model_name,
            "embedding_dimension_clip": cfg.clip_dimension,
            "embedding_dimension_text": cfg.text_dimension,
            "embedding_schema_version_clip": cfg.clip_schema_version,
            "embedding_schema_version_text": cfg.text_schema_version,
        },
    )


class PhotosListTests(unittest.TestCase):
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
        self.existing = root / "exists.jpg"
        self.existing.write_bytes(b"fake")
        self.missing = root / "missing.jpg"

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        _seed_image(conn, self.cfg, "img-1", str(self.existing), "exists", ["a"])
        _seed_image(conn, self.cfg, "img-2", str(self.missing), "missing", ["b"])
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_photos_list_include_and_exclude_missing(self):
        included = photos_list(limit=10, offset=0, include_missing=True, check_paths=True, cfg=self.cfg)
        self.assertEqual(included["total_indexed"], 2)
        self.assertEqual(included["returned"], 2)
        self.assertEqual(len(included["items"]), 2)
        self.assertTrue(included["path_checks_performed"])
        self.assertTrue(any(item["exists_on_disk"] is False for item in included["items"]))

        excluded = photos_list(limit=10, offset=0, include_missing=False, check_paths=True, cfg=self.cfg)
        self.assertEqual(excluded["total_indexed"], 2)
        self.assertEqual(excluded["returned"], 1)
        self.assertEqual(len(excluded["items"]), 1)
        self.assertTrue(excluded["path_checks_performed"])
        self.assertTrue(all(item["exists_on_disk"] for item in excluded["items"]))


if __name__ == "__main__":
    unittest.main()
