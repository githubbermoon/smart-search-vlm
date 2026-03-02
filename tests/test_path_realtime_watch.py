import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from mm_stack.config import StackConfig
from mm_stack.db import (
    add_watched_folder,
    connect_sqlite,
    ensure_schema,
    upsert_image_metadata,
)
from mm_stack.fs_watch import LivePathWatcher
from mm_stack.preprocess import preprocess_image


def _seed_image(conn, cfg: StackConfig, *, image_id: str, file_path: Path, sha256_hash: str, width: int, height: int) -> None:
    upsert_image_metadata(
        conn,
        {
            "id": image_id,
            "file_path": str(file_path),
            "sha256_hash": sha256_hash,
            "width": width,
            "height": height,
            "caption": "seed",
            "summary": "seed",
            "tags": [],
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


class RealtimePathWatchTests(unittest.TestCase):
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
        self.images_dir = root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_delete_marks_stale_after_grace(self):
        old_path = self.images_dir / "old.png"
        Image.new("RGB", (24, 24), color=(100, 10, 10)).save(old_path)
        prepared = preprocess_image(old_path, self.cfg)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        _seed_image(
            conn,
            self.cfg,
            image_id="img-1",
            file_path=old_path,
            sha256_hash=prepared.sha256_hash,
            width=prepared.width,
            height=prepared.height,
        )
        conn.commit()
        conn.close()

        old_path.unlink()
        watcher = LivePathWatcher(self.cfg, debounce_ms=10, move_grace_sec=0.2)
        watcher.handle_deleted(old_path, now=0.0)

        watcher.process_pending(now=0.1)
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row_pre = conn.execute("SELECT is_stale FROM images WHERE id = ?", ("img-1",)).fetchone()
        conn.close()
        self.assertEqual(int(row_pre["is_stale"]), 0)

        watcher.process_pending(now=0.3)
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row = conn.execute("SELECT is_stale FROM images WHERE id = ?", ("img-1",)).fetchone()
        conn.close()
        self.assertEqual(int(row["is_stale"]), 1)

    def test_create_delete_pair_fallback_relinks(self):
        old_path = self.images_dir / "old.png"
        new_path = self.images_dir / "new.png"
        img = Image.new("RGB", (28, 20), color=(20, 80, 150))
        img.save(old_path)
        img.save(new_path)
        prepared_old = preprocess_image(old_path, self.cfg)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        _seed_image(
            conn,
            self.cfg,
            image_id="img-1",
            file_path=old_path,
            sha256_hash=prepared_old.sha256_hash,
            width=prepared_old.width,
            height=prepared_old.height,
        )
        conn.commit()
        conn.close()

        old_path.unlink()
        watcher = LivePathWatcher(self.cfg, debounce_ms=10, move_grace_sec=0.2)
        watcher.handle_created(new_path, now=0.0)
        watcher.handle_deleted(old_path, now=0.0)

        watcher.process_pending(now=0.05)  # process create fallback first
        watcher.process_pending(now=0.3)   # then process delete grace

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row = conn.execute("SELECT file_path, is_stale FROM images WHERE id = ?", ("img-1",)).fetchone()
        conn.close()

        self.assertEqual(str(row["file_path"]), str(new_path))
        self.assertEqual(int(row["is_stale"]), 0)
        self.assertGreaterEqual(int(watcher.stats["fallback_matched"]), 1)

    def test_unknown_new_file_is_ignored(self):
        unknown_path = self.images_dir / "unknown.png"
        Image.new("RGB", (16, 16), color=(4, 5, 6)).save(unknown_path)

        watcher = LivePathWatcher(self.cfg, debounce_ms=10, move_grace_sec=0.1)
        watcher.handle_created(unknown_path, now=0.0)
        watcher.process_pending(now=0.2)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row = conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()
        conn.close()

        self.assertEqual(int(row["c"]), 0)
        self.assertEqual(int(watcher.stats["ignored_new_files"]), 1)

    def test_hourly_refresh_relinks_missing_without_preprocess_when_custom_hash(self):
        old_path = self.images_dir / "old.png"
        new_path = self.images_dir / "new.png"
        img = Image.new("RGB", (18, 18), color=(11, 111, 211))
        img.save(old_path)
        prepared_old = preprocess_image(old_path, self.cfg)
        old_path.rename(new_path)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        add_watched_folder(conn, str(self.images_dir))
        _seed_image(
            conn,
            self.cfg,
            image_id="img-1",
            file_path=old_path,  # stale path on purpose
            sha256_hash=prepared_old.sha256_hash,
            width=prepared_old.width,
            height=prepared_old.height,
        )
        conn.commit()
        conn.close()

        def fixed_hash(_path: Path) -> str:
            return prepared_old.sha256_hash

        with patch("mm_stack.fs_watch.preprocess_image") as preprocess_mock:
            watcher = LivePathWatcher(self.cfg, hash_resolver=fixed_hash)
            out = watcher.run_hourly_refresh()
            preprocess_mock.assert_not_called()

        self.assertEqual(int(out["resolved"]), 1)
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        row = conn.execute("SELECT file_path, is_stale FROM images WHERE id = ?", ("img-1",)).fetchone()
        conn.close()
        self.assertEqual(str(row["file_path"]), str(new_path))
        self.assertEqual(int(row["is_stale"]), 0)


if __name__ == "__main__":
    unittest.main()
