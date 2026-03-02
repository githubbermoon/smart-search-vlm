import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mm_stack.config import StackConfig
from mm_stack.ingestion import MultimodalIngestor
from mm_stack.models import PreparedImage


class IngestBatchingTests(unittest.TestCase):
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
            ingest_image_batch_size=2,
        )
        self.files: list[Path] = []
        for i in range(5):
            p = root / f"img_{i}.png"
            p.write_bytes(b"fake")
            self.files.append(p)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ingest_images_processes_in_batches(self):
        batch_sizes: list[int] = []

        def fake_preprocess(path: Path, _cfg: StackConfig) -> PreparedImage:
            return PreparedImage(
                source_path=path,
                normalized_path=path,
                sha256_hash=f"sha-{path.name}",
                width=10,
                height=10,
            )

        def fake_process(self, candidates, safe_reprocess):  # type: ignore[no-untyped-def]
            batch_sizes.append(len(candidates))
            return {
                "ingested": len(candidates),
                "skipped_duplicates": 0,
                "relinked_paths": 0,
                "failed": [],
            }

        ingestor = MultimodalIngestor(self.cfg, image_batch_size=2)
        with patch("mm_stack.ingestion.preprocess_image", side_effect=fake_preprocess), patch(
            "mm_stack.ingestion.extract_ocr_structured", return_value=([], 0.0)
        ), patch.object(MultimodalIngestor, "_process_candidates", new=fake_process):
            out = ingestor._ingest_images(self.files, safe_reprocess=False)

        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertEqual(out["ingested"], 5)
        self.assertEqual(out["skipped_duplicates"], 0)
        self.assertEqual(out["relinked_paths"], 0)
        self.assertEqual(out["failed"], [])


if __name__ == "__main__":
    unittest.main()
