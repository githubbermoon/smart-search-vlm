import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mm_stack.config import StackConfig
from mm_stack.ingest_telemetry import IngestTelemetry, TelemetryOptions
from mm_stack.ingestion import MultimodalIngestor
from mm_stack.models import PreparedImage


class _Collector:
    def __init__(self):
        self.items: list[tuple[str, dict]] = []

    def emit(self, event: str, **payload):
        self.items.append((event, payload))


class _DummyResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class IngestTelemetryTests(unittest.TestCase):
    def test_webhook_events_are_sent(self):
        calls: list[tuple[str, bytes | None, float | None]] = []

        def _fake_urlopen(req, timeout=None):
            calls.append((req.full_url, req.data, timeout))
            return _DummyResponse()

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            telemetry = IngestTelemetry(
                run_id="run-1",
                options=TelemetryOptions(
                    command="ingest-path",
                    emit_to_stderr=False,
                    webhook_url="https://example.test/ingest",
                    webhook_timeout_sec=1.25,
                ),
            )
            telemetry.emit("stage_started", stage="preprocess", total=10)
            telemetry.emit("stage_completed", stage="preprocess", total=10)
            telemetry.close()

        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(url == "https://example.test/ingest" for (url, _, _) in calls))

    def test_ingest_images_emits_progress_events(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg = StackConfig(
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

        files = [root / f"img_{i}.png" for i in range(5)]
        for file_path in files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"x")

        collector = _Collector()
        ingestor = MultimodalIngestor(
            cfg,
            image_batch_size=2,
            telemetry=collector,  # duck-typed telemetry sink
            progress_every=2,
        )

        def _fake_preprocess(path: Path, _cfg: StackConfig) -> PreparedImage:
            idx = int(path.stem.split("_")[-1])
            return PreparedImage(
                source_path=path,
                normalized_path=path,
                sha256_hash=f"hash-{idx}",
                width=8,
                height=8,
            )

        def _fake_process(_self, candidates, _safe_reprocess):
            return {
                "ingested": len(candidates),
                "skipped_duplicates": 0,
                "relinked_paths": 0,
                "failed": [],
            }

        with patch("mm_stack.ingestion.preprocess_image", side_effect=_fake_preprocess), patch(
            "mm_stack.ingestion.extract_ocr_structured", return_value=([], 0.0)
        ), patch.object(MultimodalIngestor, "_process_candidates", new=_fake_process):
            out = ingestor._ingest_images(files, safe_reprocess=False)

        self.assertEqual(out["ingested"], 5)
        names = [name for (name, _payload) in collector.items]
        self.assertIn("ingest_images_started", names)
        self.assertIn("ingest_images_completed", names)
        progress_events = [p for (name, p) in collector.items if name == "ingest_images_progress"]
        self.assertGreaterEqual(len(progress_events), 2)


if __name__ == "__main__":
    unittest.main()
