import tempfile
import unittest
from pathlib import Path

from mm_stack.config import StackConfig
from mm_stack.db import (
    connect_sqlite,
    ensure_schema,
    upsert_cluster,
    upsert_cluster_assignment,
    upsert_image_metadata,
)
from mm_stack.memory import MemoryManager
import mm_stack.memory as memory_module


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


class MemoryClusterTests(unittest.TestCase):
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

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_clusters_and_items(self):
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        _seed_image(conn, self.cfg, "img-1", "/tmp/one.jpg", "One", ["a", "b"])
        _seed_image(conn, self.cfg, "img-2", "/tmp/two.jpg", "Two", ["a"])
        _seed_image(conn, self.cfg, "img-3", "/tmp/three.jpg", "Three", ["c"])

        upsert_cluster(
            conn,
            {
                "id": "cluster-a",
                "name": "Cluster A",
                "topic_label": "People Outdoors",
                "centroid_vector": "[0.1, 0.2]",
                "created_at": "2026-02-20T00:00:00Z",
                "updated_at": "2026-02-20T00:00:00Z",
            },
        )
        upsert_cluster(
            conn,
            {
                "id": "cluster-b",
                "name": "Cluster B",
                "topic_label": "Documents",
                "centroid_vector": "[0.3, 0.4]",
                "created_at": "2026-02-20T00:00:00Z",
                "updated_at": "2026-02-20T00:00:00Z",
            },
        )
        upsert_cluster_assignment(
            conn,
            {
                "item_id": "img-1",
                "cluster_id": "cluster-a",
                "distance": 0.10,
                "assigned_at": "2026-02-20T00:00:00Z",
            },
        )
        upsert_cluster_assignment(
            conn,
            {
                "item_id": "img-2",
                "cluster_id": "cluster-a",
                "distance": 0.20,
                "assigned_at": "2026-02-20T00:00:00Z",
            },
        )
        upsert_cluster_assignment(
            conn,
            {
                "item_id": "img-3",
                "cluster_id": "cluster-b",
                "distance": 0.15,
                "assigned_at": "2026-02-20T00:00:00Z",
            },
        )
        conn.commit()
        conn.close()

        mm = MemoryManager(self.cfg)
        clusters = mm.list_clusters(limit=10, min_items=1)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]["cluster_id"], "cluster-a")
        self.assertEqual(clusters[0]["item_count"], 2)
        self.assertIsNotNone(clusters[0]["sample_item"])
        self.assertEqual(clusters[0]["sample_item"]["image_id"], "img-1")
        self.assertEqual(clusters[0]["sample_item"]["tags"], ["a", "b"])

        items = mm.get_cluster_items("cluster-a", limit=10)
        self.assertEqual([i["image_id"] for i in items], ["img-1", "img-2"])
        self.assertEqual(items[0]["tags"], ["a", "b"])

    def test_update_clusters_replaces_old_rows(self):
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        _seed_image(conn, self.cfg, "img-1", "/tmp/one.jpg", "One", ["a"])
        _seed_image(conn, self.cfg, "img-2", "/tmp/two.jpg", "Two", ["b"])
        upsert_cluster(
            conn,
            {
                "id": "stale-cluster",
                "name": "Stale",
                "topic_label": "Old",
                "centroid_vector": "[0.0, 0.0]",
                "created_at": "2026-02-20T00:00:00Z",
                "updated_at": "2026-02-20T00:00:00Z",
            },
        )
        upsert_cluster_assignment(
            conn,
            {
                "item_id": "img-1",
                "cluster_id": "stale-cluster",
                "distance": 0.50,
                "assigned_at": "2026-02-20T00:00:00Z",
            },
        )
        conn.commit()
        conn.close()

        original_store = memory_module.LanceStore
        original_kmeans = memory_module.MiniBatchKMeans

        class _FakeStore:
            def __init__(self, cfg):
                self.cfg = cfg

            def get_all_clip_vectors(self):
                return ["img-1", "img-2"], [[0.0, 0.0], [1.0, 1.0]]

        class _FakeKMeans:
            def __init__(self, n_clusters, random_state, batch_size, n_init):
                import numpy as np

                self.cluster_centers_ = np.array([[0.5, 0.5]])

            def fit_predict(self, vecs):
                return [0 for _ in vecs]

        memory_module.LanceStore = _FakeStore
        memory_module.MiniBatchKMeans = _FakeKMeans
        try:
            mm = MemoryManager(self.cfg)
            out = mm.update_clusters(n_clusters=1)
        finally:
            memory_module.LanceStore = original_store
            memory_module.MiniBatchKMeans = original_kmeans

        self.assertEqual(out["clusters"], 1)
        self.assertEqual(out["assignments"], 2)

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        stale_exists = conn.execute("SELECT 1 FROM clusters WHERE id='stale-cluster'").fetchone()
        cluster_count = conn.execute("SELECT COUNT(*) AS c FROM clusters").fetchone()["c"]
        assignment_count = conn.execute("SELECT COUNT(*) AS c FROM cluster_assignments").fetchone()["c"]
        conn.close()

        self.assertIsNone(stale_exists)
        self.assertEqual(cluster_count, 1)
        self.assertEqual(assignment_count, 2)


if __name__ == "__main__":
    unittest.main()
