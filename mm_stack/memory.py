from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .config import StackConfig
from .db import connect_sqlite, ensure_schema, upsert_cluster, upsert_cluster_assignment
from .lancedb_store import LanceStore
from .utils import utc_now_iso

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        raw = (raw or "").strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            import json

            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [x.strip() for x in raw.split(",") if x.strip()]

    def update_clusters(self, n_clusters: int = 20) -> dict[str, Any]:
        """
        Re-clusters all CLIP vectors using MiniBatchKMeans.
        Updates 'clusters' and 'cluster_assignments' tables.
        """
        store = LanceStore(self.cfg)
        ids, vecs = store.get_all_clip_vectors()

        if not ids:
            return {"clusters": 0, "assignments": 0, "msg": "No vectors found"}

        vecs_np = np.array(vecs)
        n_samples = len(vecs_np)
        
        # Adjust clusters if not enough samples
        if n_samples < n_clusters:
            logger.info("Fewer samples (%d) than requested clusters (%d). Reducing n_clusters.", n_samples, n_clusters)
            n_clusters = max(1, n_samples)

        logger.info("Clustering %d vectors into %d clusters...", n_samples, n_clusters)
        
        # Use MiniBatchKMeans for performance
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=256,
            n_init="auto"
        )
        labels = kmeans.fit_predict(vecs_np)
        centroids = kmeans.cluster_centers_

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        # Replace previous clustering so repeated recalc does not accumulate stale rows.
        conn.execute("DELETE FROM cluster_assignments")
        conn.execute("DELETE FROM clusters")
        cluster_map = {}  # label_idx -> cluster_uuid

        # 1. Update Clusters
        for i, centroid in enumerate(centroids):
            # Create a new Cluster ID
            c_id = str(uuid.uuid4())
            cluster_map[i] = c_id

            upsert_cluster(conn, {
                "id": c_id,
                "name": f"Cluster {i+1}",
                "topic_label": "Analysis Pending", # Placeholder for Auto-Labeling
                "centroid_vector": str(centroid.tolist()),
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso()
            })

        # 2. Update Assignments
        assignments_count = 0
        for idx, label in enumerate(labels):
            img_id = ids[idx]
            c_id = cluster_map[label]
            
            # Distance: L2 norm logic for approximate distance
            dist = float(np.linalg.norm(vecs_np[idx] - centroids[label]))

            upsert_cluster_assignment(conn, {
                "item_id": img_id,
                "cluster_id": c_id,
                "distance": dist,
                "assigned_at": utc_now_iso()
            })
            assignments_count += 1

        # TODO: Prune old clusters? For now we just add new ones.
        # Ideally we should clear the 'clusters' table before inserting new ones 
        # or mark old ones as inactive. But lets keep it simple for MVP.

        conn.commit()
        conn.close()

        return {
            "clusters": len(centroids),
            "assignments": assignments_count
        }

    def list_clusters(self, *, limit: int = 64, min_items: int = 1) -> list[dict[str, Any]]:
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.name,
                c.topic_label,
                c.created_at,
                c.updated_at,
                COUNT(a.item_id) AS item_count
            FROM clusters c
            LEFT JOIN cluster_assignments a ON a.cluster_id = c.id
            GROUP BY c.id
            HAVING COUNT(a.item_id) >= ?
            ORDER BY item_count DESC, COALESCE(c.updated_at, c.created_at) DESC
            LIMIT ?
            """,
            (max(0, int(min_items)), max(1, int(limit))),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            cid = str(row["id"])
            sample = conn.execute(
                """
                SELECT i.id, i.file_path, i.caption, i.tags, a.distance
                FROM cluster_assignments a
                JOIN images i ON i.id = a.item_id
                WHERE a.cluster_id = ?
                ORDER BY a.distance ASC
                LIMIT 1
                """,
                (cid,),
            ).fetchone()
            sample_item = None
            if sample is not None:
                sample_item = {
                    "image_id": str(sample["id"]),
                    "file_path": str(sample["file_path"]),
                    "caption": str(sample["caption"] or ""),
                    "tags": self._parse_tags(str(sample["tags"] or "")),
                    "distance": float(sample["distance"] or 0.0),
                }

            out.append(
                {
                    "cluster_id": cid,
                    "name": str(row["name"]),
                    "topic_label": str(row["topic_label"] or ""),
                    "item_count": int(row["item_count"] or 0),
                    "created_at": str(row["created_at"] or ""),
                    "updated_at": str(row["updated_at"] or ""),
                    "sample_item": sample_item,
                }
            )
        conn.close()
        return out

    def get_cluster_items(self, cluster_id: str, *, limit: int = 120) -> list[dict[str, Any]]:
        cid = (cluster_id or "").strip()
        if not cid:
            return []
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT i.id, i.file_path, i.caption, i.tags, i.created_at, a.distance
            FROM cluster_assignments a
            JOIN images i ON i.id = a.item_id
            WHERE a.cluster_id = ?
            ORDER BY a.distance ASC, i.created_at DESC
            LIMIT ?
            """,
            (cid, max(1, int(limit))),
        ).fetchall()
        conn.close()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "image_id": str(row["id"]),
                    "file_path": str(row["file_path"]),
                    "caption": str(row["caption"] or ""),
                    "tags": self._parse_tags(str(row["tags"] or "")),
                    "created_at": str(row["created_at"] or ""),
                    "distance": float(row["distance"] or 0.0),
                }
            )
        return out

    def auto_label_clusters(self) -> int:
        """
        Generates topic labels for clusters that are pending analysis.
        Uses the VLM on the centroid image of each cluster.
        """
        conn = connect_sqlite(self.cfg)
        # Check for clusters needing labels
        cursor = conn.execute(
            "SELECT id, name FROM clusters WHERE topic_label IS NULL OR topic_label = 'Pending' OR topic_label = 'Analysis Pending'"
        )
        pending_clusters = cursor.fetchall()
        
        if not pending_clusters:
            conn.close()
            return 0
        
        from .vlm_analyzer import VLMAnalyzer
        logger.info("Auto-labeling %d clusters...", len(pending_clusters))
        
        count = 0
        # Initialize VLM
        with VLMAnalyzer(self.cfg.vlm_model_name) as vlm:
            for cluster in pending_clusters:
                c_id = cluster["id"]
                # Find closest image (center of cluster)
                row = conn.execute(
                    "SELECT item_id FROM cluster_assignments WHERE cluster_id=? ORDER BY distance ASC LIMIT 1", 
                    (c_id,)
                ).fetchone()
                
                if not row:
                    logger.info("Cluster %s has no items, skipping label", cluster["name"])
                    continue
                
                img_id = row["item_id"]
                img_row = conn.execute("SELECT file_path FROM images WHERE id=?", (img_id,)).fetchone()
                if not img_row:
                    continue
                
                path = Path(img_row["file_path"])
                if not path.exists():
                    continue

                try:
                    short_prompt = (
                        "Analyze this image. It is the representative center of a cluster of similar images.\n"
                        "Provide a very short, specific 3-5 word topic label for this visual group (e.g., 'Sunset Landscapes', 'Technical Diagrams', 'Receipts').\n"
                        "Return ONLY the label text."
                    )
                    label = vlm.generate_text(path, short_prompt).strip()
                    # Cleanup
                    label = label.replace('"', '').replace('\n', ' ').strip()
                    if len(label) > 50:
                        label = label[:50]
                    
                    conn.execute("UPDATE clusters SET topic_label=?, updated_at=? WHERE id=?", (label, utc_now_iso(), c_id))
                    conn.commit()
                    count += 1
                    logger.info("Labeled %s as '%s'", cluster["name"], label)
                except Exception as e:
                    logger.error("Failed to label cluster %s: %s", c_id, e)

        conn.close()
        return count
