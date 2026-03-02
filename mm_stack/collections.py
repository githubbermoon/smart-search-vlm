from __future__ import annotations

import logging
import sqlite3
import uuid
from typing import Any

from .config import StackConfig
from .db import connect_sqlite, upsert_collection
from .utils import utc_now_iso

logger = logging.getLogger(__name__)

class CollectionManager:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()

    def create_collection(self, name: str, query: str, filters: dict[str, Any] | None = None) -> str:
        """
        Creates a new dynamic collection with a specific query.
        """
        c_id = str(uuid.uuid4())
        filters_json = str(filters) if filters else "{}"
        
        conn = connect_sqlite(self.cfg)
        upsert_collection(conn, {
            "id": c_id,
            "name": name,
            "query": query,
            "filters": filters_json,
            "is_dynamic": 1,
            "created_at": utc_now_iso(),
            "last_accessed": utc_now_iso()
        })
        conn.commit()
        conn.close()
        return c_id

    def list_collections(self) -> list[dict[str, Any]]:
        conn = connect_sqlite(self.cfg)
        rows = conn.execute("SELECT * FROM collections ORDER BY name ASC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def evaluate_collection(self, collection_name: str) -> list[dict[str, Any]]:
        """
        Executes the collection's query against the image database.
        Supported simple query syntax (case-insensitive contains):
        - "tag:invoice" -> Checks tags JSON
        - "date:2024" -> Checks created_at
        - "text:cat" -> Checks caption/summary/OCR
        - "cat" -> Defaults to text search
        """
        conn = connect_sqlite(self.cfg)
        row = conn.execute("SELECT * FROM collections WHERE name=?", (collection_name,)).fetchone()
        
        if not row:
            conn.close()
            return []
            
        query = row["query"]
        sql = "SELECT * FROM images WHERE 1=1"
        params = []
        
        # Simple AND-based query parser
        parts = query.split()
        for part in parts:
            if ":" in part:
                key, val = part.split(":", 1)
                key = key.lower()
                if key == "tag":
                    sql += " AND lower(tags) LIKE ?"
                    params.append(f"%{val.lower()}%")
                elif key == "date":
                    sql += " AND created_at LIKE ?"
                    params.append(f"{val}%")
                elif key == "text":
                    sql += " AND (lower(caption) LIKE ? OR lower(summary) LIKE ? OR lower(ocr_structured) LIKE ?)"
                    v = f"%{val.lower()}%"
                    params.extend([v, v, v])
                elif key == "cat" or key == "category":
                    sql += " AND lower(category) LIKE ?"
                    params.append(f"%{val.lower()}%")
            else:
                 # Default text search
                 sql += " AND (lower(caption) LIKE ? OR lower(summary) LIKE ?)"
                 v = f"%{part.lower()}%"
                 params.extend([v, v])
                 
        # Update usage stats
        try:
            conn.execute("UPDATE collections SET last_accessed=? WHERE id=?", (utc_now_iso(), row["id"]))
            conn.commit()
        except Exception:
            pass
        
        results = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in results]

    def delete_collection(self, name: str) -> bool:
        conn = connect_sqlite(self.cfg)
        cursor = conn.execute("DELETE FROM collections WHERE name=?", (name,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
