from __future__ import annotations

from typing import Any

import lancedb

from .config import StackConfig


class LanceStore:
    def __init__(self, cfg: StackConfig):
        self.cfg = cfg
        self.db = lancedb.connect(str(cfg.lancedb_path))

    def _table_names(self) -> set[str]:
        try:
            raw = self.db.list_tables()
            if isinstance(raw, dict):
                tables = raw.get("tables", [])
            elif hasattr(raw, "tables"):
                tables = list(getattr(raw, "tables"))
            elif isinstance(raw, list):
                tables = raw
            else:
                tables = list(raw)
            return {str(x) for x in tables}
        except Exception:
            return {str(x) for x in self.db.table_names()}

    def _ensure_table(self, table_name: str, seed_row: dict[str, Any]) -> Any:
        if table_name in self._table_names():
            return self.db.open_table(table_name)
        return self.db.create_table(table_name, data=[seed_row])

    def upsert_clip_vector(
        self,
        *,
        image_id: str,
        vector: list[float],
        model_name: str,
        schema_version: str,
        created_at: str,
    ) -> None:
        row = {
            "id": f"clip:{image_id}",
            "image_id": image_id,
            "embedding": [float(x) for x in vector],
            "model_name": model_name,
            "schema_version": schema_version,
            "created_at": created_at,
        }
        table = self._ensure_table(self.cfg.clip_index_name, row)
        table.delete(f"image_id = '{image_id}'")
        table.add([row])

    def upsert_text_vector(
        self,
        *,
        image_id: str,
        vector: list[float],
        model_name: str,
        schema_version: str,
        created_at: str,
    ) -> None:
        row = {
            "id": f"text:{image_id}",
            "image_id": image_id,
            "embedding": [float(x) for x in vector],
            "model_name": model_name,
            "schema_version": schema_version,
            "created_at": created_at,
        }
        table = self._ensure_table(self.cfg.text_index_name, row)
        table.delete(f"image_id = '{image_id}'")
        table.add([row])

    def search_clip(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        if self.cfg.clip_index_name not in self._table_names():
            return []
        table = self.db.open_table(self.cfg.clip_index_name)
        search = table.search(vector)
        try:
            search = search.metric("cosine")
        except Exception:
            pass
        rows = search.limit(max(1, int(top_k))).to_list()
        return [dict(r) for r in rows]

    def search_text(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        if self.cfg.text_index_name not in self._table_names():
            return []
        table = self.db.open_table(self.cfg.text_index_name)
        search = table.search(vector)
        try:
            search = search.metric("cosine")
        except Exception:
            pass
        rows = search.limit(max(1, int(top_k))).to_list()
        return [dict(r) for r in rows]

    def get_clip_vector_for_image(self, image_id: str) -> list[float] | None:
        if self.cfg.clip_index_name not in self._table_names():
            return None
        table = self.db.open_table(self.cfg.clip_index_name)
        rows = table.to_pandas()
        subset = rows[rows["image_id"] == image_id]
        if len(subset) == 0:
            return None
        return [float(x) for x in subset.iloc[0]["embedding"]]
