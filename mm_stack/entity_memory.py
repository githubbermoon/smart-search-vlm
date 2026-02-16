from __future__ import annotations

import json
import uuid
from typing import Any

from .utils import json_dumps, utc_now_iso


def _to_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json_dumps(value if value is not None else {})


def replace_image_entity_memory(
    conn,
    *,
    image_id: str,
    entities: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
    mentions: list[dict[str, Any]] | None = None,
    source_model: str = "",
    schema_version: str = "entity-v1",
) -> None:
    """Replace structured memory rows for an image in one transaction scope."""
    now = utc_now_iso()
    entities = entities or []
    relations = relations or []
    mentions = mentions or []

    conn.execute("DELETE FROM entity_attributes WHERE entity_id IN (SELECT id FROM image_entities WHERE image_id = ?)", (image_id,))
    conn.execute("DELETE FROM image_relations WHERE image_id = ?", (image_id,))
    conn.execute("DELETE FROM entity_mentions WHERE image_id = ?", (image_id,))
    conn.execute("DELETE FROM image_entities WHERE image_id = ?", (image_id,))

    label_to_entity_id: dict[str, str] = {}
    for ent in entities:
        entity_id = str(ent.get("id") or uuid.uuid4())
        label = str(ent.get("entity_label", "")).strip().lower()
        entity_type = str(ent.get("entity_type", "unknown")).strip().lower() or "unknown"
        conf = float(ent.get("confidence", 0.0) or 0.0)
        bbox = ent.get("bbox", None)
        conn.execute(
            """
            INSERT INTO image_entities (
                id,image_id,entity_label,entity_type,bbox_json,confidence,source_model,schema_version,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                entity_id,
                image_id,
                label,
                entity_type,
                _to_json(bbox if bbox is not None else []),
                conf,
                source_model,
                schema_version,
                now,
                now,
            ),
        )
        if label and label not in label_to_entity_id:
            label_to_entity_id[label] = entity_id
        attrs = ent.get("attributes", [])
        if isinstance(attrs, dict):
            attrs = [{"attr_key": k, "attr_value": v} for k, v in attrs.items()]
        for attr in attrs if isinstance(attrs, list) else []:
            key = str(attr.get("attr_key", "")).strip().lower()
            value = str(attr.get("attr_value", "")).strip().lower()
            if not key or not value:
                continue
            attr_conf = float(attr.get("confidence", conf) or conf or 0.0)
            conn.execute(
                """
                INSERT INTO entity_attributes (
                    id,entity_id,attr_key,attr_value,confidence,schema_version,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()),
                    entity_id,
                    key,
                    value,
                    attr_conf,
                    schema_version,
                    now,
                    now,
                ),
            )

    for rel in relations:
        subject_label = str(rel.get("subject", "")).strip().lower()
        relation = str(rel.get("relation", "")).strip().lower()
        object_label = str(rel.get("object", "")).strip().lower()
        if not relation:
            continue
        subject_entity_id = label_to_entity_id.get(subject_label)
        object_entity_id = label_to_entity_id.get(object_label)
        conf = float(rel.get("confidence", 0.0) or 0.0)
        evidence = str(rel.get("evidence_text", "")).strip()
        conn.execute(
            """
            INSERT INTO image_relations (
                id,image_id,subject_entity_id,relation,object_entity_id,confidence,evidence_text,schema_version,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                image_id,
                subject_entity_id,
                relation,
                object_entity_id,
                conf,
                evidence,
                schema_version,
                now,
                now,
            ),
        )

    for mention in mentions:
        text = str(mention.get("mention", "")).strip()
        if not text:
            continue
        conn.execute(
            """
            INSERT INTO entity_mentions (
                id,image_id,mention,mention_type,confidence,source_field,created_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                image_id,
                text,
                str(mention.get("mention_type", "name")),
                float(mention.get("confidence", 0.0) or 0.0),
                str(mention.get("source_field", "summary")),
                now,
            ),
        )


def load_entity_memory_for_images(conn, image_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not image_ids:
        return {}
    placeholders = ",".join("?" for _ in image_ids)
    entities_rows = conn.execute(
        f"""
        SELECT id,image_id,entity_label,entity_type,bbox_json,confidence
        FROM image_entities
        WHERE image_id IN ({placeholders})
        ORDER BY confidence DESC
        """,
        image_ids,
    ).fetchall()
    rel_rows = conn.execute(
        f"""
        SELECT image_id,subject_entity_id,relation,object_entity_id,confidence,evidence_text
        FROM image_relations
        WHERE image_id IN ({placeholders})
        ORDER BY confidence DESC
        """,
        image_ids,
    ).fetchall()
    attr_rows = conn.execute(
        f"""
        SELECT ea.entity_id,ea.attr_key,ea.attr_value,ea.confidence,ie.image_id
        FROM entity_attributes ea
        JOIN image_entities ie ON ie.id = ea.entity_id
        WHERE ie.image_id IN ({placeholders})
        ORDER BY ea.confidence DESC
        """,
        image_ids,
    ).fetchall()
    mention_rows = conn.execute(
        f"""
        SELECT image_id,mention,mention_type,confidence
        FROM entity_mentions
        WHERE image_id IN ({placeholders})
        ORDER BY confidence DESC
        """,
        image_ids,
    ).fetchall()

    out: dict[str, dict[str, Any]] = {
        image_id: {"entities": [], "relations": [], "attributes": {}, "mentions": []}
        for image_id in image_ids
    }

    by_entity_id: dict[str, dict[str, Any]] = {}
    for row in entities_rows:
        image_id = str(row["image_id"])
        entity = {
            "id": str(row["id"]),
            "entity_label": str(row["entity_label"]),
            "entity_type": str(row["entity_type"]),
            "bbox": [],
            "confidence": float(row["confidence"] or 0.0),
        }
        try:
            entity["bbox"] = json.loads(str(row["bbox_json"] or "[]"))
        except Exception:
            entity["bbox"] = []
        out[image_id]["entities"].append(entity)
        by_entity_id[str(row["id"])] = entity

    for row in attr_rows:
        image_id = str(row["image_id"])
        key = str(row["attr_key"])
        value = str(row["attr_value"])
        out[image_id]["attributes"][key] = value
        ent = by_entity_id.get(str(row["entity_id"]))
        if ent is not None:
            attrs = ent.setdefault("attributes", [])
            attrs.append(
                {
                    "attr_key": key,
                    "attr_value": value,
                    "confidence": float(row["confidence"] or 0.0),
                }
            )

    for row in rel_rows:
        image_id = str(row["image_id"])
        out[image_id]["relations"].append(
            {
                "relation": str(row["relation"]),
                "entities": [
                    str(row["subject_entity_id"] or ""),
                    str(row["object_entity_id"] or ""),
                ],
                "confidence": float(row["confidence"] or 0.0),
                "evidence": str(row["evidence_text"] or ""),
            }
        )

    for row in mention_rows:
        image_id = str(row["image_id"])
        out[image_id]["mentions"].append(
            {
                "mention": str(row["mention"]),
                "mention_type": str(row["mention_type"]),
                "confidence": float(row["confidence"] or 0.0),
            }
        )

    return out
