from __future__ import annotations

import re
from typing import Any

from .intent_types import QueryIntent
from .query_normalization import fuzzy_match_score


PRESENCE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "car": ("suv", "vehicle", "automobile", "sedan", "hatchback"),
    "bike": ("bicycle", "motorcycle", "cycle"),
}


def _tok(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


def _candidate_text(row: dict[str, Any]) -> str:
    tags = row.get("tags", [])
    tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags or "")
    attrs = row.get("attributes", {})
    attr_text = " ".join(f"{k} {v}" for k, v in attrs.items()) if isinstance(attrs, dict) else str(attrs)
    rel = row.get("relation_evidence", [])
    rel_text = " ".join(str(x.get("relation", "")) for x in rel) if isinstance(rel, list) else str(rel)
    return (
        f"{row.get('caption', '')} "
        f"{row.get('summary', '')} "
        f"{row.get('ocr_structured', '')} "
        f"{tags_text} "
        f"{attr_text} "
        f"{rel_text}"
    )


def _extract_candidate_entities(row: dict[str, Any]) -> set[str]:
    out = _tok(_candidate_text(row))
    entities = row.get("entities", [])
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict):
                out.update(_tok(str(entity.get("entity_label", ""))))
                out.update(_tok(str(entity.get("entity_type", ""))))
    mentions = row.get("mentions", [])
    if isinstance(mentions, list):
        for mention in mentions:
            if isinstance(mention, dict):
                out.update(_tok(str(mention.get("mention", ""))))
    return out


def rerank_with_intent(
    rows: list[dict[str, Any]],
    intent: QueryIntent,
    *,
    retrieval_weight: float,
    attribute_weight: float,
    relation_weight: float,
    required_entity_penalty: float,
    activity_boost: float,
    color_boost: float,
    pattern_boost: float,
    presence_required: bool,
) -> list[dict[str, Any]]:
    if not rows:
        return rows

    max_vector = max(float(r.get("score", 0.0) or 0.0) for r in rows) or 1.0
    relation_terms = [t for t in intent.relation_terms if t]
    attribute_terms = [t for t in intent.attribute_terms if t]
    presence_terms = [t for t in intent.presence_terms if t]
    retrieval_terms = [t for t in intent.retrieval_terms if t]

    out: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        text = _candidate_text(row_copy)
        text_tokens = _tok(text)
        entity_tokens = _extract_candidate_entities(row_copy)

        vector_similarity = max(0.0, min(1.0, float(row_copy.get("score", 0.0) / max_vector)))

        query_for_overlap = retrieval_terms + attribute_terms + relation_terms
        semantic_overlap = fuzzy_match_score(query_for_overlap, text, fuzzy_threshold=0.84)

        attribute_score = 0.0
        if attribute_terms:
            attribute_score = fuzzy_match_score(attribute_terms, text, fuzzy_threshold=0.84)
        if intent.appearance.get("colors"):
            color_term_score = fuzzy_match_score(intent.appearance["colors"], text, fuzzy_threshold=0.88)
            attribute_score += color_boost * color_term_score
        if intent.appearance.get("patterns"):
            pattern_term_score = fuzzy_match_score(intent.appearance["patterns"], text, fuzzy_threshold=0.88)
            attribute_score += pattern_boost * pattern_term_score
        if intent.activity_terms:
            activity_term_score = fuzzy_match_score(intent.activity_terms, text, fuzzy_threshold=0.84)
            attribute_score += activity_boost * activity_term_score
        attribute_score = max(0.0, min(1.0, attribute_score))

        relation_score = 0.0
        if relation_terms:
            relation_score = fuzzy_match_score(relation_terms, text, fuzzy_threshold=0.90)
            rel_evidence = row_copy.get("relation_evidence", [])
            if isinstance(rel_evidence, list) and rel_evidence:
                relation_score = max(relation_score, max(float(x.get("confidence", 0.0) or 0.0) for x in rel_evidence))

        presence_hits = 0
        for term in presence_terms:
            synonyms = PRESENCE_SYNONYMS.get(term, ())
            synonym_hit = any(s in entity_tokens or s in text_tokens for s in synonyms)
            if term in entity_tokens or term in text_tokens or synonym_hit:
                presence_hits += 1
        presence_score = (
            (presence_hits / max(1, len(presence_terms)))
            if presence_terms
            else (1.0 if not presence_required else 0.0)
        )

        if intent.require_person:
            person_terms = {"person", "people", "man", "men", "woman", "women", "couple", "boy", "girl", "male", "female"}
            if entity_tokens & person_terms:
                presence_score = max(presence_score, 1.0)
            else:
                presence_score = min(presence_score, 0.25)

        # Keep weights normalized and conservative.
        final_score = (
            retrieval_weight * vector_similarity
            + (1.0 - retrieval_weight) * semantic_overlap
            + attribute_weight * attribute_score
            + relation_weight * relation_score
            + (attribute_weight * 0.5) * presence_score
        )

        if presence_required and presence_terms and presence_score < 0.50:
            penalty = required_entity_penalty
            if intent.attribute_terms and intent.retrieval_terms:
                # Queries like "color car" should not rank items without car presence.
                penalty = max(penalty, 0.60)
            final_score *= max(0.0, 1.0 - penalty)

        row_copy["component_scores"] = {
            "vector_similarity": round(vector_similarity, 6),
            "semantic_overlap": round(semantic_overlap, 6),
            "attribute_score": round(attribute_score, 6),
            "relation_score": round(relation_score, 6),
            "presence_score": round(presence_score, 6),
        }
        row_copy["final_score"] = round(final_score, 6)
        row_copy["score"] = round(final_score, 6)
        out.append(row_copy)

    out.sort(key=lambda r: float(r.get("final_score", r.get("score", 0.0))), reverse=True)
    return out
