from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .intent_reranker import rerank_with_intent
from .intent_types import QueryIntent, QueryTypeFlags
from .query_normalization import normalize_query
from .relation_rules import find_relation_phrases, relation_token_set


STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "of",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "show",
    "find",
    "any",
    "image",
    "images",
    "photo",
    "photos",
    "there",
    "where",
    "does",
    "do",
    "did",
    "having",
}

PERSON_TERMS: set[str] = {
    "person",
    "people",
    "man",
    "men",
    "woman",
    "women",
    "boy",
    "girl",
    "male",
    "female",
    "couple",
    "family",
    "elderly",
    "senior",
    "old",
}
CLOTHING_TERMS: set[str] = {
    "shirt",
    "tshirt",
    "tee",
    "kurta",
    "sari",
    "saree",
    "dress",
    "jacket",
    "coat",
    "pant",
    "pants",
    "trouser",
    "trousers",
    "uniform",
    "robe",
}
COLOR_TERMS: set[str] = {
    "white",
    "black",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "gray",
    "grey",
    "gold",
    "silver",
    "beige",
}
PATTERN_TERMS: set[str] = {"check", "checked", "checkered", "plaid", "stripe", "striped"}
ACTIVITY_TERMS: set[str] = {
    "meal",
    "eating",
    "eat",
    "dining",
    "dinner",
    "lunch",
    "breakfast",
    "walking",
    "holding",
    "sitting",
    "standing",
}
AGE_TERMS: set[str] = {"old", "elderly", "senior", "aged"}
ATTRIBUTE_CUE_TERMS: set[str] = {"color", "colour", "pattern", "age", "activity", "style"}

SYNONYM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "old": ("elderly", "senior", "aged"),
    "elderly": ("old", "senior", "aged"),
    "check": ("checkered", "checked", "plaid"),
    "checkered": ("check", "checked", "plaid"),
}


def _uniq(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _extract_name_terms(raw_query: str) -> list[str]:
    names = re.findall(r"\b[A-Z][a-z]{2,}\b", raw_query or "")
    return _uniq([name.lower() for name in names if name.lower() not in STOPWORDS])


def _expand_attribute_terms(values: list[str]) -> list[str]:
    out = list(values)
    for value in list(values):
        out.extend(list(SYNONYM_EXPANSIONS.get(value, ())))
    return _uniq(out)


def parse_query(raw_query: str) -> QueryIntent:
    normalized = normalize_query(raw_query)
    tokens = list(normalized.tokens_normalized)
    token_set = set(tokens)
    relation_phrases = find_relation_phrases(normalized.normalized_query)
    relation_terms = _uniq([item[0] for item in relation_phrases])
    relation_words = relation_token_set()

    colors = [tok for tok in tokens if tok in COLOR_TERMS]
    patterns = [tok for tok in tokens if tok in PATTERN_TERMS]
    clothing_terms = [tok for tok in tokens if tok in CLOTHING_TERMS]
    activity_terms = [tok for tok in tokens if tok in ACTIVITY_TERMS]
    age_terms = [tok for tok in tokens if tok in AGE_TERMS]
    name_terms = _extract_name_terms(raw_query)

    attribute_terms = _expand_attribute_terms(
        _uniq(colors + patterns + clothing_terms + activity_terms + age_terms)
    )
    # Attribute-intent cues should not dominate retrieval (for example "color car").
    for cue in tokens:
        if cue in ATTRIBUTE_CUE_TERMS and cue not in attribute_terms:
            attribute_terms.append(cue)
    attribute_terms = _uniq(attribute_terms)
    retrieval_terms: list[str] = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if tok in relation_words:
            continue
        if tok in attribute_terms:
            continue
        retrieval_terms.append(tok)
    retrieval_terms = _uniq(retrieval_terms)

    multi_object = " and " in f" {normalized.normalized_query} " or "," in normalized.normalized_query
    if relation_terms:
        multi_object = True
    presence_terms = _uniq(retrieval_terms[:2] if multi_object else retrieval_terms[:1])

    require_person = bool(
        token_set & PERSON_TERMS
        or clothing_terms
        or any(tok in {"man", "men", "woman", "women", "couple"} for tok in tokens)
    )
    if any(term in {"shirt", "dress", "jacket", "kurta", "sari"} for term in clothing_terms):
        require_person = True

    compositional = bool(len(retrieval_terms) >= 2 or relation_terms)
    attribute_heavy = bool(len(attribute_terms) >= 2)
    relation_heavy = bool(relation_terms)
    flags = QueryTypeFlags(
        compositional=compositional,
        attribute_heavy=attribute_heavy,
        relation_heavy=relation_heavy,
    )

    require_presence = bool(multi_object or (attribute_terms and retrieval_terms))

    return QueryIntent(
        raw_query=raw_query or "",
        normalized_query=normalized.normalized_query,
        tokens_raw=normalized.tokens_raw,
        tokens_normalized=tokens,
        retrieval_terms=retrieval_terms,
        relation_terms=relation_terms,
        attribute_terms=attribute_terms,
        presence_terms=presence_terms,
        appearance={"colors": colors, "patterns": patterns},
        clothing_terms=clothing_terms,
        activity_terms=activity_terms,
        name_terms=name_terms,
        relation_pairs=[],
        query_type_flags=flags,
        require_person=require_person,
        require_presence=require_presence,
    )


@dataclass(frozen=True)
class LegacyQueryIntent:
    raw_query: str
    normalized_query: str
    tokens: list[str]
    person_terms: set[str] = field(default_factory=set)
    clothing_terms: set[str] = field(default_factory=set)
    appearance_terms: set[str] = field(default_factory=set)
    activity_terms: set[str] = field(default_factory=set)
    requires_person: bool = False
    requires_clothing: bool = False
    requires_activity: bool = False
    requires_male: bool = False
    requires_old_age: bool = False

    def has_attribute_intent(self) -> bool:
        return bool(
            self.person_terms
            or self.clothing_terms
            or self.appearance_terms
            or self.activity_terms
            or self.requires_person
            or self.requires_clothing
            or self.requires_activity
            or self.requires_male
            or self.requires_old_age
        )


def parse_query_intent(query: str) -> LegacyQueryIntent:
    intent = parse_query(query)
    token_set = set(intent.tokens_normalized)
    appearance = set(intent.appearance.get("colors", []) + intent.appearance.get("patterns", []))
    return LegacyQueryIntent(
        raw_query=intent.raw_query,
        normalized_query=intent.normalized_query,
        tokens=intent.tokens_normalized,
        person_terms={tok for tok in token_set if tok in PERSON_TERMS},
        clothing_terms=set(intent.clothing_terms),
        appearance_terms=appearance,
        activity_terms=set(intent.activity_terms),
        requires_person=intent.require_person,
        requires_clothing=bool(intent.clothing_terms or intent.appearance.get("patterns")),
        requires_activity=bool(intent.activity_terms),
        requires_male=bool(token_set & {"man", "men", "male", "gentleman"}),
        requires_old_age=bool(token_set & AGE_TERMS),
    )


def rerank_with_query_intent(
    rows: list[dict[str, Any]],
    query: str,
    *,
    appearance_weight: float = 0.14,
    activity_weight: float = 0.12,
    presence_weight: float = 0.18,
    missing_person_penalty: float = 0.40,
    missing_clothing_penalty: float = 0.35,
    semi_hard_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around the new intent reranker."""
    intent = parse_query(query)
    if not intent.has_constraints():
        return rows
    penalty = max(missing_person_penalty, missing_clothing_penalty) if semi_hard_enabled else 0.0
    return rerank_with_intent(
        rows,
        intent,
        retrieval_weight=max(0.45, 1.0 - (appearance_weight + activity_weight)),
        attribute_weight=appearance_weight + activity_weight,
        relation_weight=0.20,
        required_entity_penalty=penalty,
        activity_boost=activity_weight,
        color_boost=appearance_weight,
        pattern_boost=appearance_weight,
        presence_required=bool(semi_hard_enabled and (presence_weight > 0.0)),
    )
