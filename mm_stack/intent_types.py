from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AttributeIntent:
    colors: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    clothing_terms: list[str] = field(default_factory=list)
    activity_terms: list[str] = field(default_factory=list)
    age_terms: list[str] = field(default_factory=list)

    def terms(self) -> list[str]:
        out: list[str] = []
        for value in (
            self.colors
            + self.patterns
            + self.clothing_terms
            + self.activity_terms
            + self.age_terms
        ):
            if value not in out:
                out.append(value)
        return out


@dataclass(frozen=True)
class RelationIntent:
    relation_terms: list[str] = field(default_factory=list)
    relation_pairs: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class PresenceIntent:
    required_entities: list[str] = field(default_factory=list)
    require_person: bool = False


@dataclass(frozen=True)
class QueryTypeFlags:
    compositional: bool = False
    attribute_heavy: bool = False
    relation_heavy: bool = False


@dataclass(frozen=True)
class QueryIntent:
    raw_query: str
    normalized_query: str
    tokens_raw: list[str]
    tokens_normalized: list[str]

    retrieval_terms: list[str] = field(default_factory=list)
    relation_terms: list[str] = field(default_factory=list)
    attribute_terms: list[str] = field(default_factory=list)
    presence_terms: list[str] = field(default_factory=list)

    appearance: dict[str, list[str]] = field(
        default_factory=lambda: {"colors": [], "patterns": []}
    )
    clothing_terms: list[str] = field(default_factory=list)
    activity_terms: list[str] = field(default_factory=list)
    name_terms: list[str] = field(default_factory=list)

    relation_pairs: list[tuple[str, str, str]] = field(default_factory=list)
    query_type_flags: QueryTypeFlags = field(default_factory=QueryTypeFlags)
    query_type: str = "generic"
    policy_confidence_score: float = 1.0

    require_person: bool = False
    require_presence: bool = False

    def has_constraints(self) -> bool:
        return bool(
            self.attribute_terms
            or self.relation_terms
            or self.require_person
            or self.require_presence
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
