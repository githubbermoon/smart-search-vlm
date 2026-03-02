from __future__ import annotations


RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "next_to": ("next to", "beside", "near", "alongside"),
    "behind": ("behind", "at the back of"),
    "in_front_of": ("in front of", "ahead of"),
    "holding": ("holding", "holds", "carrying"),
    "wearing": ("wearing", "wears", "dressed in"),
    "eating": ("eating", "having meal", "dining", "having lunch", "having dinner"),
    "with": ("with",),
}


def find_relation_phrases(text: str) -> list[tuple[str, str]]:
    lowered = (text or "").lower()
    out: list[tuple[str, str]] = []
    for relation, aliases in RELATION_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                out.append((relation, alias))
                break
    return out


def relation_token_set() -> set[str]:
    toks: set[str] = set()
    for aliases in RELATION_ALIASES.values():
        for phrase in aliases:
            toks.update(part for part in phrase.split() if part)
    return toks
