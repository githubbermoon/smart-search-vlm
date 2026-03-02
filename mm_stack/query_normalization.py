from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover - fallback path when rapidfuzz missing
    fuzz = None
    process = None


@dataclass
class NormalizedQuery:
    raw_query: str
    normalized_query: str
    tokens_raw: list[str]
    tokens_normalized: list[str]
    fuzzy_tokens: list[str] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip().lower()


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in re.findall(r"[\w]+", _normalize_text(text), flags=re.UNICODE):
        if tok and tok not in out:
            out.append(tok)
    return out


def _soundex(token: str) -> str:
    if not token:
        return ""
    token = token.upper()
    if not re.fullmatch(r"[A-Z]+", token):
        return ""
    mapping = {
        **{c: "1" for c in "BFPV"},
        **{c: "2" for c in "CGJKQSXZ"},
        **{c: "3" for c in "DT"},
        "L": "4",
        **{c: "5" for c in "MN"},
        "R": "6",
    }
    first = token[0]
    digits: list[str] = []
    prev = mapping.get(first, "")
    for ch in token[1:]:
        code = mapping.get(ch, "")
        if code != prev:
            if code:
                digits.append(code)
            prev = code
    return (first + "".join(digits) + "000")[:4]


def normalize_query(raw: str) -> NormalizedQuery:
    """Normalize query text without changing user intent.

    Why we keep `raw_query` intact:
    - Vector retrieval should preserve user intent exactly.
    - Fuzzy matching supplements ranking and should not replace embedding semantics.
    - Heavy LLM/grammar rewriting before retrieval can drift intent and hurt precision.
    """
    raw_query = raw or ""
    normalized = _normalize_text(raw_query)
    return NormalizedQuery(
        raw_query=raw_query,
        normalized_query=normalized,
        tokens_raw=_tokenize(raw_query),
        tokens_normalized=_tokenize(normalized),
    )


def _pair_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    best = 0.0
    if process is not None and fuzz is not None:
        hit = process.extractOne(a, [b], scorer=fuzz.ratio)
        if hit:
            best = max(best, float(hit[1]) / 100.0)
    else:
        best = max(best, SequenceMatcher(None, a, b).ratio())

    # Optional phonetic assist for latin-script tokens.
    # Guard with prefix consistency to avoid false positives like "check"~"couch".
    sx_a = _soundex(a)
    sx_b = _soundex(b)
    has_same_start = len(a) >= 1 and len(b) >= 1 and a[0] == b[0]
    has_prefix_or_suffix_anchor = (
        (len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2])
        or (len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:])
    )
    if sx_a and sx_b and sx_a == sx_b and has_same_start and has_prefix_or_suffix_anchor:
        best = max(best, 0.86)

    return best


def _best_vocab_match(token: str, vocab: list[str]) -> tuple[str, float]:
    if not token or not vocab:
        return "", 0.0

    if process is not None and fuzz is not None:
        hit = process.extractOne(token, vocab, scorer=fuzz.ratio)
        if not hit:
            return "", 0.0
        return str(hit[0]), float(hit[1]) / 100.0

    best_token = ""
    best_score = 0.0
    for cand in vocab:
        s = SequenceMatcher(None, token, cand).ratio()
        if s > best_score:
            best_score = s
            best_token = cand
    return best_token, best_score


def fuzzy_match_score(query_tokens: list[str], text: str, *, fuzzy_threshold: float = 0.84) -> float:
    """Compute fuzzy overlap score [0,1] between query tokens and a candidate text."""
    tokens_q = [t for t in (_tokenize(" ".join(query_tokens)) if query_tokens else []) if len(t) >= 2]
    tokens_t = [t for t in _tokenize(text) if len(t) >= 2]
    if not tokens_q or not tokens_t:
        return 0.0

    set_t = set(tokens_t)
    score_sum = 0.0
    for qt in tokens_q:
        if qt in set_t:
            score_sum += 1.0
            continue
        best = 0.0
        for tt in tokens_t:
            sim = _pair_similarity(qt, tt)
            if sim > best:
                best = sim
        if best >= fuzzy_threshold:
            score_sum += best

    return max(0.0, min(1.0, score_sum / max(1, len(tokens_q))))


def _candidate_text(row: dict[str, Any]) -> str:
    tags = row.get("tags", [])
    tags_text = " ".join(str(x) for x in tags) if isinstance(tags, list) else str(tags)
    return (
        f"{row.get('caption', '')} "
        f"{row.get('summary', '')} "
        f"{row.get('ocr_structured', '')} "
        f"{tags_text}"
    )


def combined_rank(
    candidates: list[dict[str, Any]],
    query: str | NormalizedQuery,
    *,
    alpha: float = 0.8,
    beta: float = 0.2,
    fuzzy_threshold: float = 0.84,
    min_combined_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Rerank candidates with vector + fuzzy score fusion.

    Vector similarity remains primary (alpha), fuzzy typo-aware overlap is secondary (beta).
    This preserves embedding precision while improving typo robustness.
    """
    if not candidates:
        return []

    nq = query if isinstance(query, NormalizedQuery) else normalize_query(str(query))

    vocab_counter: dict[str, int] = {}
    for row in candidates:
        for tok in _tokenize(_candidate_text(row)):
            if len(tok) < 3:
                continue
            vocab_counter[tok] = vocab_counter.get(tok, 0) + 1

    vocab = [k for k, _ in sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)[:4000]]

    fuzzy_tokens: list[str] = []
    correction_threshold = min(0.80, fuzzy_threshold)
    for tok in nq.tokens_normalized:
        if len(tok) < 5:
            continue
        best_tok, best_score = _best_vocab_match(tok, vocab)
        if best_tok and best_score >= correction_threshold and best_tok != tok and best_tok not in fuzzy_tokens:
            fuzzy_tokens.append(best_tok)
    nq.fuzzy_tokens = fuzzy_tokens

    effective_tokens = []
    for tok in [*nq.tokens_normalized, *nq.fuzzy_tokens]:
        if tok not in effective_tokens:
            effective_tokens.append(tok)

    max_vector = max(float(r.get("score", 0.0) or 0.0) for r in candidates)
    if max_vector <= 0.0:
        max_vector = 1.0

    prelim: list[tuple[dict[str, Any], float]] = []
    max_fuzzy = 0.0
    for row in candidates:
        fs = fuzzy_match_score(effective_tokens, _candidate_text(row), fuzzy_threshold=fuzzy_threshold)
        max_fuzzy = max(max_fuzzy, fs)
        prelim.append((row, fs))
    has_fuzzy_signal = max_fuzzy >= 0.30
    typo_recovery_mode = bool(nq.fuzzy_tokens)
    alpha_eff = alpha
    beta_eff = beta
    if not typo_recovery_mode:
        # For non-typo queries, keep semantic/vector ranking primary and reduce
        # lexical-fuzzy influence to avoid over-promoting exact-token partial hits.
        alpha_eff = max(alpha, 0.9)
        beta_eff = min(beta, max(0.0, 1.0 - alpha_eff))

    ranked: list[dict[str, Any]] = []
    for row, fuzzy_score in prelim:
        row_copy = dict(row)
        vector_score = float(row_copy.get("score", 0.0) or 0.0)
        vector_norm = max(0.0, min(1.0, vector_score / max_vector))
        text_support = float(row_copy.get("text_score", vector_norm) or vector_norm)
        text_support = max(0.0, min(1.0, text_support))
        source_mode = str(row_copy.get("source", "")).lower()
        if source_mode == "hybrid":
            # Query is text (this rerank runs only for text queries), so in hybrid
            # mode prefer text-index semantic strength over fused vector score.
            semantic_score = (0.30 * vector_norm) + (0.70 * text_support)
        else:
            semantic_score = (0.85 * vector_norm) + (0.15 * text_support)

        final_score = (alpha_eff * semantic_score) + (beta_eff * fuzzy_score)
        if final_score < min_combined_score:
            final_score = final_score * 0.95
        # If some lexical/fuzzy signal exists in this candidate set, down-rank
        # clip-leaning rows that have zero fuzzy support for the query terms.
        if typo_recovery_mode and has_fuzzy_signal and fuzzy_score <= 0.0:
            final_score *= (0.85 + (0.15 * text_support))

        row_copy["vector_score"] = round(vector_score, 6)
        row_copy["vector_score_norm"] = round(vector_norm, 6)
        row_copy["semantic_score"] = round(semantic_score, 6)
        row_copy["fuzzy_score"] = round(fuzzy_score, 6)
        row_copy["score"] = round(final_score, 6)
        ranked.append(row_copy)

    ranked.sort(key=lambda r: (float(r.get("score", 0.0)), float(r.get("fuzzy_score", 0.0))), reverse=True)
    return ranked
