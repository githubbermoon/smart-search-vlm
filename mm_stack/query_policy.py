from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import StackConfig
from .intent_types import QueryIntent


@dataclass(frozen=True)
class AdaptiveQueryPolicy:
    query_type: str
    policy_confidence_score: float
    fallback_to_generic: bool
    similarity_gate: float
    top_k_multiplier: float
    lexical_mode: str  # boost | enforce
    presence_required: bool
    required_entity_penalty: float
    clip_weight: float
    text_weight: float
    explanation: str

    def candidate_top_k(self, requested_top_k: int, *, minimum: int = 20) -> int:
        scaled = int(math.ceil(max(1, requested_top_k) * self.top_k_multiplier))
        return max(minimum, scaled)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_query_policy(
    intent: QueryIntent | None,
    cfg: StackConfig,
    *,
    requested_top_k: int,
) -> AdaptiveQueryPolicy:
    base_gate = _clamp(cfg.policy_base_similarity_gate, 0.0, 1.0)
    max_adj = abs(cfg.policy_gate_adjustment_max)

    query_type = "generic"
    confidence = 1.0
    if intent is not None:
        query_type = (intent.query_type or "generic").strip().lower() or "generic"
        confidence = _clamp(float(intent.policy_confidence_score), 0.0, 1.0)

    fallback_to_generic = confidence < cfg.policy_confidence_fallback_threshold
    effective_type = "generic" if fallback_to_generic else query_type

    if not cfg.adaptive_policy_enabled:
        return AdaptiveQueryPolicy(
            query_type="generic",
            policy_confidence_score=confidence,
            fallback_to_generic=fallback_to_generic,
            similarity_gate=base_gate,
            top_k_multiplier=1.0,
            lexical_mode="boost",
            presence_required=bool(intent.require_presence) if intent is not None else False,
            required_entity_penalty=cfg.intent_required_entity_penalty,
            clip_weight=0.60,
            text_weight=0.40,
            explanation=(
                f"type=generic confidence={confidence:.2f} fallback={fallback_to_generic} "
                f"gate={base_gate:.2f} topk_multiplier=1.00 lexical=boost adaptive=off"
            ),
        )

    similarity_adj = 0.0
    top_k_multiplier = 1.0
    lexical_mode = "boost"
    presence_required = bool(intent.require_presence) if intent is not None else False
    required_penalty = cfg.intent_required_entity_penalty
    clip_weight, text_weight = 0.60, 0.40

    if effective_type == "generic":
        similarity_adj = -max_adj
        top_k_multiplier = min(2.0, cfg.policy_max_top_k_multiplier)
        lexical_mode = "boost"
        presence_required = False
        required_penalty = min(required_penalty, 0.20)
        clip_weight, text_weight = 0.60, 0.40
    elif effective_type == "constrained":
        similarity_adj = max_adj * 0.70
        top_k_multiplier = min(1.25, cfg.policy_max_top_k_multiplier)
        lexical_mode = "enforce"
        presence_required = True
        required_penalty = max(required_penalty, 0.45)
        clip_weight, text_weight = 0.62, 0.38
    elif effective_type == "attribute":
        similarity_adj = 0.0
        top_k_multiplier = min(1.5, cfg.policy_max_top_k_multiplier)
        lexical_mode = "boost"
        presence_required = bool(intent.require_person or intent.require_presence) if intent is not None else False
        required_penalty = max(required_penalty, 0.35)
        clip_weight, text_weight = 0.45, 0.55
    elif effective_type == "identity":
        similarity_adj = max_adj * 0.35
        top_k_multiplier = min(1.5, cfg.policy_max_top_k_multiplier)
        lexical_mode = "enforce"
        presence_required = False
        required_penalty = max(required_penalty, 0.25)
        clip_weight, text_weight = 0.35, 0.65

    similarity_gate = _clamp(base_gate + similarity_adj, 0.0, 1.0)
    explanation = (
        f"type={effective_type} confidence={confidence:.2f} fallback={fallback_to_generic} "
        f"gate={similarity_gate:.2f} topk_multiplier={top_k_multiplier:.2f} lexical={lexical_mode}"
    )
    return AdaptiveQueryPolicy(
        query_type=effective_type,
        policy_confidence_score=confidence,
        fallback_to_generic=fallback_to_generic,
        similarity_gate=similarity_gate,
        top_k_multiplier=top_k_multiplier,
        lexical_mode=lexical_mode,
        presence_required=presence_required,
        required_entity_penalty=required_penalty,
        clip_weight=clip_weight,
        text_weight=text_weight,
        explanation=explanation,
    )
