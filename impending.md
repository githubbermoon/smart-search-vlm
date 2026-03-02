# Impending Plan (Adaptive Retrieval Policy)

## Objective
Stop per-word patching and move to a deterministic, query-type-aware retrieval policy that improves generic query recall without harming constrained-query precision.

## Root Cause
Current failures are primarily from fixed gating and inconsistent lexical controls, not missing vocabulary.
A single fixed threshold across all query shapes causes under-retrieval for generic queries and instability for compositional queries.

## Phase 1: Policy Foundation (Now)
1. Add deterministic query-type classifier (minimal v1 set only):
- `generic`
- `constrained`
- `attribute`
- `identity`
2. Build adaptive retrieval policy per query type:
- dynamic similarity gate (bounded)
- lexical influence mode (boost vs enforce)
- entity/presence gating strength
- top-k expansion bounds
3. Keep Search + Chat on the same policy path to avoid drift.
4. Add `policy_confidence_score`; if low confidence, fallback to `generic`.

Guardrails:
- gate adjustment max `±0.07`
- top-k expansion max `2x` (example: `8 -> 16`)
- lexical signals are boost-only unless constraints are explicit

## Phase 2: Safe Rollout + Validation
1. Roll out behind feature flag (`SMART_STACK_ADAPTIVE_POLICY_ENABLED`).
2. Keep legacy word patches behind separate flag (`SMART_STACK_LEGACY_PATCHES_ENABLED`).
3. Run side-by-side evaluation on ~150 real queries.
4. Compare:
- precision@k
- recall@k
- false-positive rate
- “not found” rate for generic single-word queries

Acceptance checks:
- `code` surfaces XML/code-like images reliably
- constrained queries remain precise (`car next to bike`, `white check shirt having meal`)
- no major regression vs legacy path

## Phase 3: Consolidation
1. Tune policy bounds/weights using eval logs.
2. Deprecate and remove legacy per-word patches only after regression checks pass.
3. Keep policy explainability in outputs:
- query type
- gate used
- lexical mode
- presence mode
- confidence explanation

## Phase 4: Optional Advanced Layer
1. Add richer intent decomposition (relation-heavy, QA-style, vibe) only if needed.
2. Add structured attribute/relation reasoning enhancements incrementally.
3. Keep deterministic-first and local-only constraints.

## Public API / Interface Impact
No breaking API changes in this documentation phase.
For later implementation, additive response fields are expected:
- `query_intent`
- `policy_confidence_score`
- `policy_applied`
- component score breakdown

## Test Scenarios (for implementation phase)
1. Generic: `code`, `dog`, `sunset`, `design`
2. Constrained: `car next to bike`
3. Attribute: `white check shirt having meal`
4. Identity: `David in photos`
5. Typo robustness: `jwellery`, `reciept`, `enviroment`
6. Chat/Search parity: same query should not diverge due to different gating paths

## Assumptions and Defaults
1. Local-only inference, no cloud routing.
2. Deterministic policy logic, no LLM-based query rewriting.
3. Bounded adaptivity to prevent instability.
4. Legacy patches retained temporarily for safe rollback.
