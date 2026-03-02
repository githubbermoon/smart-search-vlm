# Cumulative Context v2 (Smart Stack)

Last updated: 2026-02-20
Workspace: `/Users/pranjal/garage/smart_stack`

## 1) Current Snapshot

- Branch: `master`
- Working tree: intentionally dirty (many ongoing local edits)
- Python runtime actively used: `3.11` (`.venv`)
- Core data paths:
  - SQLite: `/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db`
  - LanceDB: `/Users/pranjal/Pranjal-Obs/clawd/vectors.lance`

## 2) High-Impact Changes Already Implemented

### 2.1 Search Routing and CLI/UI

Implemented Hybrid V1 local routing for `auto` mode:

- `mm_cli.py search` supports:
  - `--mode {auto,keyword,semantic}`
  - `--auto-strategy {legacy,hybrid}` (default `hybrid`)
  - `--verify` (default off)
  - `--semantic-fallback-threshold`
- `mm_stack/api.py` and `mm_stack/search_engine.py` accept/pass these controls.
- `auto` + `hybrid` behavior:
  - run FTS keyword first (`porter unicode61`)
  - if keyword hits `== 0`: semantic text fallback
  - if keyword hits within hard-cutoff: weighted RRF fuse of keyword + semantic
  - if keyword hits above hard-cutoff: keyword-only fast path
- `auto` + `legacy` retains previous threshold-based fallback behavior.

Routing now returns explicit `routing_mode` and `routing_reason`.

### 2.1.1 Hybrid V1 Search Upgrade (What was done + what it aims to achieve)

Implemented:

1. Configurable Hybrid V1 knobs in `/Users/pranjal/garage/smart_stack/mm_stack/config.py`
- `search_auto_strategy_default`
- `search_hybrid_rrf_k`
- `search_hybrid_weight_keyword`
- `search_hybrid_weight_semantic`
- `search_hybrid_candidate_k_min`
- `search_hybrid_candidate_k_max`
- `search_keyword_hard_cutoff`

2. Weighted RRF lexical+semantic fusion in `/Users/pranjal/garage/smart_stack/mm_stack/fusion.py`
- new `weighted_rrf_fuse(...)`
- diagnostics per result: `keyword_rank`, `semantic_rank`, `keyword_score`, `semantic_score`, `fusion_score`

3. Search response telemetry in `/Users/pranjal/garage/smart_stack/mm_stack/search_types.py`
- added optional `retrieval_debug`

4. CLI/API plumbing:
- `/Users/pranjal/garage/smart_stack/mm_cli.py`: `--auto-strategy {legacy,hybrid}`
- `/Users/pranjal/garage/smart_stack/mm_stack/api.py`: `search(..., auto_strategy=...)`

What this hopes to achieve:

- Keep local-first search fast (no query-time VLM load).
- Improve recall on conceptual queries while preserving lexical precision.
- Reduce failure mode where text-only fallback over-ranks weak semantic mentions.
- Make routing and ranking decisions inspectable in JSON via `retrieval_debug`.

### 2.1.2 Phase-1B: Cross-Rerank + Composite Confidence + Chat Abstain

Implemented:

1. Bounded cross-encoder rerank (complements existing embeddings)  
Files:
- `/Users/pranjal/garage/smart_stack/mm_stack/cross_rerank.py`
- `/Users/pranjal/garage/smart_stack/mm_stack/search_engine.py`

Behavior:
- reranks only top shortlist (`rerank_k` budgeted by `top_k`, query length, and keyword-hit density)
- fuses base score + cross score: `final = (1-w)*base + w*cross`
- keeps per-row diagnostics:
  - `base_score_pre_cross`
  - `cross_score`
  - `score_post_cross`

2. Composite retrieval confidence estimator  
Files:
- `/Users/pranjal/garage/smart_stack/mm_stack/retrieval_confidence.py`
- `/Users/pranjal/garage/smart_stack/mm_stack/search_engine.py`

Computed fields:
- `confidence_score`
- `confidence_band` (`high|medium|low`)
- `lexical_support_top3`
- `top1_score`
- `top2_margin`
- `rerank_signal`
- `abstain_recommended`
- `abstain_reason`

3. Verification gate tightened to medium-confidence only  
File:
- `/Users/pranjal/garage/smart_stack/mm_stack/verification.py`

Behavior:
- verify only when:
  - verification enabled,
  - query has constraints,
  - confidence is medium (`abstain_threshold <= score < verify_threshold`)
- skip verify for high-confidence and low-confidence-abstain
- emit verification decision reason in debug payload

4. Chat abstention before VLM load (low-confidence only)  
File:
- `/Users/pranjal/garage/smart_stack/mm_stack/chat.py`

Behavior:
- for normal chat mode (not focus-mode, not compare-mode), if `search_resp.abstain_recommended`:
  - return deterministic abstain answer immediately
  - skip VLM load/generation entirely
  - keep response grounded and avoid nearest-neighbor hallucination

5. Search response/debug shape upgrades  
File:
- `/Users/pranjal/garage/smart_stack/mm_stack/search_types.py`

Added optional response fields:
- `rerank_debug`
- `confidence_debug`
- `abstain_recommended`
- `abstain_reason`

Why this was added:
- replace brittle patch-style semantic handling with learned reranking
- reduce semantic drift on weak/generic queries (e.g., role/kinship terms)
- preserve local-first latency by keeping VLM out of search and gating expensive verification

Rollback knobs (env):
- `SMART_STACK_SEARCH_CROSS_RERANK_ENABLED=false`
- `SMART_STACK_SEARCH_CROSS_RERANK_MODEL=...`
- `SMART_STACK_SEARCH_CROSS_RERANK_WEIGHT=...`
- `SMART_STACK_SEARCH_CONFIDENCE_ABSTAIN_THRESHOLD=...`
- `SMART_STACK_SEARCH_CONFIDENCE_VERIFY_THRESHOLD=...`

### 2.2 SQLite Keyword Engine (FTS5)

`mm_stack/db.py` includes:

- `images_fts` virtual table over: `file_path, caption, summary, tags, ocr_structured`
- tokenizer: `porter unicode61`
- insert/update/delete triggers
- migration/backfill logic for existing DBs

### 2.3 Query-Time Embedding Daemon (No Query-Time VLM)

Added persistent local text embedding daemon:

- `/Users/pranjal/garage/smart_stack/mm_stack/text_embed_daemon.py`
- `/Users/pranjal/garage/smart_stack/mm_stack/text_embed_client.py`

Behavior:

- search tries daemon first
- falls back to one-shot embedder if daemon unavailable
- avoids repeated model cold-load for normal semantic search

### 2.4 SmartStackUI Integration + Kill Switch

`SmartStackUI` updated:

- keyword search path now uses CLI auto mode (not raw sqlite LIKE)
- semantic search path uses CLI semantic mode
- emergency red kill switch added and hardened to kill:
  - stack worker python processes
  - embed daemon and children
  - stale socket/pid artifacts

Documentation exists at:
- `/Users/pranjal/garage/smart_stack/doucment.md`

### 2.5 Ranking / Matching Fixes

- Short-token FTS wildcard issue fixed:
  - `car` no longer expands to `car*` (prevents false match like `carrier`)
- Metadata quality penalty added to demote malformed caption/summary rows.

## 3) Critical 2026-02-17 Fix: Broken VLM Metadata (`caption: "{"`)

### Problem
Some ingested rows had malformed metadata patterns like:

- `caption = "{"`
- `summary = '"caption": "...",'`
- `tags = ["image"]`

This corrupted keyword and semantic behavior.

### Fixes applied

1. **Ingestion-time parse hardening** in `/Users/pranjal/garage/smart_stack/mm_stack/vlm_analyzer.py`
- `_parse_json_like(...)` now recovers partial JSON payloads
- extracts `caption`, `summary`, `tags` from truncated JSON-like output
- avoids persisting brace-only fallback metadata

2. **Read-time legacy repair** in `/Users/pranjal/garage/smart_stack/mm_stack/search_engine.py`
- `_repair_legacy_metadata_text(...)` repairs old malformed DB rows at search output time
- prevents user-facing broken caption/summary even before full reprocess

3. **Regression tests** added:
- `/Users/pranjal/garage/smart_stack/tests/test_vlm_metadata_repair.py`

### Verified result
Repro query now returns clean mountain metadata:

```bash
cd /Users/pranjal/garage/smart_stack
./.venv/bin/python mm_cli.py search mountain -n 8 --mode auto --semantic-fallback-threshold 0 --json
```

No `caption: "{"` returned for the tested row.

## 4) Validation Status (Latest)

Executed and passing:

```bash
./.venv/bin/python -m unittest \
  tests.test_fusion_rrf \
  tests.test_keyword_search \
  tests.test_query_fuzzy_metrics \
  tests.test_query_normalization \
  tests.test_query_planner \
  tests.test_query_policy \
  tests.test_search_rerank \
  tests.test_intent_ranking \
  tests.test_vlm_metadata_repair \
  tests.test_cross_rerank \
  tests.test_retrieval_confidence \
  tests.test_chat_abstain \
  tests.test_verification_gate
```

Result: `Ran 47 tests ... OK`

Also validated:

- `search mountain` keyword path returns repaired metadata
- safe reprocess completed on affected mountain image (`ingested: 1, failed: []`)

## 5) Known Open Gaps (Still Important)

1. `ingestion.py` persistence failure accounting is still weak in `_process_candidates`.
   - exceptions are printed; returned `failed` list can be incomplete
2. `ingest_batch()` modality result merging via `dict.update()` can overwrite keys.
3. Video/audio dedupe and content hashing are still weaker than image dedupe.
4. Hybrid weighting and cutoff are still static constants (no dataset-scaled auto-tuning yet).
5. Log-scaled adaptive weighting is deferred (planned for V1.1; not in Hybrid V1).
6. Mass-scale ingestion hardening (chunking/checkpointing/commit batching) is not fully done yet.
7. Cross-rerank model cold start can still add startup latency on first use; warm caching mitigates it.

## 6) Suggested New-Thread Start Prompt

```md
Task: [one exact objective]

Workspace: /Users/pranjal/garage/smart_stack
Read first: /Users/pranjal/garage/smart_stack/CUMULATIVE_CONTEXT_v2.md

Scope (edit only):
- [absolute path 1]
- [absolute path 2]

Repro:
1. [exact command]
2. [exact command]

Observed:
- [paste exact JSON snippets]

Acceptance criteria:
1. [expected behavior]
2. [non-regression]
3. [tests/commands that must pass]
```

## 7) Quick Commands for Next Thread

```bash
cd /Users/pranjal/garage/smart_stack

git branch --show-current
git status --short

./.venv/bin/python mm_cli.py search "mountain" -n 8 --mode auto --semantic-fallback-threshold 0 --json
./.venv/bin/python mm_cli.py search "mountain" -n 8 --mode auto --auto-strategy hybrid --json
./.venv/bin/python -m unittest tests.test_fusion_rrf tests.test_keyword_search tests.test_cross_rerank tests.test_retrieval_confidence tests.test_chat_abstain tests.test_verification_gate

# Force-disable cross-rerank (rollback switch)
SMART_STACK_SEARCH_CROSS_RERANK_ENABLED=false ./.venv/bin/python mm_cli.py search "mother" -n 8 --mode auto --json
```
