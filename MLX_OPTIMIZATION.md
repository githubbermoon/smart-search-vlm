# MLX Optimization Guide for Smart Stack

## Scope
This guide is specific to `/Users/pranjal/garage/smart_stack` on Apple Silicon (M4, 16 GB RAM), using local-only multimodal search + chat with MLX VLM.

It covers:
1. Common performance issues
2. Workarounds and tuning knobs
3. Tradeoffs (latency, quality, RAM, precision)
4. Operational caveats

---

## Current Latency Anatomy
From recent `mm_cli.py chat ... --json` runs, typical timing buckets are:
1. `retrieval_ms`: usually largest chunk (vector search + rerank + DB metadata join)
2. `vlm_load_ms`: MLX model load cost per process
3. `vlm_generate_ms`: answer generation cost (usually smaller than load)
4. `total_ms`: end-to-end wall clock

Example observed profile:
- `retrieval_ms`: ~13s
- `vlm_load_ms`: ~3s
- `vlm_generate_ms`: ~1.7s
- `total_ms`: ~18s

---

## Primary Issues and Why They Happen

## 1) Slow first token / long wait
- Cause: model load + long prompt prefill.
- In this stack, retrieval context can be large and VLM prefill dominates.

## 2) Duplicate heavy work
- Cause: doing VLM verification during search and then loading VLM again in chat.
- Status: mitigated by disabling search-time verification in chat retrieval path.

## 3) HF "Fetching 14 files" appears each run
- Cause: Hugging Face cache/index check, not always full download.
- Impact: small overhead + noisy logs.

## 4) Retrieval dominates runtime even when answer is short
- Cause: hybrid search + rerank + metadata joins over broad candidate pool.
- Tradeoff: better precision but higher latency.

## 5) High RAM pressure / beachball risk
- Cause: multiple concurrent Python/ML processes or repeated model loads.
- Impact: system instability and process thrash.

---

## Workarounds and Tuning

## A) Keep one heavy stage at a time
1. Do not run CLIP/VLM/text embedding simultaneously.
2. Avoid concurrent ingest + chat + search.
3. Use guarded scripts (`run_guarded_ingest.sh`, kill-switch in `SmartStackUI/local_run.sh`).

## B) Reduce prompt/context payload
1. Lower `top_k` for chat (`-n 2` or `-n 3` unless needed).
2. Keep OCR filtering strict (already implemented).
3. Limit history length in chat context.

## C) Reduce retrieval cost
1. For attribute queries, keep `retrieval_terms` focused (already done via planner).
2. Keep candidate depth moderate (`20` is a good default; avoid unnecessary `40+` except deep analysis mode).
3. Use query intent to avoid broad hybrid fallback when not needed.

## D) Control verification cost
1. Use low-confidence-only verification.
2. Keep verification top-k small (`SMART_STACK_VERIFY_TOP_K=2` or `3`).
3. Disable verification during chat retrieval (already integrated).

## E) Improve cache behavior
1. Set `HF_TOKEN` to reduce hub throttling and metadata overhead.
2. Reuse warm process when possible (UI session persistence helps more than repeated CLI process starts).

---

## Recommended Defaults (M4 / 16 GB)

Set in shell before launch:

```bash
export SMART_STACK_VERIFY_ENABLED=1
export SMART_STACK_VERIFY_LOW_CONF_THRESHOLD=0.72
export SMART_STACK_VERIFY_TOP_K=2
export SMART_STACK_INTENT_WEIGHT_RETRIEVAL=0.60
export SMART_STACK_INTENT_WEIGHT_ATTRIBUTE=0.25
export SMART_STACK_INTENT_WEIGHT_RELATION=0.20
export SMART_STACK_INTENT_REQUIRED_ENTITY_PENALTY=0.35
export SMART_STACK_FUZZY_ALPHA=0.8
export SMART_STACK_FUZZY_BETA=0.2
```

For max responsiveness:
1. Use `chat ... -n 2`
2. Keep one SmartStackUI instance only
3. Avoid running ingest during interactive chat

---

## MLX Model Suggestions

## Faster
- `Qwen3-VL-4B` style MLX quantized models
- Pros: lower load + prefill time
- Cons: lower fine-grained visual reasoning than 8B

## Higher quality
- `Qwen3-VL-8B` style local quantized variants
- Pros: better compositional detail
- Cons: larger load latency and higher RAM pressure

For your constraints, 4B remains the best default for interactive UI, with optional 8B for offline deep analysis.

---

## Tradeoff Matrix

| Option | Latency | RAM | Quality | Precision Risk |
|---|---:|---:|---:|---:|
| Lower `top_k` | Better | Better | Slightly lower recall | Medium |
| Disable verification globally | Better | Better | Lower constraint reliability | High |
| Low-confidence-only verification | Balanced | Balanced | High on hard queries | Low |
| Larger context/history | Worse | Worse | Better grounding sometimes | Medium |
| 8B VLM over 4B | Worse | Worse | Better detail | Low |

---

## Caveats
1. Aggressive penalties can improve precision but may suppress true positives in sparse metadata datasets.
2. Verification reranking can correct top results, but if candidate set lacks the target class, final answers still fail.
3. CLI process-per-query naturally adds startup overhead; app session reuse is faster.
4. HF warning is not usually the core bottleneck, but missing token can still add friction.

---

## Suggested Operational Modes

## Mode 1: Interactive (default)
1. 4B model
2. chat/search with `top_k <= 3`
3. low-confidence verification only
4. no simultaneous ingest

## Mode 2: Deep Inspection
1. higher `top_k`
2. optional 8B model
3. verification enabled
4. run when system is idle

## Mode 3: Bulk Ingestion
1. run guarded ingest script
2. disable chat/UI heavy usage during ingest
3. process in batches

---

## What to Monitor Per Query
From chat JSON output `timings`:
1. If `retrieval_ms` dominates: optimize candidate depth + rerank rules.
2. If `vlm_load_ms` dominates: improve process reuse, avoid duplicate loads.
3. If `vlm_generate_ms` dominates: shorten prompt/context or reduce max tokens.
4. If `total_ms` spikes with high RAM: check concurrent Python processes and kill-switch logs.

---

## Practical Next Steps
1. Keep current architecture (already improved for correctness).
2. Add a "fast chat mode" switch:
- lower top-k
- shorter prompt template
- smaller token budget
3. Add periodic timing aggregation to SQLite (`search_logs` + chat timings) for trend-based tuning.
