# Smart Stack Evaluation (Phase 1)

This project now has a seeded benchmark fixture and regression thresholds for retrieval quality.

## Files

- Fixture: `/Users/pranjal/garage/smart_stack/mm_stack/evaluation/benchmark_cases.json`
- Thresholds: `/Users/pranjal/garage/smart_stack/mm_stack/evaluation/thresholds.json`
- Harness: `/Users/pranjal/garage/smart_stack/mm_stack/evaluation.py`

## What is measured

For each mode (`clip`, `text`, `hybrid`) and each case:

- `precision@5`
- `recall@10`
- `avg_similarity`
- `hit@1`, `hit@3`, `hit@5`, `hit@10`
- top-10 retrieved IDs

The harness returns:

- aggregated mode summary
- case hit-rate at 10
- per-case diagnostics
- regression pass/fail with explicit failure reasons

## Run

```bash
cd /Users/pranjal/garage/smart_stack
source .venv/bin/activate
./mm_cli.py evaluate
```

Optional fixture override:

```bash
./mm_cli.py evaluate --fixture /absolute/path/to/benchmark_cases.json
```

## Fixture rules

- Keep at least 20 cases.
- Keep query text stable over time to make regressions comparable.
- If a full reindex changes IDs, only update `relevant_image_ids`.
- Prefer 1-4 relevant images per case to capture ambiguity.

## Threshold tuning

Edit `thresholds.json`:

- `min_summary.*` controls aggregate floors.
- `min_case_hit_rate_at_10.*` controls per-case success rate floors.
- `max_failed_cases_hybrid` controls maximum allowed hybrid misses.

Recommendation: after major model/index upgrades, run evaluation on a stable corpus, then raise thresholds gradually.
