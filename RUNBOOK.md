# Smart Stack Runbook

Operational guide for `/Users/pranjal/garage/smart_stack`.

## 1. Scope

Use this when:

- Running ingestion manually or via cron
- Investigating failures in `failed/`
- Validating search functionality
- Recovering from dependency/model/storage issues

## 2. Runtime Overview

`ingest.py` flow:

1. Ensure directories and SQLite table/index exist
2. Scan `inbox/` for supported image files
3. Compute SHA256 and skip duplicates
4. OCR with Apple Vision
5. Caption/tag with MLX Qwen3-VL model
6. Embed payload with Sentence Transformers
7. Write metadata to SQLite and vectors to LanceDB
8. Copy media into Obsidian `Media/`
9. Move input file to `processed/`

`search.py` flow:

1. Parse query + optional expansion
2. Embed query
3. Vector search in LanceDB (images + optional notes)
4. Join image metadata from SQLite by `file_hash`
5. Print ranked results; optionally open top hit

`notes_index.py` flow:

1. Scan vault markdown files
2. Chunk note text
3. Embed chunks
4. Upsert note chunks in LanceDB note table
5. Track indexed note hashes in SQLite (`indexed_notes`)

## 3. Standard Commands

Environment:

```bash
cd /Users/pranjal/garage/smart_stack
source .venv/bin/activate
```

Manual ingest:

```bash
python ingest.py
```

Low-RAM guarded ingest:

```bash
./run_guarded_ingest.sh
```

Guarded ingest with explicit knobs:

```bash
python ingest.py \
  --memory-threshold-mb 8704 \
  --memory-gate-mode wait \
  --memory-timeout-sec 180 \
  --memory-poll-sec 5 \
  --memory-relief-cmd "bash /Users/pranjal/clawdGIT/scripts/purge_and_run.sh --threshold-mb 8704 --relief-only"
```

Tune app close list for relief command (optional):

```bash
export PURGE_APPS_CSV="Arc,Google Chrome,Code,Spotify,Slack"
```

Safe reprocess (upsert mode):

```bash
python ingest.py --safe-reprocess
```

Safe reprocess in batches:

```bash
python ingest.py --safe-reprocess --limit 25
```

Manual search:

```bash
python search.py "invoice from january"
python search.py "meeting whiteboard notes" --open --open-app obsidian
python search.py "project roadmap" --no-notes
python search.py "south indian breakfast" --json --no-notes
python search.py "banana" --no-notes --min-score 0.60
./openclaw_imgsearch.py "south indian breakfast" -n 5
./openclaw_imgsearch.py "banana" -n 8 --min-score 0.60
```

Multimodal pipeline (Nomic + CLIP):

```bash
./mm_cli.py ingest-image "/absolute/path/to/image.jpg"
./mm_cli.py ingest-inbox --limit 25
./mm_cli.py search "invoice total amount"
./mm_cli.py search --image-path "/absolute/path/to/query_image.jpg"
./mm_cli.py reembed-all
./mm_cli.py evaluate --init-fixture
./mm_cli.py evaluate
```

SwiftUI wrapper (local compatibility build):

```bash
cd /Users/pranjal/garage/smart_stack/SmartStackUI
./local_run.sh
```

Note: `local_run.sh` only rebuilds when inputs change.

Build only:

```bash
cd /Users/pranjal/garage/smart_stack/SmartStackUI
./local_run.sh --build-only
```

Install clickable app bundle:

```bash
cd /Users/pranjal/garage/smart_stack/SmartStackUI
./install_app.sh
open ~/Applications/SmartStackUI.app
```

Index notes:

```bash
python notes_index.py --embed-model nomic-ai/nomic-embed-text-v1.5
python notes_index.py --limit 50
```

Cron log check:

```bash
tail -n 200 /Users/pranjal/garage/smart_stack/night_shift.log
```

## 4. Health Checks

Directory sanity:

```bash
ls -la /Users/pranjal/garage/smart_stack/inbox
ls -la /Users/pranjal/garage/smart_stack/processed
ls -la /Users/pranjal/garage/smart_stack/failed
```

SQLite file exists:

```bash
ls -la /Users/pranjal/Pranjal-Obs/clawd/smart_stack.db
```

LanceDB exists:

```bash
ls -la /Users/pranjal/Pranjal-Obs/clawd/vectors.lance
```

Basic row count:

```bash
sqlite3 /Users/pranjal/Pranjal-Obs/clawd/smart_stack.db "select count(*) from processed_images;"
```

## 5. Common Failure Modes

### `No images in inbox`

- Cause: expected when queue is empty.
- Action: no-op.

### `ModuleNotFoundError` / import failures

- Cause: missing dependencies in `.venv`.
- Action:

```bash
cd /Users/pranjal/garage/smart_stack
source .venv/bin/activate
uv pip install mlx-vlm lancedb sentence-transformers watchdog python-dotenv sqlite-utils pyobjc-framework-Vision rich
```

### `LanceDB table 'image_embeddings' not found`

- Cause: table is created on first successful ingest.
- Action: ingest at least one image, then rerun search.

### `LanceDB table 'note_embeddings...' not found`

- Cause: note index not built yet for selected embedding model.
- Action: run `python notes_index.py --embed-model <same-model-as-search>`.

### Vision OCR decode/read failures

- Cause: invalid/corrupted image or unsupported encoding edge case.
- Action: file is moved to `failed/`; inspect and retry with a clean image.

### MLX model load/generation failure

- Cause: model download/cache issues, incompatible env, or memory pressure.
- Action: verify internet access for initial model pull, retry with fewer concurrent system workloads, and recreate the venv if persistent.

### Search opens wrong app/path

- Cause: `obsidian_path` missing or stale.
- Action: verify the record in SQLite and use `--open-app finder` as fallback.

## 6. Failed Queue Recovery

Review failures:

```bash
ls -la /Users/pranjal/garage/smart_stack/failed
```

Retry one file:

```bash
mv /Users/pranjal/garage/smart_stack/failed/<file> /Users/pranjal/garage/smart_stack/inbox/
python /Users/pranjal/garage/smart_stack/ingest.py
```

Retry all files:

```bash
mv /Users/pranjal/garage/smart_stack/failed/* /Users/pranjal/garage/smart_stack/inbox/
python /Users/pranjal/garage/smart_stack/ingest.py
```

## 7. Duplicate Handling

- Duplicate check key: SHA256 hash
- Duplicate behavior: input is moved to `processed/` as `duplicate_<filename>` and skipped
- No new DB/vector record is written for duplicates

Safe reprocess behavior (`--safe-reprocess`):

- Source directory switches to `processed/`
- Duplicate skip is disabled (existing rows are intentionally updated)
- SQLite metadata is upserted on `file_hash`
- LanceDB rows are replaced by `file_hash` before insert
- Files are not moved to `failed/` on per-file errors (file remains in place)

## 8. Data Model Reference

SQLite table: `processed_images`

- `id` (PK, autoincrement via sqlite-utils insert behavior)
- `filename`
- `file_hash` (unique indexed)
- `tags` (JSON string)
- `ocr_text`
- `caption`
- `processed_at` (UTC ISO-8601)
- `obsidian_path`

LanceDB table: `image_embeddings`

- `id` / `file_hash` (same hash key)
- `filename`
- `embedding` (vector)
- `text` (embedded payload)
- `processed_at`
- `obsidian_path`

LanceDB note table: `note_embeddings` (or model-suffixed variant)

- `id` (`<file_hash>:<chunk_index>`)
- `file_hash`
- `note_path`
- `note_title`
- `chunk_index`
- `chunk_text`
- `embedding`
- `processed_at`

SQLite note index table: `indexed_notes`

- `note_path` (PK)
- `file_hash`
- `chunk_count`
- `embed_model`
- `indexed_at`

## 9. Maintenance

Recommended:

- Back up `/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db`
- Back up `/Users/pranjal/Pranjal-Obs/clawd/vectors.lance`
- Keep `.venv` dependency install command documented (README)
- Monitor `night_shift.log` for repeated failures

Optional periodic checks:

```bash
sqlite3 /Users/pranjal/Pranjal-Obs/clawd/smart_stack.db "select substr(processed_at,1,10) as day, count(*) from processed_images group by day order by day desc limit 14;"
```

## 10. Change Control Notes

- This runbook reflects current scripts: `/Users/pranjal/garage/smart_stack/ingest.py` and `/Users/pranjal/garage/smart_stack/search.py`.
- If path constants or schema change, update this file and `README.md` in the same commit.
