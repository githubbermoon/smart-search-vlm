# Smart Stack

Local-first image intelligence pipeline for an Obsidian-based second brain.

`smart_stack` ingests screenshots/receipts/images, extracts OCR text, generates captions and tags, stores metadata + embeddings, and lets you semantically search results from the terminal.

MM-only architecture note:
- `mm_cli.py` + `mm_stack/` is the single active ingestion/search pipeline.
- `ingest.py` and `search.py` are compatibility wrappers that forward to multimodal.

## What It Does

- Watches `inbox/` for images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.heic`, `.heif`, `.bmp`, `.tiff`)
- Runs Apple Vision OCR (native macOS framework)
- Runs Qwen3-VL (MLX) for caption + tags
- Embeds combined text with Nomic text embeddings (`nomic-ai/nomic-embed-text-v1.5` by default)
- Writes metadata to SQLite (default: `~/Library/Application Support/SmartStack/smart_stack.db`)
- Writes vectors to LanceDB (default: `~/Library/Application Support/SmartStack/vectors.lance`)
- Copies media into the configured vault media directory (default: `~/Library/Application Support/SmartStack/Media`)
- Moves processed files to `processed/`, failed files to `failed/`

## Repository Layout

```text
smart_stack/
├── ingest.py         # compatibility wrapper -> mm_cli ingest-inbox
├── notes_index.py    # markdown note indexing CLI
├── search.py         # compatibility wrapper -> mm_cli search
├── mm_cli.py         # multimodal CLI (source of truth)
├── main.py           # placeholder entrypoint
├── RUNBOOK.md        # operational guide
├── inbox/            # drop new images here
├── processed/        # archive after successful ingest
├── failed/           # failed files for retry/debug
├── night_shift.log   # optional cron log output
└── README.md
```

## Requirements

- macOS (Apple Vision + MLX tooling)
- Apple Silicon recommended
- Python 3.14 (repo `.python-version` is `3.14`)
- A local checkout of this repo
- Optional custom data root via `SMART_STACK_VAULT_ROOT=/absolute/path/to/data-root`

## Setup

`pyproject.toml` does not currently pin runtime dependencies, so install them explicitly:

```bash
cd /path/to/smart_stack
uv venv --python 3.14
source .venv/bin/activate
uv pip install mlx-vlm lancedb sentence-transformers watchdog python-dotenv sqlite-utils pyobjc-framework-Vision rich
```

By default Smart Stack uses:

- repo runtime folders: `inbox/`, `processed/`, `failed/`, `.cache/`
- user data root: `~/Library/Application Support/SmartStack`

Backward-compatibility note:

- if `~/Pranjal-Obs/clawd` already exists on a machine, Smart Stack continues using it automatically

If you want a custom data root:

```bash
export SMART_STACK_VAULT_ROOT="$HOME/SmartStackData"
```

## Developer-Installable From Git

This repo is now installable from any clone path on macOS.

```bash
git clone https://github.com/githubbermoon/smart-search-vlm.git
cd smart-search-vlm/smart_stack
uv venv --python 3.14
source .venv/bin/activate
uv pip install mlx-vlm lancedb sentence-transformers watchdog python-dotenv sqlite-utils pyobjc-framework-Vision rich
cd SmartStackUI
./install_app.sh
open ~/Applications/SmartStackUI.app
```

Notes:

- keep the git clone on disk after installing; the app uses that checkout for its Python backend
- if your clone lives somewhere unusual, the launcher passes the correct repo root automatically
- if you want a non-default data directory, set `SMART_STACK_VAULT_ROOT` before launching the app

## Run Ingestion

1. Put images into `./inbox`
2. Run:

```bash
cd /path/to/smart_stack
source .venv/bin/activate
./mm_cli.py ingest-inbox
```

Low-RAM guarded run (recommended on 16GB machines):

```bash
cd /path/to/smart_stack
./run_guarded_ingest.sh
```

This wrapper runs one startup memory gate, then calls `mm_cli.py ingest-inbox`.
Default threshold is `8704MB` (8.5GB, Active+Wired) and is checked once at startup.

Detailed ingest telemetry (stderr JSON lines + optional webhook):

```bash
./mm_cli.py ingest-path "/absolute/path/to/folder" --progress --progress-every 10
./mm_cli.py rescan-all --progress --webhook-url "https://<your-n8n-webhook>"
```

Guarded runner env toggles for telemetry:

- `SMART_STACK_INGEST_PROGRESS=1`
- `SMART_STACK_INGEST_PROGRESS_EVERY=10`
- `SMART_STACK_INGEST_WEBHOOK_URL=https://...`
- `SMART_STACK_INGEST_WEBHOOK_TIMEOUT_SEC=2.0`

Expected output examples:

- `[OK] ...` for success
- `[SKIP] duplicate ...` for hash duplicates
- `[FAIL] ...` and file moved to `failed/` on error

Model override options:

- Set `SMART_STACK_VLM_MODEL=<hf-model-id>` for VLM override
- Set `SMART_STACK_TEXT_MODEL=<hf-model-id>` for text embedding override
- Set `SMART_STACK_MEMORY_THRESHOLD_MB=<int>` for guarded runner threshold
- Set `SMART_STACK_MEMORY_GATE_MODE=wait|skip|fail` for guarded runner behavior

## Multimodal Stack (Nomic + OpenCLIP)

Production-grade multimodal path is available via `mm_stack/` and `mm_cli.py`.

Capabilities:

- structured OCR blocks (`type`, `text`, `bbox`, `confidence`)
- CLIP image/text embeddings (`clip_index`)
- VLM caption+summary+tags
- Nomic text embeddings (`text_index`)
- deterministic query router + hybrid scoring (0.6 clip / 0.4 text)
- embedding versioning + stale re-embedding + evaluation harness

CLI entrypoints:

```bash
cd /path/to/smart_stack
source .venv/bin/activate

# Ingest one image
./mm_cli.py ingest-image "/absolute/path/to/image.jpg"

# Ingest inbox batch
./mm_cli.py ingest-inbox --limit 25

# Deterministic routed search
./mm_cli.py search "receipt total amount"
./mm_cli.py search "poster style like this"
./mm_cli.py search --image-path "/absolute/path/to/query_image.jpg"

# Auto-mode strategy (Hybrid V1 default)
./mm_cli.py search "mountain people lanyard" --mode auto --auto-strategy hybrid --json

# Backward-compatible auto behavior
./mm_cli.py search "mountain people lanyard" --mode auto --auto-strategy legacy --semantic-fallback-threshold 0 --json

# Re-embed stale entries
./mm_cli.py reembed-all

# Initialize + run evaluation harness
./mm_cli.py evaluate --init-fixture
./mm_cli.py evaluate
```

Architecture reference:

- `mm_stack/ARCHITECTURE.md`

## Safe Reprocess (Existing Archive)

Re-run OCR/VLM/embedding for files already in `processed/` and safely update existing records by `file_hash`.

```bash
cd /path/to/smart_stack
source .venv/bin/activate
python ingest.py --safe-reprocess
```

Optional:

- `--limit N` process only first `N` files from `processed/` for controlled batches

### Use Nomic Embeddings

Use this to index by semantic meaning with Nomic:

```bash
cd /path/to/smart_stack
source .venv/bin/activate
python ingest.py --safe-reprocess --embed-model nomic-ai/nomic-embed-text-v1.5
```

Then search with the same embedding model:

```bash
python search.py "south indian breakfast" --embed-model nomic-ai/nomic-embed-text-v1.5 -n 5
```

Notes:

- Vectors are stored in a model-specific LanceDB table automatically.
- Keep ingest/search on the same embedding model for correct results.
- Default BGE vectors are kept intact.

## Run Search

```bash
cd /path/to/smart_stack
source .venv/bin/activate
python search.py "receipts from starbucks last month" -n 5
```

Useful flags:

- `--no-expand` disable query expansion
- `--no-notes` search only image vectors
- `--open` open top result after search
- `--open-app obsidian|finder` choose opener for `--open`
- `--embed-model <hf-model-id>` use a specific embedding model/table
- `--json` print machine-readable JSON payload (integration mode)

## Index Obsidian Notes

Create/update semantic vectors for markdown notes in your vault:

```bash
cd /path/to/smart_stack
source .venv/bin/activate
python notes_index.py --embed-model nomic-ai/nomic-embed-text-v1.5
```

Optional:

- `--limit N` index only first `N` notes
- `--force` re-index unchanged notes
- `--chunk-chars` and `--chunk-overlap` tune chunking

After this, `search.py` returns both images and notes when they share the same embedding model.

## OpenClaw Integration

Use the wrapper script for bot-friendly output:

```bash
cd /path/to/smart_stack
source .venv/bin/activate
./openclaw_imgsearch.py "south indian breakfast" -n 5
./openclaw_imgsearch.py "banana" -n 8 --min-score 0.60
```

Options:

- `--embed-model <hf-model-id>`
- `--min-score <float>` filter weak semantic matches (e.g. `0.60`)
- `--with-notes` include note vectors in results

This script internally calls `search.py --json` and prints compact text suitable for chat channels.

## SwiftUI Wrapper (macOS)

A local UI wrapper has been added at:

- `SmartStackUI/Package.swift`
- `SmartStackUI/Sources/SmartStackUI/main.swift`

It supports:

- semantic and keyword image search
- source/score filters
- one-click `Ingest Inbox`, `Safe Reprocess`, and `Index Notes`
- opening result files directly
- live command logs
- guarded ingest buttons now route through `run_guarded_ingest.sh` (memory gate enabled)

Run via local compatibility wrapper (works around CLT Swift/SDK mismatch):

```bash
cd /path/to/smart_stack/SmartStackUI
./local_run.sh
```

`local_run.sh` now skips rebuild when the binary is up-to-date.

Build only:

```bash
cd /path/to/smart_stack/SmartStackUI
./local_run.sh --build-only
```

If you want to recreate the local SDK cache:

```bash
cd /path/to/smart_stack/SmartStackUI
./local_run.sh --clean-sdk --build-only
```

Install a clickable macOS app bundle:

```bash
cd /path/to/smart_stack/SmartStackUI
./install_app.sh
open ~/Applications/SmartStackUI.app
```

The app also exposes a menu bar dropdown with:

- Open Console
- Ingest Inbox
- Safe Reprocess
- Index Notes
- Quit

## Nightly Automation (Optional)

Crontab example (3:00 AM):

```cron
0 3 * * * cd /path/to/smart_stack && caffeinate -i ./.venv/bin/python ./ingest.py >> ./night_shift.log 2>&1
```

## Important Paths

- Ingest input: `./inbox`
- Ingest success archive: `./processed`
- Ingest failures: `./failed`
- Default SQLite metadata DB: `~/Library/Application Support/SmartStack/smart_stack.db`
- Default LanceDB vectors: `~/Library/Application Support/SmartStack/vectors.lance`
- Default media copies: `~/Library/Application Support/SmartStack/Media`

## Operations

See `RUNBOOK.md` for checks, failure handling, reprocessing flow, and maintenance.
