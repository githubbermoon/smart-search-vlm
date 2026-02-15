# Smart Stack

Local-first image intelligence pipeline for an Obsidian-based second brain.

`smart_stack` ingests screenshots/receipts/images, extracts OCR text, generates captions and tags, stores metadata + embeddings, and lets you semantically search results from the terminal.

## What It Does

- Watches `inbox/` for images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.heic`, `.heif`, `.bmp`, `.tiff`)
- Runs Apple Vision OCR (native macOS framework)
- Runs Qwen3-VL (MLX) for caption + tags
- Embeds combined text with `BAAI/bge-small-en-v1.5`
- Indexes markdown notes into note vectors (optional)
- Writes metadata to SQLite (`~/Pranjal-Obs/clawd/smart_stack.db`)
- Writes vectors to LanceDB (`~/Pranjal-Obs/clawd/vectors.lance`)
- Copies media into Obsidian vault (`~/Pranjal-Obs/clawd/Media`)
- Moves processed files to `processed/`, failed files to `failed/`

## Repository Layout

```text
smart_stack/
├── ingest.py         # ingestion pipeline
├── notes_index.py    # markdown note indexing CLI
├── search.py         # semantic search CLI
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
- Obsidian vault at `~/Pranjal-Obs/clawd` (or update paths in code)

## Setup

`pyproject.toml` does not currently pin runtime dependencies, so install them explicitly:

```bash
cd /Users/pranjal/garage/smart_stack
uv venv --python 3.14
source .venv/bin/activate
uv pip install mlx-vlm lancedb sentence-transformers watchdog python-dotenv sqlite-utils pyobjc-framework-Vision rich
```

## Run Ingestion

1. Put images into `/Users/pranjal/garage/smart_stack/inbox`
2. Run:

```bash
cd /Users/pranjal/garage/smart_stack
source .venv/bin/activate
python ingest.py
```

Low-RAM guarded run (recommended on 16GB machines):

```bash
cd /Users/pranjal/garage/smart_stack
./run_guarded_ingest.sh
```

This wrapper calls `ingest.py` with a memory gate and optional relief command hook.
Default threshold is `8704MB` (8.5GB, Active+Wired) and is checked once at startup.

Expected output examples:

- `[OK] ...` for success
- `[SKIP] duplicate ...` for hash duplicates
- `[FAIL] ...` and file moved to `failed/` on error

Model override options:

- `--vlm-model <hf-model-id>` choose VLM at runtime
- `--embed-model <hf-model-id>` choose embedding model at runtime
- `--memory-threshold-mb <int>` gate when Active+Wired memory is high
- `--memory-gate-mode wait|skip|fail` behavior under memory pressure
- `--memory-relief-cmd "<cmd>"` optional one-shot relief command while gated

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
cd /Users/pranjal/garage/smart_stack
source .venv/bin/activate

# Ingest one image
./mm_cli.py ingest-image "/absolute/path/to/image.jpg"

# Ingest inbox batch
./mm_cli.py ingest-inbox --limit 25

# Deterministic routed search
./mm_cli.py search "receipt total amount"
./mm_cli.py search "poster style like this"
./mm_cli.py search --image-path "/absolute/path/to/query_image.jpg"

# Re-embed stale entries
./mm_cli.py reembed-all

# Initialize + run evaluation harness
./mm_cli.py evaluate --init-fixture
./mm_cli.py evaluate
```

Architecture reference:

- `/Users/pranjal/garage/smart_stack/mm_stack/ARCHITECTURE.md`

## Safe Reprocess (Existing Archive)

Re-run OCR/VLM/embedding for files already in `processed/` and safely update existing records by `file_hash`.

```bash
cd /Users/pranjal/garage/smart_stack
source .venv/bin/activate
python ingest.py --safe-reprocess
```

Optional:

- `--limit N` process only first `N` files from `processed/` for controlled batches

### Use Nomic Embeddings

Use this to index by semantic meaning with Nomic:

```bash
cd /Users/pranjal/garage/smart_stack
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
cd /Users/pranjal/garage/smart_stack
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
cd /Users/pranjal/garage/smart_stack
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
cd /Users/pranjal/garage/smart_stack
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

- `/Users/pranjal/garage/smart_stack/SmartStackUI/Package.swift`
- `/Users/pranjal/garage/smart_stack/SmartStackUI/Sources/SmartStackUI/main.swift`

It supports:

- semantic and keyword image search
- source/score filters
- one-click `Ingest Inbox`, `Safe Reprocess`, and `Index Notes`
- opening result files directly
- live command logs
- guarded ingest buttons now route through `run_guarded_ingest.sh` (memory gate enabled)

Run via local compatibility wrapper (works around CLT Swift/SDK mismatch):

```bash
cd /Users/pranjal/garage/smart_stack/SmartStackUI
./local_run.sh
```

`local_run.sh` now skips rebuild when the binary is up-to-date.

Build only:

```bash
cd /Users/pranjal/garage/smart_stack/SmartStackUI
./local_run.sh --build-only
```

If you want to recreate the local SDK cache:

```bash
cd /Users/pranjal/garage/smart_stack/SmartStackUI
./local_run.sh --clean-sdk --build-only
```

Install a clickable macOS app bundle:

```bash
cd /Users/pranjal/garage/smart_stack/SmartStackUI
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
0 3 * * * caffeinate -i /Users/pranjal/garage/smart_stack/.venv/bin/python /Users/pranjal/garage/smart_stack/ingest.py >> /Users/pranjal/garage/smart_stack/night_shift.log 2>&1
```

## Important Paths

- Ingest input: `/Users/pranjal/garage/smart_stack/inbox`
- Ingest success archive: `/Users/pranjal/garage/smart_stack/processed`
- Ingest failures: `/Users/pranjal/garage/smart_stack/failed`
- SQLite metadata DB: `/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db`
- LanceDB vectors: `/Users/pranjal/Pranjal-Obs/clawd/vectors.lance`
- Obsidian media copies: `/Users/pranjal/Pranjal-Obs/clawd/Media`

## Operations

See `RUNBOOK.md` for checks, failure handling, reprocessing flow, and maintenance.
