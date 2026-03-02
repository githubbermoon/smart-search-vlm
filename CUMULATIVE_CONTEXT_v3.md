# Cumulative Context v3 (Smart Stack)

Last updated: 2026-02-23
Workspace: `/Users/pranjal/garage/smart_stack`

## 1) Current Snapshot

- Branch: `master`
- Working tree: intentionally dirty with many local edits and untracked files.
- Python runtime: `3.11` (`.venv`)
- Core data paths:
  - SQLite: `/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db`
  - LanceDB: `/Users/pranjal/Pranjal-Obs/clawd/vectors.lance`
- UI app bundle path currently used:
  - `/Users/pranjal/Applications/SmartStackUI.app`

## 2) High-Impact Features Already in Place (from prior context)

- Hybrid auto search routing (`legacy` + `hybrid`) with weighted RRF fusion.
- Cross-encoder rerank + retrieval confidence + abstain gate in chat.
- FTS5 keyword index and text embedding daemon path.
- Context Lens and Semantic Timeline in SmartStackUI.
- Critical malformed VLM metadata repair (`caption: "{"` issue) and tests.

## 3) New Changes Implemented in This Session

### 3.1 Memory Clusters + UI Surfacing

Implemented cluster APIs and UI integration end-to-end:

- Backend/API:
  - `mm_stack/memory.py`
    - `list_clusters(...)`
    - `get_cluster_items(...)`
    - recluster now clears old `clusters` and `cluster_assignments` before recompute.
  - `mm_stack/api.py`
    - `cluster_recalc(...)`
    - `cluster_label(...)`
    - `cluster_list(...)`
    - `cluster_items(...)`
- CLI:
  - `mm_cli.py cluster` now supports:
    - `--auto`
    - `--list`
    - `--items <cluster_id>`
    - `--limit`
    - `--min-items`
- SmartStackUI:
  - Added clusters state/loaders in `SmartStackViewModel`.
  - Added `MemoryClustersSheet` to browse clusters and member photos.
  - Added top-bar + menu entries for clusters and auto cluster actions.

Note: current `Auto Cluster` path still triggers recluster + auto-label (VLM), which is intentionally heavy.

### 3.2 Chat Crash Fix for Missing Image Files

Problem observed:

- `chat student --json` crashed with:
  - `ValueError: The image ... must be a valid URL or existing file.`

Root cause:

- stale `file_path` rows in DB (image moved/deleted on disk), chat attempted to load missing file.

Fix in `mm_stack/chat.py`:

- Skip stale/missing image files before VLM load.
- `_resize_image_if_needed(...)` now returns empty for missing input path.
- Attached-image handling now ignores attached entries missing on disk.
- If all retrieved files are missing, chat returns deterministic low-confidence message instead of raising.

### 3.3 All Indexed Photos View (Blind-Spot Fix)

Added new full-index listing path and UI button:

- API:
  - `mm_stack/api.py` -> `photos_list(...)`
- CLI:
  - `mm_cli.py` -> new `photos-list`
- UI:
  - `SmartStackUI` -> `runAllPhotos(...)`
  - New top-bar icon/button: `All Indexed Photos`
  - Menu entry also added.

### 3.4 All Photos Performance Fix

Issue:

- `All Indexed Photos` initially felt slow.

Fix:

- `photos_list(...)` now defaults to fast mode (no per-file `Path.exists()` checks).
- Added explicit slow-mode CLI switch for path checks:
  - `--check-exists`
- UI `runAllPhotos(...)` default load size reduced from 500 -> 180 for faster first paint.

### 3.5 Header UX Improvement

Added tiny text labels above top header buttons in SmartStackUI:

- `Mode`, `Kill`, `Timeline`, `Clusters`, `All Photos`, `Visual`, `Paste`, `Settings`, `Expand`.

### 3.6 Dynamic Path Relink for Moved Files

Requested behavior:

- if image path changes on macOS, DB should update.

Implemented hash-based relink during ingest/rescan ingest pass:

- New DB helper:
  - `mm_stack/db.py` -> `update_image_file_location(...)`
- Ingestion patch:
  - `mm_stack/ingestion.py` in `_ingest_images(...)`
  - If same `sha256_hash` exists but with different path and not reprocessing:
    - update `file_path`, inode/size/mtime,
    - clear stale flag,
    - count as `relinked_paths`.

Behavior now:

- Path updates when moved file is re-seen via `ingest-path` or `rescan-all` watched scan.
- Not yet true real-time FS event sync; it is scan/ingest-driven.

## 4) Tests Added/Run in This Session

Added tests:

- `tests/test_memory_clusters.py`
- `tests/test_chat_missing_paths.py`
- `tests/test_photos_list.py`
- `tests/test_path_relink.py`

Validated passing (targeted runs):

- `python -m unittest tests.test_memory_clusters`
- `python -m unittest tests.test_chat_abstain tests.test_chat_missing_paths`
- `python -m unittest tests.test_photos_list`
- `python -m unittest tests.test_path_relink`

## 5) Practical Notes for Next Thread

1. `Auto Cluster` in UI is expensive by design today because it includes VLM labeling.
   - Use `cluster --recalc` first for quick clustering.
   - Use `cluster --label` only when needed.
2. Chat now avoids crashing on stale image paths, but stale rows should still be cleaned.
3. Path relink is implemented for duplicate-hash moved files during ingest scans.

## 6) Suggested New-Thread Start Prompt

```md
Task: [one exact objective]

Workspace: /Users/pranjal/garage/smart_stack
Read first: /Users/pranjal/garage/smart_stack/CUMULATIVE_CONTEXT_v3.md

Scope (edit only):
- [absolute path 1]
- [absolute path 2]

Repro:
1. [exact command]
2. [exact command]

Observed:
- [paste exact stderr/JSON snippets]

Acceptance criteria:
1. [expected behavior]
2. [non-regression]
3. [tests/commands that must pass]
```

## 7) Quick Resume Commands

```bash
cd /Users/pranjal/garage/smart_stack

git branch --show-current
git status --short

# Fast all-photo index listing (default fast mode)
./.venv/bin/python mm_cli.py photos-list --limit 50

# Slow mode with existence checks
./.venv/bin/python mm_cli.py photos-list --limit 50 --check-exists --exclude-missing

# Quick cluster recompute only
./.venv/bin/python mm_cli.py cluster --recalc --n-clusters 12

# Optional expensive labeling
./.venv/bin/python mm_cli.py cluster --label

# Refresh stale/moved paths from watched folders
./.venv/bin/python mm_cli.py rescan-all

# Chat sanity after stale-path fix
./.venv/bin/python mm_cli.py chat "student" --json -n 3

# Targeted tests
./.venv/bin/python -m unittest \
  tests.test_memory_clusters \
  tests.test_chat_missing_paths \
  tests.test_photos_list \
  tests.test_path_relink
```
