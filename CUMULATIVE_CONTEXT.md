# Cumulative Context (Smart Stack)

Last updated: 2026-02-16
Workspace: `/Users/pranjal/garage/smart_stack`

## 1) Current Repository State

- Branch: `master` tracking `origin/master`
- Recent commits:
  - `2a79567` Add documentation, session module, and ingestion scripts
  - `0875dc4` Add core multimodal stack components
  - `2e4b548` Index-in-Place + Intelligence upgrades
  - `3347966` initial `mm_stack` + CLI + indexer
- Working tree has substantial uncommitted changes.

### Modified tracked files
- `SmartStackUI/Sources/SmartStackUI/main.swift`
- `SmartStackUI/local_run.sh`
- `mm_cli.py`
- `mm_stack/config.py`
- `mm_stack/db.py`
- `mm_stack/ingestion.py`
- `mm_stack/lancedb_store.py`
- `mm_stack/models.py`
- `mm_stack/search_engine.py`
- `mm_stack/vlm_analyzer.py`
- plus regenerated `mm_stack/__pycache__/*.pyc`

### New untracked files/directories (key)
- `FEATURES.md`
- `SmartStackUI/Sources/SmartStackUI/CommandPalette.swift`
- `mm_stack/memory.py`
- `mm_stack/collections.py`
- `mm_stack/perception.py`
- runtime artifacts: `processed/`, `inbox/`, `feasibility_logs/`, `night_shift.log`

## 2) What Was Added (Cumulative)

## 2.1 Multimodal scope expanded beyond images

`mm_stack/config.py` now supports:
- Image: `png/jpg/jpeg/webp/heic/heif/bmp/tiff`
- Video: `mp4/mov/mkv/webm/avi`
- Audio: `mp3/wav/m4a/aac`

New constants:
- `IMAGE_EXTENSIONS`
- `VIDEO_EXTENSIONS`
- `AUDIO_EXTENSIONS`

## 2.2 DB schema expanded for "universal brain" objects

`mm_stack/db.py` adds tables:
- `videos`
- `video_segments`
- `clusters`
- `cluster_assignments`
- `collections`

Also adds upsert helpers:
- `upsert_video`
- `upsert_video_segment`
- `upsert_cluster`
- `upsert_cluster_assignment`
- `upsert_collection`

## 2.3 Ingestion pipeline now dispatches by media type

`mm_stack/ingestion.py` now:
- routes batch input into image/audio/video paths
- adds `_ingest_images`, `_ingest_videos`, `_ingest_audios`
- adds shared `_process_candidates` (CLIP -> VLM -> text embedding -> persist)
- allows non-visual candidates (`Candidate.is_visual=False`) for transcript-only segments

Video flow introduced:
- frame extraction via `VideoProcessor`
- optional audio extraction + transcription via `AudioProcessor`
- frame candidates indexed visually + textually
- transcript segment links persisted to `video_segments`

## 2.4 Search metadata enrichment for video hits

`mm_stack/search_engine.py` adds `_attach_video_metadata()`:
- joins `video_segments` + `videos`
- appends `video_id`, `start_time`, `end_time`, `video_path` to result rows

`mm_stack/models.py` extends result model with optional video fields.

## 2.5 New modules

- `mm_stack/perception.py`
  - `AudioProcessor` (Whisper)
  - `VideoProcessor` (MoviePy frame/audio extraction)
- `mm_stack/memory.py`
  - CLIP vector clustering (`MiniBatchKMeans`)
  - optional VLM cluster auto-labeling
- `mm_stack/collections.py`
  - create/list/evaluate/delete dynamic collections over SQLite metadata

## 2.6 CLI surface expanded

`mm_cli.py` new commands:
- `cluster --recalc --label --n-clusters`
- `collection-create <name> <query>`
- `collection-list`
- `collection-eval <name>`
- `collection-delete <name>`

Existing commands remain (ingest/search/chat/explain/compare/reembed/watch/exclude/rescan).

## 2.7 SwiftUI UX additions

`SmartStackUI` additions:
- singleton `SmartStackViewModel.shared`
- process cancellation for overlapping search/chat subprocesses
- command palette support bridge (`SearchResultItem`, `performSearch`, `ingestPath`)
- app delegate with hotkey monitor (Cmd+Shift+Space) and floating command palette window
- new file `CommandPalette.swift` with:
  - fast search box
  - debounced live search
  - click-to-open result file
  - drag-and-drop ingest

`SmartStackUI/local_run.sh` improved:
- robust SDK detection via `xcrun --show-sdk-path`
- compiles all Swift files in `Sources/SmartStackUI`
- incremental rebuild check across all source files
- Bash 3.2 compatibility adjustments

## 3) Current Runtime/Behavior Snapshot

- `mm_cli.py --help` works and shows all commands.
- `mm_cli.py search "test" -n 1` works and returns results from DB/Lance.
- Python compile check passed for `mm_cli.py` and all `mm_stack/*.py`.

Observed from runtime:
- Hybrid search path is active by default for plain queries.
- HF warning appears when unauthenticated (`HF_TOKEN` not set).

## 4) Important Risks / Regressions Currently in Code

These are present in the current working tree and should be treated as active issues:

1. `search_engine.py` duplicated `elif decision.mode == "text"` block.
   - First text branch sets `base_rows` only and does not assign `results`.
   - This can cause text-routed queries to return empty/incorrect output.

2. `search_engine.py` duplicates the hybrid source assignment loop.
   - Benign but redundant.

3. `ingestion.py` duplicate imports and duplicate imported symbols.
   - `upsert_image_metadata` imported twice.
   - `TextEmbedder` and utility imports duplicated.

4. `_process_candidates()` failure handling is incomplete.
   - Persistence exceptions are printed but not collected into returned `failed` list.
   - Returns `"failed": []` even when individual candidates fail.

5. `ingest_batch()` uses plain `dict.update()` across modality results.
   - If image/audio/video ingest all run in one call, keys may overwrite each other (`ingested`, `failed`, etc.).

6. Video/audio dedupe integrity is weak right now.
   - frame/transcript candidates use random UUID-based pseudo hash for `sha256_hash`, so content-level dedupe is not deterministic.
   - `videos.file_hash` currently written as empty string.

7. Dependency drift likely.
   - New modules require packages not in `pyproject.toml` dependencies (`numpy`, `scikit-learn`, `moviepy`, `openai-whisper`, potentially `torch` runtime considerations).

8. Python version mismatch signal.
   - `pyproject.toml` says `requires-python = ">=3.14"`.
   - active environment has been Python 3.11 in actual runs.

## 5) Data Paths / Storage Context

Configured in `mm_stack/config.py`:
- SQLite DB: `/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db`
- LanceDB: `/Users/pranjal/Pranjal-Obs/clawd/vectors.lance`
- Inbox: `/Users/pranjal/garage/smart_stack/inbox`
- Processed: `/Users/pranjal/garage/smart_stack/processed`
- Media vault dir: `/Users/pranjal/Pranjal-Obs/clawd/Media`

## 6) 2026-02-16 Incremental Update (Phase 2C + Stability Fixes)

### 6.1 Context Lens backend is now wired end-to-end

- Existing engine file `mm_stack/context_lens.py` is now integrated through public surfaces:
  - `mm_stack/api.py`: new `context_lens(...)`
  - `mm_cli.py`: new command `context-lens --image-id|--file-path -n`
- Context Lens returns:
  - target image metadata
  - ring groups: `similarity`, `cluster`, `entity`, `time`
  - `cluster_info`, `entity_terms`, and per-ring counts

### 6.2 Search engine regression fixed

- `mm_stack/search_engine.py` duplicated `elif decision.mode == "text"` branch was removed.
- Hybrid duplicate source assignment loop removed.
- Result: text-routed queries now reliably attach metadata and return results instead of falling through.

### 6.3 SmartStackUI now includes Phase 2C visualization

- `SmartStackUI/Sources/SmartStackUI/main.swift` additions:
  - Context Lens response models
  - ViewModel state + runner: `runContextLens(for:topK:)`
  - Result cards include a `Lens` action button
  - New `ContextLensSheet` + `ContextLensRingPlot`:
    - concentric rings
    - hover hints explaining why neighbors appear
    - selectable neighbors with quick open action

### 6.4 Roadmap status updates

- `PROJECT_STATE.md` updated:
  - Phase 2B marked complete
  - Phase 2C marked complete
  - Phase 2D marked complete

### 6.5 Semantic Timeline (Phase 2D) implementation

- New backend module: `mm_stack/timeline.py`
  - SQLite-native aggregation by:
    - `year` (`YYYY`)
    - `month` (`YYYY-MM`)
    - `day` (`YYYY-MM-DD`)
  - Optional text filter over indexed metadata fields:
    - `file_path`, `caption`, `summary`, `tags`, `ocr_structured`
  - Returns:
    - `total_items`, `bucket_count`, ordered `buckets`, and summary `stats`
- API/CLI wiring:
  - `mm_stack/api.py`: new `timeline(...)`
  - `mm_cli.py`: new command `timeline --granularity --query --limit`
- SwiftUI wiring in `SmartStackUI/Sources/SmartStackUI/main.swift`:
  - `SemanticTimelineSheet`
  - `TimelineBarPlot`
  - Pinch gesture zoom (Year ↔ Month ↔ Day)
  - Toolbar/menu entrypoints to open timeline
  - Bucket selection with optional "Open Sample" action

### 6.6 Visual Query UX (Next Phase) implementation

- Main UI supports image-to-image search directly:
  - drag-drop image onto search bar
  - picker button to choose a query image
  - clipboard paste (`Cmd+V`) for copied images
  - clear visual-query chip in header
- Explicit actions added:
  - `Paste Image as Visual Query`
  - `Paste Image and Ingest`
- Search routing changes:
  - if visual query image is set and Pro Stack is enabled, search runs via:
    - `mm_cli.py search --image-path ... --json`
  - empty text query is allowed for visual search mode
- Result mapping now carries `image_id` for multimodal cards, enabling Context Lens compatibility on those cards.

## 6) Architectural Intent vs Current Implementation

Intent that remains consistent:
- local-first multimodal ingestion/search
- SQLite metadata + LanceDB vectors
- sequential model loading in core image pipeline

New direction now visible:
- evolving from image-only stack into a broader "universal brain" (image + video + audio + clustering + collections + command palette UX)

Main gap to stabilize next:
- clean reliability pass on newly added media/cluster/search branches before further feature expansion.

## 7) Suggested Immediate Next Workstream

1. Normalize ingest result aggregation across mixed media in one batch.
2. Make failure reporting truthful in `_process_candidates`.
3. Add missing dependencies + lock Python version strategy.
4. Add a focused smoke-test script for: image ingest, video ingest, text query, clip query, hybrid query, cluster recalc.
5. Start Phase 2D timeline work once backend reliability pass is complete.

## 8) New Safety Guardrail (2026-02-16)

Added critical-RAM kill-switch to `/Users/pranjal/garage/smart_stack/SmartStackUI/local_run.sh`.

Behavior:
- Launches a background watchdog when SmartStackUI starts.
- Monitors memory every few seconds using:
  - Active+Wired MB from `vm_stat`
  - free-memory percentage from `memory_pressure -Q`
- Monitors duplicate SmartStackUI instance count (`pgrep -x SmartStackUI`).
- Prints preflight RAM hogs at startup (processes >= 500MB).
- Monitors Smart Stack venv Python fan-out count (`.../smart_stack/.venv/bin/python`) to catch Activity Monitor `Python 3.11` bursts.
- If critical condition is sustained, it hard-stops:
  - `SmartStackUI` app process
  - descendant subprocess tree
  - stack-related subprocess commands (`mm_cli.py`, `search.py`, `ingest.py`, `notes_index.py`, `run_guarded_ingest.sh`)
- Emits a top-RAM-hogs snapshot at trigger time for diagnosis.

Default thresholds:
- `kill_threshold_mb=15872` (~15.5GB Active+Wired; trigger uses `used_mb > threshold`)
- `kill_free_pct_threshold=8`
- `kill_python_count_threshold=6`
- `kill_breach_count=3`
- `kill_poll_sec=3`

CLI options:
- `--no-killswitch`
- `--kill-threshold-mb N`
- `--kill-free-pct N`
- `--kill-breach-count N`
- `--kill-poll-sec N`
- `--kill-python-count N`

Env overrides:
- `SMART_STACK_KILL_THRESHOLD_MB`
- `SMART_STACK_KILL_FREE_PCT`
- `SMART_STACK_KILL_BREACH_COUNT`
- `SMART_STACK_KILL_POLL_SEC`
- `SMART_STACK_KILL_PY_COUNT_THRESHOLD`

Validation done:
- Forced test: `./local_run.sh --kill-threshold-mb 1 --kill-breach-count 1 --kill-poll-sec 1`
- Result: watchdog triggered and terminated SmartStackUI + stack subprocesses as intended.

## 9) Clickable App Deployment Fix (2026-02-16)

Problem observed:
- Clicking `~/Applications/SmartStackUI.app` showed stale behavior while terminal run showed new features.

Root cause:
- Installed app bundle executable was an older copied Mach-O binary (`Contents/MacOS/SmartStackUI`) and was not auto-refreshing with new builds.

Fix applied:
- Reworked `/Users/pranjal/garage/smart_stack/SmartStackUI/install_app.sh` to install:
  - launcher script at `Contents/MacOS/SmartStackUI`
  - bundled binary at `Contents/Resources/SmartStackUI.bin`
- Launcher behavior:
  - auto-syncs `Contents/Resources/SmartStackUI.bin` from dev binary `SmartStackUI/.build-local/SmartStackUI` when newer
  - runs bundled binary directly (no compile on click)
  - includes kill-switch watchdog checks (RAM/free-memory/python fan-out)

Verification:
- App launch now starts:
  - `.../Applications/SmartStackUI.app/Contents/MacOS/SmartStackUI` (launcher)
  - `.../Applications/SmartStackUI.app/Contents/Resources/SmartStackUI.bin` (actual UI binary)
- SHA256 of bundled binary matches dev build binary.

## 10) MM-Only Consolidation (2026-02-16)

Objective executed:
- Remove legacy execution paths and make multimodal stack (`mm_cli.py` + `mm_stack/*`) the single active pipeline for ingest/search.

What changed:
- `run_guarded_ingest.sh`
  - now runs one-time memory gate at run start (not per image),
  - then executes `mm_cli.py ingest-inbox`,
  - supports `--safe-reprocess` and `--limit`,
  - ignores legacy-only args with warnings.
- `ingest.py`
  - replaced with compatibility shim that forwards to `mm_cli.py ingest-inbox`,
  - optional env pass-through for `--vlm-model` and `--embed-model`,
  - no legacy dual-write/legacy vector pipeline execution.
- `search.py`
  - replaced with compatibility shim that forwards to `mm_cli.py search`,
  - emits old marker payload format (`@@SMARTSTACK_JSON@@...`) for backward callers,
  - filters by `--min-score` in wrapper output.
- `openclaw_imgsearch.py`
  - switched from `search.py` backend to direct `mm_cli.py search`,
  - `--embed-model`/`--with-notes` now explicitly warned as ignored (MM query-time behavior).
- `SmartStackUI/Sources/SmartStackUI/main.swift`
  - removed legacy/pro-stack toggle and legacy semantic path,
  - semantic search always uses multimodal search,
  - visual query always uses multimodal image search,
  - safe reprocess/inbox ingest always use guarded MM ingest script,
  - keyword search now queries MM `images` table (`file_path/caption/summary/tags/ocr_structured`),
  - removed note-index action from primary UI/menu for MM-only consistency.

Validation performed:
- Python compile check passed for `ingest.py`, `search.py`, `openclaw_imgsearch.py`, `mm_cli.py`.
- `run_guarded_ingest.sh` skip-mode behavior verified (exits cleanly without ingest when above threshold).
- `search.py horse --json` verified to return MM-backed JSON marker payload.
- `openclaw_imgsearch.py horse` verified to return MM-backed results.
- `SmartStackUI/local_run.sh --build-only` succeeded after migration edits.

## 11) Continuous Chat + Attached Image Focus (2026-02-16)

Objective executed:
- Move chat from stateless one-shot behavior to practical continuous multi-turn UX in SmartStackUI.
- Add explicit "attach one result image and continue chatting on it" capability.

Backend/CLI changes:
- `mm_cli.py chat` now supports:
  - `--image-id` (pin indexed image as focus context)
  - `--file-path` (fallback pin by path)
  - `--history-json` (conversation history payload)
  - `--history-file` (history payload file path; used by UI to avoid huge command-line args)
- `mm_stack/api.py` chat/stream wrappers now pass:
  - `attached_image_id`
  - `attached_file_path`
  - `history`
- `mm_stack/chat.py` now:
  - normalizes and injects conversation history into prompt context,
  - resolves attached image from SQLite `images` table and prepends it to retrieval context,
  - dedupes attached image from retrieval list and keeps memory-safe image batch behavior.

UI changes (`SmartStackUI/Sources/SmartStackUI/main.swift`):
- Added persistent chat turns (`chatTurns`) for multi-turn conversation display.
- Added pinned chat image state (`attachedChatImage`) with clear action.
- `runChat()` now:
  - sends last turns as `--history-json`,
  - sends pinned image via `--image-id` or `--file-path`,
  - appends user + assistant turns to on-screen history (not replacing prior answer each turn).
- Result cards now expose "Chat" attach action (button + context menu).
- Header now shows an attached-image chip in chat mode.
- Added "Clear Attached Chat Image" and "Clear Chat Conversation" actions.

Validation performed:
- `mm_cli.py chat --help` confirms new chat flags.
- `mm_cli.py chat ... --history-json ... --image-id ... --json` returns grounded response using pinned image as first source.
- Python compile check passed for `mm_cli.py`, `mm_stack/api.py`, `mm_stack/chat.py`.
- `SmartStackUI/local_run.sh --build-only` succeeded after UI updates.

### Follow-up fix: attached-focus chat relevance bug

Issue:
- Attached-image chat could still return lexical fallback summaries from unrelated retrieved images (e.g., matching generic words like "have"), and miss visual answers.

Fixes in `mm_stack/chat.py`:
- Added strict focus mode when image is attached (unless query explicitly asks compare/vs/similar).
- In focus mode, only attached image is used as retrieval context.
- Relaxed post-answer grounding override in focus mode so metadata-token mismatch does not force unrelated fallback text.
- Added small query normalization (`jwellery -> jewelry`) for model prompt robustness.
- Expanded stopword list to suppress weak lexical matches from helper verbs.

Validation:
- Re-ran failing command with:
  - `--image-id f4e3f455-46ec-4541-859e-1dde75fab97e`
  - `--history-file .../history_1771255208489.json`
- Output now correctly answers jewelry is present, with attached image as the only source.

### Follow-up fix: direct query `jwellery` (no attachment)

Issue:
- Direct `mm_cli.py chat jwellery` could still return `Not found` despite `spiritual_art_2.jpg` being in retrieval results.

Root causes:
- Chat context previously underused semantic fields for match gating.
- Retrieval spelling normalization (`jwellery` -> `jewelry`) degraded ranking for this index.

Fixes:
- Kept retrieval query as user-typed string; use normalized spelling only for model prompt and matching.
- Included `summary` in:
  - overlap scoring (`_row_query_overlap`)
  - retrieval match score (`_retrieval_query_match_score`)
  - assembled context text sent to VLM
- Matching now uses combined query tokens (`original + normalized`) for robust typo handling.

Validation:
- Re-ran:
  - `mm_cli.py chat jwellery --json -n 6 --history-file ...`
- Output now returns spiritual image as source and answers jewelry present.

## 12) Dynamic Typo-Robust Retrieval (2026-02-16)

Objective executed:
- Replace hardcoded typo patches in search ranking with a dynamic, generic typo-aware rerank layer.

New module:
- `mm_stack/query_normalization.py`
  - `normalize_query(raw: str) -> NormalizedQuery`
  - `fuzzy_match_score(query_tokens, text) -> float`
  - `combined_rank(candidates, query) -> sorted_candidates`

Behavior:
- Preserves raw query for vector retrieval intent.
- Normalizes text/tokens (case + Unicode NFKC + punctuation split).
- Builds candidate vocabulary dynamically from retrieved metadata (caption/summary/OCR/tags).
- Computes typo-aware fuzzy overlap (RapidFuzz preferred, difflib fallback).
- Optional phonetic assist via local Soundex for latin-script terms.
- Combines scores:
  - `final_score = alpha * vector_similarity_norm + beta * fuzziness_score`
  - defaults: `alpha=0.8`, `beta=0.2`
  - configurable via config/env.

Integration:
- `mm_stack/search_engine.py` now routes lexical rerank through `combined_rank(...)`.
- `mm_stack/config.py` adds:
  - `fuzzy_alpha`
  - `fuzzy_beta`
  - `fuzzy_ratio_threshold`
  - `fuzzy_min_combined_score`

Dependencies:
- `pyproject.toml` updated with `rapidfuzz`.
- Installed in venv: `rapidfuzz==3.14.3`.

Tests:
- Added `tests/test_query_normalization.py` (unittest) covering:
  - `jwellery` ≈ `jewelry`
  - `reciept` ≈ `receipt`
  - `enviroment` ≈ `environment`
  - combined-rank promotion of fuzzy-relevant candidate without hard filtering.
- Added `tests/test_query_fuzzy_metrics.py` (unittest) covering:
  - precision@k / recall@k checks
  - fuzzy-scoring influence vs baseline vector-only ordering.
- Test run:
  - `python -m unittest discover -s tests -p 'test_query_*.py'` -> OK.

Validation:
- `mm_cli.py search jwellery -n 8 --json` now ranks `spiritual_art_2.jpg` first.
- `mm_cli.py chat jwellery --json ...` remains correct with spiritual image source.

## 13) Phase 1 Policy Foundation (2026-02-16, latest)

Objective executed:
- Move from per-word patching to deterministic query-policy control with bounded adaptivity and shared Search/Chat behavior.

Implemented architecture:
- Added explicit query typing + confidence in planner intent output:
  - `generic`, `constrained`, `attribute`, `identity`
  - `policy_confidence_score` for fallback safety.
- Added centralized adaptive policy module:
  - `mm_stack/query_policy.py`
  - policy fields include:
    - similarity gate
    - top-k multiplier
    - lexical mode (`boost` vs `enforce`)
    - presence gating behavior
    - CLIP/Text hybrid weights.
- Search and Chat now share the same policy output path (single source of retrieval gating behavior).

Files changed (Phase 1):
- `mm_stack/intent_types.py`
- `mm_stack/query_planner.py`
- `mm_stack/query_policy.py` (new)
- `mm_stack/config.py`
- `mm_stack/search_types.py`
- `mm_stack/search_engine.py`
- `mm_stack/chat.py`
- `tests/test_query_planner.py`
- `tests/test_query_policy.py` (new)

New config/env controls:
- `SMART_STACK_ADAPTIVE_POLICY_ENABLED`
- `SMART_STACK_LEGACY_PATCHES_ENABLED`
- `SMART_STACK_POLICY_BASE_SIMILARITY_GATE`
- `SMART_STACK_POLICY_GATE_ADJUSTMENT_MAX` (bounded adaptivity)
- `SMART_STACK_POLICY_MAX_TOPK_MULTIPLIER` (bounded expansion)
- `SMART_STACK_POLICY_CONFIDENCE_FALLBACK_THRESHOLD`

Guardrails implemented:
- Query-type set intentionally minimal for v1 (`generic|constrained|attribute|identity`).
- Gate adjustment bounded by configured max (default target range ±0.07).
- Top-k expansion bounded (default cap 2x).
- Low-confidence policy classification falls back to generic behavior.
- Legacy word patches retained behind feature flag (not force-removed).

Validation:
- Compile checks passed for touched modules.
- Unit tests passed:
  - `python -m unittest tests.test_query_planner tests.test_query_policy tests.test_search_rerank tests.test_intent_ranking`
  - Result: `Ran 17 tests ... OK`.

Current branch + state snapshot:
- Branch: `master`
- Working tree: still heavily dirty (expected in this repo), including many prior unrelated modifications and untracked files.
- This context section is the latest authoritative note for Phase 1 policy foundation status.
