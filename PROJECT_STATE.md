# Smart Stack - Project State & Roadmap

This document provides a consolidated view of the current project state, completed milestones, and immediate roadmap to facilitate context sharing.

## 🚀 Project Overview

Smart Stack is being transformed into a **Mac-Native AI OS Extension**. It combines a powerful multimodal intelligence backend (Vision, Audio, Search) with deep macOS integration (Spotlight-style Command Palette, Menu Bar, Finder hooks).

---

## ✅ Completed Tasks (Checklist)

### Phase 1-4: Intelligence Infrastructure (Done)

- [x] **Multimodal Backend**: CLIP-based image search, Whisper audio processing, and frame extraction for video.
- [x] **Universal Brain**: Online K-Means clustering for visual memory and VLM-based auto-labeling.
- [x] **Smart Ingestion**: Index-in-place architecture with file-system watching and "Watched Folders" management.
- [x] **Chat RAG**: Full Retrieval-Augmented Generation for natural language Q&A over local visual data.

### Phase 5: Native OS Extension

#### Phase 2A: Core Interaction (Done)

- [x] **Floating NSPanel**: Implemented `CommandPaletteWindow` (Spotlight-style floating bar).
- [x] **Global Hotkey**: Hooked `Cmd+Shift+Space` via native `NSEvent` monitoring.
- [x] **Zero-Friction Ingest**: Implemented native Drag & Drop handler for the palette.
- [x] **Performance Stabilization**:
  - [x] Implemented **Debouncing** (900ms) with query-length guard (`>=3`) and duplicate-query skip.
  - [x] Implemented **Active Process Management** (serial command queue + stale search/chat process termination).
  - [x] Implemented **Single-Instance Launch Guard** (prevents accidental multi-instance SmartStackUI fan-out).
  - [x] Implemented **Critical Memory Kill-Switch Watchdog**:
    - [x] Default trigger above ~15.5GB Active+Wired memory (`>15872MB`) with sustained breach logic.
    - [x] Free-memory pressure guard (`memory_pressure -Q`) and duplicate SmartStack instance checks.
    - [x] Smart Stack venv Python fan-out guard (covers Activity Monitor `Python 3.11` bursts).
    - [x] Automatic teardown of SmartStackUI + stack subprocesses + RAM-hog snapshot logging.
- [x] **Build System**: Resolved macOS SDK mismatches and updated `local_run.sh` for multi-file Swift project structure.

---

## 🛠 Current Implementation Details

### Build & Run

- **Script**: `SmartStackUI/local_run.sh`
- **Fixes**: Automatically detects SDK via `xcrun`, patches `.swiftinterface` version stamps, and compiles all source files in `Sources/SmartStackUI`.
- **Runtime Guardrails**:
  - Single-instance reuse by default (unless explicitly overridden).
  - Kill-switch watchdog enabled by default with configurable thresholds.
  - Preflight RAM-hog scan at launch + breach-time diagnostics.
- **Clickable App Deployment**:
  - `install_app.sh` installs launcher-wrapper + bundled binary (`Contents/Resources/SmartStackUI.bin`).
  - Click-launch does **not** compile; launcher auto-syncs bundled binary from latest `.build-local` build when newer.

### Core Components

| Component           | File                   | Description                                                                                  |
| :------------------ | :--------------------- | :------------------------------------------------------------------------------------------- |
| **Command Palette** | `CommandPalette.swift` | Native `NSPanel` UI with search input and results list.                                      |
| **ViewModel**       | `main.swift`           | `SmartStackViewModel` (Singleton) manages backend Python process lifecycle, throttled search, and serialized command execution. |
| **App Entry**       | `main.swift`           | `SmartStackUIApp` manages window state and Menu Bar presence.                                |
| **Process Safety**  | `local_run.sh`         | Startup memory guardrails, duplicate-instance prevention, and emergency kill-switch watchdog. |

---

## 📅 Immediate Roadmap (Next Steps)

### Phase 2B: Grid Stabilization & Performance

- [x] **Masonry Grid**: Result grid upgraded with card-based masonry lanes in `ContentView`.
- [x] **Async Caching**: Async thumbnail loading enabled in cards/palette (backed by system URL loading cache).
- [x] **Low-Res Previews**: UI now uses constrained previews and avoids full-size eager rendering in list flows.

### Phase 2C: Context Lens

- [x] **Relationship Engine**: Backend API/CLI returns neighbours by similarity, cluster, entity, and time.
- [x] **Circular Visualization**: SwiftUI Context Lens sheet renders concentric relationship rings with hover hints.

### Phase 2D: Semantic Timeline

- [x] **Pinch-to-Zoom**: Timeline sheet with Year -> Month -> Day aggregation and magnification zoom controls.

### Phase 3A: Visual Query UX

- [x] **Image-to-Image UI Search**: Main UI now supports image-path search via picker and drag-drop.
- [x] **Mode Routing**: Visual query auto-routes through multimodal CLIP search (`mm_cli.py search --image-path`).
- [x] **Result Interop**: Visual-search results preserve `image_id`, enabling Context Lens on eligible cards.

### Phase 3B: Intent-Aware Multimodal Engine (Done)

- [x] **Deterministic Query Planner v2**: Added retrieval/relation/attribute/presence decomposition with additive backward compatibility.
- [x] **Staged Intent Rerank**: Added component-scored reranking pipeline (vector + semantic + attribute + relation + presence).
- [x] **Structured Entity Memory Schema**: Added `image_entities`, `entity_attributes`, `image_relations`, and `entity_mentions` tables with indexes.
- [x] **Ingest Entity/Relation Persistence**: VLM output now supports structured entities/attributes/relations/mentions and persists them per image.
- [x] **Low-Confidence Verification Layer**: Added optional top-k local VLM verification guarded by confidence threshold.
- [x] **Explainable Search Output**: Search JSON now includes `query_intent`, `component_scores`, `relation_evidence`, `attributes`, and confidence explanation.
- [x] **CLI Backward-Compatible Expansion**: Added `--intent-debug` for explicit planner decomposition output.
- [x] **Phase-3 Test Coverage**: Added scenario-driven tests for planner, rerank behavior, and intent ranking.
