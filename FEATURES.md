# Smart Stack Features

A comprehensive guide to the capabilities of your multimodal AI stack.

## 🟢 Live Features (Available Now)

### 1. **Index-in-Place (Phase 3)**

- **What it is**: Indexes your files directly where they live (e.g., in your Obsidian vault or Pictures folder) without moving or copying them.
- **How to use**:
  - **Single File/Folder**: Click "Ingest File/Folder" in the UI → Select items.
  - **Watched Folders**: Go to "Settings..." → Add folders to watch.
  - **Rescan**: Click "Rescan Changed" to update index if you edited files externally.
- **Why use it**: Saves disk space and keeps your files organized your way.

### 2. **Multimodal Search**

- **What it is**: Find images using natural language text or other images.
- **Modes**:
  - **Semantic**: "Show me receipts from March" (Uses CLIP/Text understanding).
  - **Visual (CLI Only)**: Search using an input image to find similar ones.
    - Command: `./mm_cli.py search --image-path path/to/photo.jpg`
  - **Hybrid**: Automatically blends text and visual matches based on your query.
- **How to use**: Type in the search bar.

### 3. **AI Chat & Reasoning**

- **What it is**: Have a conversation _with_ your documents/images.
- **How to use**: Switch to the **Chat** tab.
- **Examples**:
  - "Who sent this invoice and what is the total?"
  - "Summarize the key points from these diagrams."
  - "Does this screenshot show the new UI or the old one?"
- **Note**: Triggers the VLM (~7GB RAM usage) for deep reasoning.

### 4. **Compare Mode**

- **What it is**: Side-by-side analysis of _retrieved_ items.
- **How it works**: It strictly uses **text queries** to find the top 2 matches and then compares them. It does _not_ accept two image paths directly yet.
- **How to use (Chat/CLI)**:
  - **Query**: "Compare the old logo with the new logo"
  - **Query**: "Analyze the difference between the invoice from March and the one from April"
  - **CLI Command**: `./mm_cli.py compare "old logo vs new logo"`
- **Why use it**: Perfect for design reviews, document versioning, or identifying outliers.

### 5. **Intelligence Upgrades (Phase 2) - Advanced**

The following features work silently in the background to make the system smarter:

- **Cross-Image Comparison**:
  - _Logic_: Retrieve top 2 images for a concept and analyze differences/similarities using the VLM.
  - _Status_: ✅ Active (via `compare` command/intent).
- **Context Compression Layer**:
  - _Logic_: Filters OCR text to keep only relevant blocks (using confidence & query overlap) before sending to VLM.
  - _Benefit_: Reduces token usage and RAM spike duration.
  - _Status_: ✅ Active (in `chat.py`).
- **Retrieval Explainability**:
  - _Logic_: Analyze _why_ a result was chosen (text match vs visual match).
  - _Command_: `./mm_cli.py explain "query"`
  - _Status_: ✅ Active.
- **Adaptive Retrieval Depth**:
  - _Logic_: If query contains "compare", "analyze", or is long (>12 words), it automatically fetches more candidates (top-20) and reranks them.
  - _Status_: ✅ Active (in `search_engine.py`).
- **Automatic Image Classification**:
  - _Logic_: Assigns categories (Invoice, Diagram, Photo) during ingestion based on visual content.
  - _Status_: ✅ Active (in `ingestion.py` / `db.py`).
- **Region-Aware Answering**:
  - _Logic_: The Chat VLM can point to specific bounding boxes in the image for its answer.
  - _Status_: ✅ Active (in `chat.py`).
- **Session Awareness**:
  - _Logic_: Remembers your last query to handle follow-ups ("What about the second one?").
  - _Status_: ✅ Active (in `session.py`).
- **Personalization Learning**:
  - _Logic_: Tracks your routing choices (Visual vs Text) to adjust future search weights.
  - _Status_: ✅ Active (in `session.py`).

### 6. **Semantic Timeline (Phase 2D)**

- **What it is**: Time-bucket analytics over indexed media, with zoom from Year -> Month -> Day.
- **How to use**:
  - **UI**: Click the calendar icon (`Semantic Timeline`) in the top bar.
  - **Pinch**: Use trackpad pinch to zoom bucket granularity.
  - **CLI**: `./mm_cli.py timeline --granularity month --query "invoice"`
- **Why use it**: Understand when content clusters happened and navigate spikes quickly.

## 🟡 Planned / Up Next

- **[x] Visual Search UI**: Drag & Drop, pick, or paste a copied image in the search bar to find similar items.
- **[ ] Direct Image Comparison**: "Compare [this specific photo] with [that specific photo]" via drag-and-drop.
- **[ ] Smart Collections**: Save search results as dynamic collections.
- **[ ] Audio/Video Indexing**: Support for transcribing and searching video clips.
- **[ ] Desktop Daemon**: Background watcher that auto-indexes files without manual "Rescan".
