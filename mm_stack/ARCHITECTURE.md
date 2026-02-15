# Multimodal Architecture (Nomic + OpenCLIP)

## Directory Structure

```text
smart_stack/
├── mm_cli.py                         # CLI entrypoint
├── mm_stack/
│   ├── __init__.py
│   ├── api.py                        # clean API: ingest_image(), search(), reembed_all(), evaluate()
│   ├── config.py                     # paths, model ids, schema versions, dimensions
│   ├── models.py                     # OCRBlock, PreparedImage, VLMOutput
│   ├── utils.py                      # hash, JSON, cache cleanup helpers
│   ├── preprocess.py                 # orientation normalize + resize <= 1024 + hash
│   ├── ocr.py                        # structured OCR blocks with bbox/confidence/type
│   ├── clip_embedder.py              # OpenCLIP ViT-B/32 load/unload + image/text encoding
│   ├── vlm_analyzer.py               # deterministic caption/summary/tags generation
│   ├── text_embedder.py              # Nomic text embeddings (768-dim)
│   ├── db.py                         # SQLite schema + CRUD + logging
│   ├── lancedb_store.py              # clip_index + text_index vector storage
│   ├── ingestion.py                  # production ingestion orchestration
│   ├── router.py                     # deterministic query router
│   ├── fusion.py                     # normalized weighted hybrid scoring
│   ├── search_types.py               # typed search response
│   ├── search_engine.py              # query execution + logging + memory-safe model loading
│   ├── reembed.py                    # version mismatch detection + safe batch re-embedding
│   ├── evaluation.py                 # benchmark harness (precision@5, recall@10, avg sim)
│   └── evaluation/
│       ├── benchmark_cases.json      # 20-case evaluation fixture template
│       └── test_images/              # test image set for harness
└── ... (existing legacy scripts remain)
```

## Memory-Safe Execution Order

### Ingestion order (strict, sequential)

1. Preprocess + SHA256 + OCR (no CLIP/VLM/text model loaded)
2. Load CLIP -> embed image batch -> unload CLIP
3. Load VLM -> caption/summary/tags batch -> unload VLM
4. Load Nomic text embedder -> embed text payload batch -> unload text embedder
5. Persist metadata (SQLite) + vectors (LanceDB)

Constraint satisfied: CLIP, VLM, and text embedder are never loaded together.

### Search-time order (on demand)

- image-to-image: load CLIP only
- text OCR-intent: load text embedder only
- text visual-intent: load CLIP text encoder only
- hybrid text: load CLIP for top-20, unload; then load text embedder for top-20, unload; fuse

## Ingestion Pseudocode

```python
candidates = []
for image in incoming_images:
    prepared = preprocess(image)                    # normalize orientation, max 1024, sha256
    if exists_by_hash(prepared.sha256) and not safe_reprocess:
        continue
    ocr_blocks, ocr_conf = extract_ocr_structured(prepared)
    candidates.append({prepared, ocr_blocks, ocr_conf})

with CLIP() as clip:
    clip_vectors = clip.encode_images([c.prepared for c in candidates])

with VLM(temp=0.0, top_p=1.0) as vlm:
    analyses = [vlm.analyze(c.prepared, c.ocr_blocks) for c in candidates]

payloads = [analysis.caption + "\n" + analysis.summary + "\n" + ocr_text(c.ocr_blocks)]
with NomicTextEmbedder() as text_model:
    text_vectors = text_model.encode(payloads)

for candidate in candidates:
    upsert_images_sqlite(...)
    upsert_clip_vector_lancedb(...)
    upsert_text_vector_lancedb(...)
```

## SQL Schema

```sql
CREATE TABLE images (
  id TEXT PRIMARY KEY,
  file_path TEXT NOT NULL,
  sha256_hash TEXT NOT NULL UNIQUE,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  caption TEXT NOT NULL DEFAULT '',
  summary TEXT NOT NULL DEFAULT '',
  tags TEXT NOT NULL DEFAULT '[]',
  ocr_structured TEXT NOT NULL DEFAULT '[]',
  ocr_confidence_avg REAL NOT NULL DEFAULT 0.0,
  schema_version TEXT NOT NULL,
  embedding_model_clip TEXT NOT NULL,
  embedding_model_text TEXT NOT NULL,
  embedding_dimension_clip INTEGER NOT NULL,
  embedding_dimension_text INTEGER NOT NULL,
  embedding_schema_version_clip TEXT NOT NULL,
  embedding_schema_version_text TEXT NOT NULL,
  text_payload_hash TEXT NOT NULL DEFAULT '',
  clip_content_hash TEXT NOT NULL DEFAULT '',
  is_stale INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE clip_vectors (
  id TEXT PRIMARY KEY,
  image_id TEXT NOT NULL UNIQUE,
  embedding_model_name TEXT NOT NULL,
  embedding_dimension INTEGER NOT NULL,
  embedding_schema_version TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(image_id) REFERENCES images(id)
);

CREATE TABLE text_vectors (
  id TEXT PRIMARY KEY,
  image_id TEXT NOT NULL UNIQUE,
  embedding_model_name TEXT NOT NULL,
  embedding_dimension INTEGER NOT NULL,
  embedding_schema_version TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(image_id) REFERENCES images(id)
);

CREATE TABLE search_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query TEXT NOT NULL,
  routing_decision TEXT NOT NULL,
  latency_ms INTEGER NOT NULL,
  result_ids TEXT NOT NULL,
  timestamp TEXT NOT NULL
);
```

## Query Router and Hybrid Scoring

Deterministic rules:

- image input -> CLIP only
- text with OCR keywords -> text_index only
- text with visual keywords -> clip_index only
- else -> hybrid

Hybrid:

- fetch top-20 CLIP + top-20 text
- normalize each index by its own max score
- final score = 0.6 * clip + 0.4 * text
- return top-10

Normalization is required because CLIP and text indexes have different raw score distributions; without normalization, one index can dominate rankings due to numeric scale alone.

## Versioning and Reprocessing

- Store model name + dimension + schema version per image and vector metadata
- `mark_stale_if_versions_mismatch()` flags rows where versions/models differ
- `reembed_all()` re-embeds stale/mismatched entries
- text re-embedding is skipped unless payload hash changes or model/schema mismatch

## Future Hooks

- TODO: cross-modal VLM reranking for top-N candidates
- TODO: dynamic query routing weights
- TODO: multilingual OCR expansion rules
- TODO: CLIP model migration B/32 -> B/16 with staged re-embedding
