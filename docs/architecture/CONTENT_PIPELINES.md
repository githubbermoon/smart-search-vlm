# Content Pipeline Architecture

## Purpose

Smart Stack is one product with two ingestion pipelines. Images and documents share storage and retrieval contracts, but they do not share assumptions about extraction, embeddings, or verification.

```text
                         ┌──────────────────────┐
Input ── type routing ──▶│ ContentItem contract │
                         └──────────┬───────────┘
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌──────────────────┐           ┌────────────────────┐
          │ Image pipeline   │           │ Document pipeline  │
          │ OCR + visual     │           │ extraction + chunks│
          │ analysis + CLIP  │           │ + text embeddings  │
          └────────┬─────────┘           └──────────┬─────────┘
                   └──────────────┬─────────────────┘
                                  ▼
                    shared metadata and indexes
                                  ▼
                    unified search with type filters
```

## Stable shared contract

A searchable item should expose a content type, stable source identifier, source path, display title, searchable text, timestamps, schema version, and type-specific metadata. A document chunk additionally needs a stable parent-document identifier, section label, and chunk index.

Shared code may route, store, filter, and format these items. It must not assume every path is decodable as an image or every item has a CLIP vector.

## Image pipeline

The image pipeline owns:

1. decoded-image validation;
2. orientation and size normalization;
3. Apple Vision OCR;
4. visual caption, summary, and tags;
5. image and text embeddings;
6. image-specific verification;
7. camera/gallery uploads and image previews.

Existing image identifiers, vectors, ranking behavior, and command contracts are compatibility boundaries.

## Document pipeline

The document pipeline owns:

1. document type detection and parsing;
2. page, slide, sheet, section, or plain-text extraction;
3. deterministic chunking;
4. parent/chunk identifiers;
5. text embeddings;
6. text-based relevance verification;
7. atomic replacement and deletion;
8. document icons and previews.

Initial supported formats may include PDF, TXT, Markdown, CSV, TSV, JSON, HTML, DOCX, PPTX, and XLSX. Each parser must fail clearly when optional format support is unavailable.

## Verification boundary

Image verification may inspect pixels through the visual model. Document verification must operate on extracted text and metadata. The search layer must dispatch by `content_type`; it must never convert a document-verifier error into a negative visual judgment.

If no appropriate verifier is available, retain the retrieval score and mark the result unverified. Do not apply a failure penalty for a verifier that was not applicable.

## Atomic document replacement

Reindexing an edited document is a single logical operation:

1. extract and validate every new section;
2. produce every new chunk and embedding in a staged generation;
3. persist the staged SQLite rows and vector records;
4. make the new generation visible only after all writes succeed;
5. retire the previous generation;
6. clean up abandoned staged data after failure.

Search must observe either the complete old generation or the complete new generation. SQLite and LanceDB identifiers should include enough generation information to reconcile interrupted operations.

## Search and presentation

Unified search may return both content types. Every result must carry `content_type`, and clients should support `all`, `image`, and `document` filters.

- Search may return the most relevant document chunks.
- Library browsing should normally show one card per source document.
- Grounded chat may use retrieved document text directly.
- Missing local source files should produce a stale/missing state, not crash the query.

## Workstream integration

- Image-specific work is integrated through `feature/image-pipeline`.
- Document-specific work is integrated through `feature/document-pipeline`.
- Shared contracts land in `master` first.
- Both workstreams must regularly synchronize from `master`.
- A workstream is merged into `master` only when its required tests and compatibility checks pass.

The branches organize development; the module and contract boundaries provide the lasting separation.
