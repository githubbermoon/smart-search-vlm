# Contributing to Smart Stack

Thank you for helping improve Smart Stack. This guide explains how to propose, implement, test, and review changes while keeping the local-first image experience stable.

## Project priorities

Smart Stack is a privacy-first, local multimodal search application for macOS and Apple Silicon. Its stable capability is image ingestion, understanding, retrieval, and grounded interaction. Document ingestion is developed as a separate capability and must not weaken or silently change image behavior.

The project follows these principles:

1. User content stays local unless a user explicitly configures an external integration.
2. `master` must remain demonstrable and releasable.
3. Image and document ingestion use separate processing and verification paths.
4. Shared storage and search contracts must be typed, tested, and backward compatible.
5. Pull-request checks must not download large runtime models.
6. Changes that affect indexes or stored metadata need a migration or compatibility plan.

Read [the content-pipeline architecture](docs/architecture/CONTENT_PIPELINES.md) before changing ingestion, indexing, verification, or mixed-content search.

## Development workstreams

The repository uses two integration branches while the document capability matures:

| Branch | Purpose | Must preserve |
|---|---|---|
| `feature/image-pipeline` | Image ingestion, OCR, visual analysis, image embeddings, image upload, and image verification | Existing image search and mobile ingestion behavior |
| `feature/document-pipeline` | PDF and office-document extraction, chunking, text embeddings, document replacement, and document retrieval | The complete image pipeline and shared search contracts |

These are coordination branches, not separate products. `master` remains the single stable product branch.

Choose the correct base for your task:

- Branch from `feature/image-pipeline` for image-specific changes.
- Branch from `feature/document-pipeline` for document-specific changes.
- Branch from `master` for shared infrastructure, security, documentation, CI, storage contracts, or fixes affecting both pipelines.
- Land shared changes in `master` first, then synchronize both workstream branches. Do not copy the same shared commit independently into both branches.

Use a short-lived branch with a descriptive name:

```text
image/improve-ocr-confidence
documents/atomic-reindex
fix/missing-file-search
docs/local-installation
test/mixed-content-ranking
```

Branch names and pull-request metadata must describe the product change. Do not include personal names, editor names, automation names, or development-tool branding.

## Prerequisites

- macOS 13 or newer
- Apple Silicon for native MLX execution
- Python 3.14
- `uv`
- Xcode or compatible Apple Command Line Tools
- Git and a GitHub account

Most unit tests use fakes and temporary storage, so contributors can validate ordinary changes without downloading production models.

## Development setup

Fork the repository, clone your fork, and add the upstream repository:

```bash
git clone https://github.com/YOUR-USERNAME/smart-search-vlm.git
cd smart-search-vlm
git remote add upstream https://github.com/githubbermoon/smart-search-vlm.git
git fetch upstream
```

Create the locked development environment:

```bash
uv sync --locked --all-groups
```

Confirm the basic toolchain:

```bash
uv run python --version
swift --version
```

Runtime models can be large. Do not download them merely to run the unit suite. A real-model test is appropriate only when the change directly affects model loading, preprocessing, embeddings, or model output parsing.

## Making a change

1. Update your selected base branch.
2. Create one short-lived topic branch.
3. Keep the change focused on one problem.
4. Add or update tests before requesting review.
5. Update relevant documentation and migration notes.
6. Run the local checks.
7. Open a draft pull request early when design feedback would help.
8. Mark the pull request ready only when its checklist is complete.

Example:

```bash
git fetch upstream
git switch feature/image-pipeline
git merge --ff-only upstream/feature/image-pipeline
git switch -c image/improve-ocr-confidence
```

Do not rewrite shared branch history, force-push an integration branch, or commit directly to `master`.

## Required checks

Run these commands from the repository root:

```bash
uv lock --check
uv sync --locked --all-groups
uv run ruff check .
uv run python -m compileall -q mm_stack tests ./*.py
uv run python -m unittest discover -s tests -v
bash -n SmartStackUI/local_run.sh SmartStackUI/install_app.sh run_mobile_tailscale.sh stop_mobile_tailscale.sh run_guarded_ingest.sh
swift build --package-path SmartStackUI
```

If the local Swift compiler and SDK are incompatible, include the exact versions and error in the pull request. The hosted macOS build remains required.

### Image-pipeline changes

Tests should cover the relevant parts of:

- file validation and preprocessing;
- orientation and size normalization;
- OCR structure and confidence handling;
- caption, summary, and tag parsing;
- CLIP and text embedding boundaries;
- duplicate detection and safe reprocessing;
- image-only verification and ranking;
- phone camera/gallery ingestion;
- model cleanup and memory-pressure behavior.

Use fakes for model calls in unit tests. Never commit downloaded weights, user photos, generated indexes, or local databases.

### Document-pipeline changes

Tests should cover the relevant parts of:

- file-type detection by content, not only extension;
- page, slide, sheet, or section extraction;
- deterministic chunk boundaries and overlap;
- stable source and chunk identifiers;
- text-only embedding and verification;
- atomic replacement of every chunk and vector for an edited document;
- rollback when extraction, embedding, SQLite, or LanceDB persistence fails;
- duplicate handling and removal;
- mixed image/document ranking;
- one logical library card per document.

Document results must never be sent to an image-only verifier. A failed reindex must expose either the complete previous version or the complete replacement, never a mixture.

### Stored-data and schema changes

Any change to SQLite, LanceDB, identifiers, embedding dimensions, content hashes, or schema versions must document:

- the old and new representation;
- how an existing installation upgrades;
- whether rollback is possible;
- behavior during partial failure;
- how stale entries are detected or rebuilt;
- tests using an existing pre-change fixture.

Do not make destructive migrations automatic without an explicit backup or recovery path.

### User-interface changes

Include screenshots for visible changes. Verify keyboard navigation, empty states, errors, long filenames, missing source files, and both image-only and mixed-content views where applicable.

### Documentation-only changes

Documentation changes may omit runtime tests, but links, commands, paths, and examples must be checked. State why code checks are not applicable in the pull request.

## Dependencies

Explain why every new dependency is needed and why the standard library or an existing dependency is insufficient. Prefer actively maintained packages with compatible licenses.

After changing `pyproject.toml`, update and verify the lockfile:

```bash
uv lock
uv lock --check
```

Do not add a dependency solely for a small helper that can be implemented safely and clearly in the repository.

## Privacy and security

- Never commit images, documents, databases, vector indexes, access tokens, device addresses, private Tailscale names, personal absolute paths, or model-cache contents.
- Use synthetic fixtures with no personal information.
- Bind development services to localhost unless the feature explicitly requires another boundary.
- Validate uploaded file size, decoded type, pixel count, and destination path.
- Treat document parsers and filenames as untrusted input.
- Redact paths, tokens, and user content from logs and screenshots.
- Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## Commit and pull-request standards

Write concise, imperative commit subjects, for example:

```text
Make document replacement atomic
Preserve image ranking during mixed search
Validate generated application bundle
```

A pull request must explain:

- the problem and user impact;
- the chosen workstream;
- the implementation and important tradeoffs;
- tests performed and their results;
- storage, compatibility, privacy, and memory effects;
- screenshots for visible changes;
- follow-up work that is deliberately out of scope.

Keep unrelated formatting, generated files, and personal configuration out of the diff. Resolve every review conversation or explain why the requested change should not be made.

## Review and merge policy

Protected branches require passing checks, an approving review, an up-to-date branch, and resolved review conversations. Maintainers may request smaller pull requests when a change combines unrelated responsibilities.

Use squash merging for a focused feature or fix unless preserving individual commits is important for a staged migration. The final title should describe the product outcome and be suitable for release notes.

## Releases

Version tags use semantic versioning where practical:

- patch: compatible fixes and internal improvements;
- minor: backward-compatible capabilities;
- major: incompatible storage, API, or user-workflow changes.

The release workflow packages a development macOS application bundle. Public distribution signing and notarization are separate release responsibilities.

## Getting help

Use the issue templates for reproducible bugs and scoped feature proposals. For setup questions, see [SUPPORT.md](SUPPORT.md). Do not disclose a suspected vulnerability in a public issue.
