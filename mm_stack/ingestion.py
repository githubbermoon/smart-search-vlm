from __future__ import annotations

import fnmatch
import os
import re
import shutil
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clip_embedder import OpenCLIPEmbedder
from .config import StackConfig
from .db import (
    check_stale_files,
    connect_sqlite,
    ensure_schema,
    get_image_by_hash,
    list_exclusions,
    list_watched_folders,
    mark_file_removed,
    update_image_file_location,
    upsert_image_metadata,
    upsert_image_metadata,
    upsert_vector_metadata,
    upsert_video,
    upsert_video_segment,
)
from .entity_memory import replace_image_entity_memory
from .lancedb_store import LanceStore
from .models import OCRBlock, PreparedImage, VLMOutput
from .ocr import extract_ocr_structured
from .preprocess import preprocess_image
from .text_embedder import TextEmbedder
from .utils import sha256_text, utc_now_iso
from .vlm_analyzer import VLMAnalyzer
from .perception import AudioProcessor, VideoProcessor
from .config import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS


@dataclass
class Candidate:
    image_id: str
    prepared: PreparedImage
    ocr_blocks: list[OCRBlock]
    ocr_conf_avg: float
    existing_file_path: str | None = None
    vlm: VLMOutput | None = None
    clip_vec: list[float] | None = None
    text_vec: list[float] | None = None
    text_payload_hash: str = ""
    is_visual: bool = True


def _unique_dest(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        alt = directory / f"{stem}_{i}{suffix}"
        if not alt.exists():
            return alt
        i += 1


def _copy_to_media(src: Path, media_dir: Path) -> Path:
    media_dir.mkdir(parents=True, exist_ok=True)
    dst = _unique_dest(media_dir, src.name)
    shutil.copy2(str(src), str(dst))
    return dst


def _move_to_processed(src: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dst = _unique_dest(processed_dir, src.name)
    shutil.move(str(src), str(dst))
    return dst


def _ocr_text_from_blocks(blocks: list[OCRBlock]) -> str:
    return "\n".join(block.text for block in blocks if block.text.strip())


def _ocr_to_json(blocks: list[OCRBlock]) -> list[dict[str, Any]]:
    return [
        {
            "type": b.block_type,
            "text": b.text,
            "bbox": b.bbox,
            "confidence": b.confidence,
        }
        for b in blocks
    ]


def _build_text_payload(c: Candidate) -> str:
    ocr_text = _ocr_text_from_blocks(c.ocr_blocks)
    caption = c.vlm.caption if c.vlm else ""
    summary = c.vlm.summary if c.vlm else ""
    return "\n".join(x for x in (caption, summary, ocr_text) if x.strip())


def _extract_text_mentions(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for match in re.findall(r"\b[A-Z][a-z]{2,}\b", text or ""):
        token = match.strip()
        if not token:
            continue
        out.append(
            {
                "mention": token,
                "mention_type": "name",
                "confidence": 0.5,
                "source_field": "summary",
            }
        )
    # Deduplicate by mention
    uniq: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in out:
        key = str(row["mention"]).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq[:12]


class MultimodalIngestor:
    def __init__(self, cfg: StackConfig | None = None, *, image_batch_size: int | None = None):
        self.cfg = cfg or StackConfig()
        self.cfg.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        configured = int(self.cfg.ingest_image_batch_size)
        override = int(image_batch_size) if image_batch_size is not None else configured
        self.image_batch_size = max(1, override)

    def ingest_image(self, image_path: str | Path, *, safe_reprocess: bool = False) -> dict[str, Any]:
        return self.ingest_batch([Path(image_path)], safe_reprocess=safe_reprocess)

    def ingest_inbox(self, *, limit: int = 0, safe_reprocess: bool = False) -> dict[str, Any]:
        source = self.cfg.processed_dir if safe_reprocess else self.cfg.inbox_dir
        files = [
            p
            for p in sorted(source.iterdir())
            if p.is_file() and p.suffix.lower() in self.cfg.supported_exts
        ]
        if limit > 0:
            files = files[:limit]
        return self.ingest_batch(files, safe_reprocess=safe_reprocess)

    def ingest_batch(self, paths: list[Path], *, safe_reprocess: bool = False) -> dict[str, Any]:
        """
        Main ingestion entry point. Dispatches based on file type.
        """
        video_paths = []
        audio_paths = []
        image_paths = []

        for p in paths:
            suffix = p.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                video_paths.append(p)
            elif suffix in AUDIO_EXTENSIONS:
                audio_paths.append(p)
            elif suffix in self.cfg.supported_exts: # Assume rest are images if extension supported
                image_paths.append(p)
        
        results = {}
        if video_paths:
            results.update(self._ingest_videos(video_paths, safe_reprocess))
        if audio_paths:
            results.update(self._ingest_audios(audio_paths, safe_reprocess))
        if image_paths:
            results.update(self._ingest_images(image_paths, safe_reprocess))
            
        return results

    def _ingest_images(self, image_paths: list[Path], safe_reprocess: bool) -> dict[str, Any]:
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)

        candidates: list[Candidate] = []
        skipped_duplicates = 0
        relinked_paths = 0
        failures: list[str] = []
        processed = {
            "ingested": 0,
            "skipped_duplicates": 0,
            "relinked_paths": 0,
            "failed": [],
        }

        def _flush_candidates() -> None:
            nonlocal candidates
            if not candidates:
                return
            result = self._process_candidates(candidates, safe_reprocess)
            processed["ingested"] += int(result.get("ingested", 0))
            processed["skipped_duplicates"] += int(result.get("skipped_duplicates", 0))
            processed["relinked_paths"] += int(result.get("relinked_paths", 0))
            processed["failed"].extend(list(result.get("failed", [])))
            candidates = []

        # Stage 1: preprocess + OCR + dedupe check (no CLIP/VLM/text model loaded).
        for image_path in image_paths:
            try:
                prepared = preprocess_image(Path(image_path), self.cfg)
                existing = get_image_by_hash(conn, prepared.sha256_hash)
                if existing and not safe_reprocess:
                    current_path = str(prepared.source_path)
                    existing_path = str(existing["file_path"] or "")
                    if existing_path != current_path:
                        try:
                            st = os.stat(current_path)
                            file_inode = st.st_ino
                            file_size = st.st_size
                            file_mtime = st.st_mtime
                        except OSError:
                            file_inode, file_size, file_mtime = 0, 0, 0.0
                        update_image_file_location(
                            conn,
                            image_id=str(existing["id"]),
                            file_path=current_path,
                            file_inode=file_inode,
                            file_size=file_size,
                            file_mtime=file_mtime,
                        )
                        relinked_paths += 1
                    skipped_duplicates += 1
                    continue
                image_id = str(existing["id"]) if existing else str(uuid.uuid4())
                existing_file_path = str(existing["file_path"]) if existing else None

                ocr_blocks, ocr_conf = extract_ocr_structured(prepared.normalized_path, prepared.width, prepared.height)
                candidates.append(
                    Candidate(
                        image_id=image_id,
                        prepared=prepared,
                        ocr_blocks=ocr_blocks,
                        ocr_conf_avg=ocr_conf,
                        existing_file_path=existing_file_path,
                    )
                )
                if len(candidates) >= self.image_batch_size:
                    _flush_candidates()
            except Exception as exc:
                failures.append(f"{image_path}: {exc}")
                traceback.print_exc()

        conn.close()

        _flush_candidates()

        return {
            "ingested": int(processed.get("ingested", 0)),
            "skipped_duplicates": int(processed.get("skipped_duplicates", 0)) + skipped_duplicates,
            "relinked_paths": int(processed.get("relinked_paths", 0)) + relinked_paths,
            "failed": list(processed.get("failed", [])) + failures,
        }

    def _ingest_videos(self, video_paths: list[Path], safe_reprocess: bool) -> dict[str, Any]:
        """
        Ingests videos: Extraction -> Frame Candidates -> Persistence -> Linkage.
        """
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        
        video_proc = VideoProcessor(self.cfg)
        audio_proc = AudioProcessor()

        candidates: list[Candidate] = []
        segment_links: list[tuple[str, str, float, str]] = [] # (img_id, vid_id, ts, txt)
        failures = []

        cache_dir = self.cfg.preprocessed_dir / "frames"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for vid_path in video_paths:
            try:
                # 1. Upsert Video Metadata
                st = os.stat(vid_path)
                video_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(vid_path))) 
                
                if not safe_reprocess:
                    row = conn.execute("SELECT id FROM videos WHERE id=?", (video_id,)).fetchone()
                    if row:
                        continue 

                # Extract Audio & Transcribe
                transcript_segments = []
                audio_file = video_proc.extract_audio(vid_path)
                if audio_file:
                    res = audio_proc.transcribe(audio_file)
                    transcript_segments = res.get("segments", [])
                    audio_file.unlink(missing_ok=True)

                # Extract Frames (Visual)
                frames = list(video_proc.extract_frames(vid_path, fps=0.5)) 

                duration = frames[-1][0] if frames else 0.0 # Approx
                if not frames and transcript_segments:
                     duration = transcript_segments[-1]["end"]

                upsert_video(conn, {
                    "id": video_id,
                    "file_path": str(vid_path),
                    "duration": duration,
                    "created_at": utc_now_iso(),
                    "file_hash": "", 
                    "metadata": "{}"
                })

                # A. Visual Candidates (Frames)
                for timestamp, pil_img in frames:
                    frame_name = f"{video_id}_{int(timestamp*1000)}.jpg"
                    frame_path = cache_dir / frame_name
                    if not frame_path.exists():
                        pil_img.save(frame_path)
                    
                    # Find matching transcript text
                    matched_text = ""
                    for seg in transcript_segments:
                        if seg["start"] <= timestamp <= seg["end"]:
                            matched_text = seg["text"]
                            break
                    
                    w, h = pil_img.size
                    prep = PreparedImage(
                        source_path=frame_path, 
                        normalized_path=frame_path, 
                        sha256_hash=uuid.uuid4().hex, 
                        width=w, height=h
                    )
                    
                    c_id = str(uuid.uuid4())
                    ocr_blocks = []
                    if matched_text:
                        ocr_blocks.append(OCRBlock(text=matched_text, confidence=1.0, bbox=(0,0,0,0)))
                    
                    cand = Candidate(
                        image_id=c_id,
                        prepared=prep,
                        ocr_blocks=ocr_blocks,
                        ocr_conf_avg=1.0 if matched_text else 0.0,
                        is_visual=True
                    )
                    candidates.append(cand)
                    segment_links.append((c_id, video_id, timestamp, matched_text))

                # B. Text Candidates (Transcript Chunks - Audio Only)
                # If no frames (audio file) OR we want searchable text segments independently
                if not frames: 
                     # Create "Audio Candidates"
                     for seg in transcript_segments:
                         s_start, s_end, s_text = seg["start"], seg["end"], seg["text"]
                         if not s_text.strip(): continue
                         
                         c_id = str(uuid.uuid4())
                         # Dummy prepared image (required by Candidate struct)
                         # Use video path as source?
                         prep = PreparedImage(
                            source_path=vid_path, normalized_path=vid_path,
                            sha256_hash=uuid.uuid4().hex, width=0, height=0
                         )
                         
                         cand = Candidate(
                            image_id=c_id,
                            prepared=prep,
                            ocr_blocks=[OCRBlock(text=s_text, confidence=1.0, bbox=(0,0,0,0))], # Stores content
                            ocr_conf_avg=1.0,
                            is_visual=False 
                         )
                         # Set VLM props manually so they get indexed
                         cand.vlm = VLMOutput(caption=s_text, summary="", tags=["audio_transcript"], category="AudioSegment")
                         
                         candidates.append(cand)
                         segment_links.append((c_id, video_id, s_start, s_text))

            except Exception as e:
                failures.append(f"{vid_path}: {e}")
                traceback.print_exc()

        conn.close()

        if not candidates:
             return {"ingested": 0, "failed": failures}

        # Persist Generic Candidates
        result = self._process_candidates(candidates, safe_reprocess=True) 

        # Link Segments
        conn = connect_sqlite(self.cfg)
        count = 0
        for (img_id, vid_id, ts, txt) in segment_links:
            seg_id = f"seg_{img_id}"
            upsert_video_segment(conn, {
                "id": seg_id,
                "video_id": vid_id,
                "start_time": ts,
                "end_time": ts + 5.0, # Chunk duration?
                "transcript": txt,
                "embedding_id": img_id 
            })
            count += 1
        conn.commit()
        conn.close()
        
        result["video_segments"] = count
        result["failed"].extend(failures)
        return result

    def _ingest_audios(self, audio_paths: list[Path], safe_reprocess: bool) -> dict[str, Any]:
        # Reuse video ingestion logic as it handles audio extraction/transcription!
        # VideoProcessor.extract_audio works on audio files too (usually) via moviepy or just returns path
        # But we need to ensure it processes them as "Audio Only".
        # actually _ingest_videos handles "if not frames" case.
        return self._ingest_videos(audio_paths, safe_reprocess)

    def _process_candidates(self, candidates: list[Candidate], safe_reprocess: bool) -> dict[str, Any]:
        """
        Shared pipeline: CLIP -> VLM -> TextEmbed -> DB Persist.
        """
        # Stage 2: CLIP embeddings (Visual Only).
        visual_candidates = [c for c in candidates if c.is_visual]
        if visual_candidates:
            with OpenCLIPEmbedder(self.cfg.clip_model_name) as clip:
                clip_vectors = clip.encode_images([c.prepared.normalized_path for c in visual_candidates])
                for candidate, vec in zip(visual_candidates, clip_vectors, strict=True):
                    candidate.clip_vec = vec
        
        # Audio/Text Candidates need mock CLIP vector? Or allow NULL?
        # Database schema has NOT NULL for clip_vectors? No, separate table.
        # But images table has clip_content_hash.
        # Check `store.upsert_clip_vector`.
        
        # For non-visual candidates, we SKIP CLIP embedding.
        # But `upsert_clip_vector` might be called later?
        # We need to handle `candidate.clip_vec is None`.

        # Stage 3: VLM analysis (Visual Only).
        if visual_candidates:
            with VLMAnalyzer(self.cfg.vlm_model_name) as vlm:
                for candidate in visual_candidates:
                    candidate.vlm = vlm.analyze(candidate.prepared.normalized_path, candidate.ocr_blocks)


        # Stage 4: text embeddings from caption + summary + OCR text.
        with TextEmbedder(self.cfg.text_model_name) as text_embedder:
            payloads = []
            for candidate in candidates:
                payload = _build_text_payload(candidate)
                candidate.text_payload_hash = sha256_text(payload)
                payloads.append(payload)
            vectors = text_embedder.encode(payloads, is_query=False)
            for candidate, vec in zip(candidates, vectors, strict=True):
                candidate.text_vec = vec

        # Stage 5: persist metadata + vectors.
        ingested_count = 0
        conn = connect_sqlite(self.cfg)
        store = LanceStore(self.cfg)
        now = utc_now_iso()

        for candidate in candidates:
            try:
                source_path = candidate.prepared.source_path
                # Index-in-place: use original path, or existing path for reprocess
                if safe_reprocess and candidate.existing_file_path:
                    final_path = Path(candidate.existing_file_path)
                else:
                    final_path = source_path

                # Capture filesystem stat for stale detection
                try:
                    st = os.stat(str(final_path))
                    file_inode = st.st_ino
                    file_size = st.st_size
                    file_mtime = st.st_mtime
                except OSError:
                    file_inode, file_size, file_mtime = 0, 0, 0.0

                ocr_json = _ocr_to_json(candidate.ocr_blocks)
                tags = candidate.vlm.tags if candidate.vlm else []
                caption = candidate.vlm.caption if candidate.vlm else ""
                summary = candidate.vlm.summary if candidate.vlm else ""
                category = candidate.vlm.category if candidate.vlm else "Other"
                entities = candidate.vlm.entities if candidate.vlm and candidate.vlm.entities else []
                relations = candidate.vlm.relations if candidate.vlm and candidate.vlm.relations else []
                mentions = candidate.vlm.mentions if candidate.vlm and candidate.vlm.mentions else []
                if not mentions:
                    mentions = _extract_text_mentions("\n".join([caption, summary, _ocr_text_from_blocks(candidate.ocr_blocks)]))

                upsert_image_metadata(
                    conn,
                    {
                        "id": candidate.image_id,
                        "file_path": str(final_path),
                        "sha256_hash": candidate.prepared.sha256_hash,
                        "width": candidate.prepared.width,
                        "height": candidate.prepared.height,
                        "caption": caption,
                        "summary": summary,
                        "category": category,
                        "tags": tags,
                        "ocr_structured": ocr_json,
                        "ocr_confidence_avg": candidate.ocr_conf_avg,
                        "schema_version": self.cfg.schema_version,
                        "embedding_model_clip": self.cfg.clip_model_name,
                        "embedding_model_text": self.cfg.text_model_name,
                        "embedding_dimension_clip": self.cfg.clip_dimension,
                        "embedding_dimension_text": self.cfg.text_dimension,
                        "embedding_schema_version_clip": self.cfg.clip_schema_version,
                        "embedding_schema_version_text": self.cfg.text_schema_version,
                        "text_payload_hash": candidate.text_payload_hash,
                        "clip_content_hash": candidate.prepared.sha256_hash,
                        "is_stale": 0,
                        "file_inode": file_inode,
                        "file_size": file_size,
                        "file_mtime": file_mtime,
                        "created_at": now,
                    },
                )

                replace_image_entity_memory(
                    conn,
                    image_id=candidate.image_id,
                    entities=entities,
                    relations=relations,
                    mentions=mentions,
                    source_model=self.cfg.vlm_model_name,
                    schema_version=self.cfg.schema_version,
                )

                if candidate.clip_vec:
                    store.upsert_clip_vector(
                        image_id=candidate.image_id,
                        vector=candidate.clip_vec,
                        model_name=self.cfg.clip_model_name,
                        schema_version=self.cfg.clip_schema_version,
                        created_at=now,
                    )
                    upsert_vector_metadata(
                        conn,
                        table_name="clip_vectors",
                        image_id=candidate.image_id,
                        vector_id=f"clip:{candidate.image_id}",
                        model_name=self.cfg.clip_model_name,
                        dimension=self.cfg.clip_dimension,
                        schema_version=self.cfg.clip_schema_version,
                    )

                if candidate.text_vec:
                    store.upsert_text_vector(
                        image_id=candidate.image_id,
                        vector=candidate.text_vec,
                        model_name=self.cfg.text_model_name,
                        schema_version=self.cfg.text_schema_version,
                        created_at=now,
                    )
                    upsert_vector_metadata(
                        conn,
                        table_name="text_vectors",
                        image_id=candidate.image_id,
                        vector_id=f"text:{candidate.image_id}",
                        model_name=self.cfg.text_model_name,
                        dimension=self.cfg.text_dimension,
                        schema_version=self.cfg.text_schema_version,
                    )

                conn.commit()
                ingested_count += 1
            except Exception as exc:
                conn.rollback()
                # failures.append(f"{candidate.prepared.source_path}: {exc}") # Failures list not passed here, swallow or log?
                # Better to return failures count
                print(f"Failed to persist candidate {candidate.prepared.source_path}: {exc}")
                traceback.print_exc()

        conn.close()
        return {
            "ingested": ingested_count,
            "failed": [], # TODO: Capture failures better
            "skipped_duplicates": 0
        }

    def ingest_path(self, target: str | Path, *, safe_reprocess: bool = False) -> dict[str, Any]:
        """
        Index-in-place: accepts any file or directory. Walks dirs recursively.
        No copying — stores original paths.
        """
        target = Path(target)
        if target.is_file():
            # Check against ALL supported exts
            supported = self.cfg.supported_exts # tuple of all
            if target.suffix.lower() in supported:
                return self.ingest_batch([target], safe_reprocess=safe_reprocess)
            return {"ingested": 0, "skipped_duplicates": 0, "failed": [f"{target}: unsupported extension"]}
        elif target.is_dir():
            files = [
                p for p in sorted(target.rglob("*"))
                if p.is_file() and p.suffix.lower() in self.cfg.supported_exts
            ]
            if not files:
                return {"ingested": 0, "skipped_duplicates": 0, "failed": [f"{target}: no images/media found"]}
            return self.ingest_batch(files, safe_reprocess=safe_reprocess)
        else:
            return {"ingested": 0, "skipped_duplicates": 0, "failed": [f"{target}: not found"]}

    def rescan_stale(self) -> dict[str, Any]:
        """
        Checks all indexed files for inode/size/mtime changes.
        Re-ingests changed files, marks missing as stale.
        """
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        changed = check_stale_files(conn)
        conn.close()

        re_ingest = []
        removed = []
        for entry in changed:
            if entry["reason"] == "missing":
                c = connect_sqlite(self.cfg)
                ensure_schema(c)
                mark_file_removed(c, entry["image_id"])
                c.close()
                removed.append(entry["file_path"])
            else:
                re_ingest.append(Path(entry["file_path"]))

        result: dict[str, Any] = {"changed": len(changed), "removed": removed}
        if re_ingest:
            ingest_result = self.ingest_batch(re_ingest, safe_reprocess=True)
            result["re_ingested"] = ingest_result["ingested"]
            result["failed"] = ingest_result["failed"]
        else:
            result["re_ingested"] = 0
            result["failed"] = []
        return result

    def rescan_watched(self) -> dict[str, Any]:
        """
        Scans all enabled watched folders, skipping excluded patterns.
        Ingests new/changed files in-place.
        """
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        folders = list_watched_folders(conn)
        exclusions = list_exclusions(conn)
        conn.close()

        exclude_patterns = [e["pattern"] for e in exclusions]
        all_files: list[Path] = []

        for folder in folders:
            if not folder["enabled"]:
                continue
            folder_path = Path(folder["path"])
            if not folder_path.is_dir():
                continue
            for p in sorted(folder_path.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in self.cfg.supported_exts:
                    continue
                # Check exclusions
                excluded = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(str(p), pattern) or fnmatch.fnmatch(p.name, pattern):
                        excluded = True
                        break
                if not excluded:
                    all_files.append(p)

        if not all_files:
            return {"scanned_folders": len([f for f in folders if f["enabled"]]), "new_files": 0, "ingested": 0}

        result = self.ingest_batch(all_files, safe_reprocess=False)
        result["scanned_folders"] = len([f for f in folders if f["enabled"]])
        result["new_files"] = len(all_files)
        return result
