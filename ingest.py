#!/Users/pranjal/garage/smart_stack/.venv/bin/python3
"""Level 2 Local Intelligence ingestion pipeline for Smart Stack."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlite_utils import Database

HOME = Path.home()
STACK_ROOT = HOME / "garage" / "smart_stack"
INBOX_DIR = STACK_ROOT / "inbox"
PROCESSED_DIR = STACK_ROOT / "processed"
FAILED_DIR = STACK_ROOT / "failed"
OBSIDIAN_MEDIA_DIR = HOME / "Pranjal-Obs" / "clawd" / "Media"

SQLITE_PATH = HOME / "Pranjal-Obs" / "clawd" / "smart_stack.db"
LANCEDB_PATH = HOME / "Pranjal-Obs" / "clawd" / "vectors.lance"

DEFAULT_VLM_MODEL_ID = "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit"
MLX_MODEL_ID = os.getenv("SMART_STACK_VLM_MODEL", DEFAULT_VLM_MODEL_ID)
DEFAULT_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_ID = os.getenv("SMART_STACK_EMBED_MODEL", DEFAULT_EMBED_MODEL_ID)
LANCEDB_TABLE = "image_embeddings"
TAG_COUNT = 5

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp", ".tiff"}
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
EN_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
HI_MONTHS = {
    "जनवरी",
    "फ़रवरी",
    "फरवरी",
    "मार्च",
    "अप्रैल",
    "मई",
    "जून",
    "जुलाई",
    "अगस्त",
    "सितंबर",
    "सितम्बर",
    "अक्टूबर",
    "नवंबर",
    "नवम्बर",
    "दिसंबर",
    "दिसम्बर",
}
DATE_PATTERN = re.compile(r"\b([0-9०-९]{1,2})\s+([^\s0-9०-९]{2,18})\s+([0-9०-९]{4})\b")
STOPWORDS = {
    "this",
    "that",
    "there",
    "their",
    "from",
    "with",
    "have",
    "has",
    "into",
    "about",
    "under",
    "over",
    "and",
    "the",
    "for",
    "are",
    "was",
    "were",
    "you",
    "your",
    "not",
    "but",
    "can",
    "will",
    "just",
    "image",
    "photo",
    "text",
    "total",
    "subtotal",
    "amount",
    "thank",
    "thanks",
    "date",
    "time",
}


def ensure_dirs() -> None:
    for path in (INBOX_DIR, PROCESSED_DIR, FAILED_DIR, OBSIDIAN_MEDIA_DIR, SQLITE_PATH.parent, LANCEDB_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


def build_sqlite() -> Database:
    db = Database(SQLITE_PATH)
    if "processed_images" not in db.table_names():
        db["processed_images"].create(
            {
                "id": int,
                "filename": str,
                "file_hash": str,
                "tags": str,
                "ocr_text": str,
                "caption": str,
                "processed_at": str,
                "obsidian_path": str,
            },
            pk="id",
        )
    db["processed_images"].create_index(["file_hash"], unique=True, if_not_exists=True)
    return db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest images into Smart Stack.")
    parser.add_argument(
        "--vlm-model",
        default=MLX_MODEL_ID,
        help=f"VLM model id to load (default: {MLX_MODEL_ID})",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL_ID,
        help=f"Embedding model id to load (default: {EMBED_MODEL_ID})",
    )
    parser.add_argument(
        "--safe-reprocess",
        action="store_true",
        help=(
            "Reprocess files from processed/ and upsert DB/vector rows by file hash "
            "without moving files or creating duplicate Obsidian media."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of files to process in this run (0 = no limit).",
    )
    parser.add_argument(
        "--no-print-fields",
        action="store_true",
        help="Do not print extracted caption/tags/OCR preview per image.",
    )
    parser.add_argument(
        "--memory-threshold-mb",
        type=int,
        default=0,
        help=(
            "Optional memory guard. If Active+Wired RAM is above this value, gate processing "
            "(0 disables guard)."
        ),
    )
    parser.add_argument(
        "--memory-gate-mode",
        choices=["wait", "skip", "fail"],
        default="wait",
        help=(
            "Behavior when memory is above threshold: wait (default), skip current item, or fail run."
        ),
    )
    parser.add_argument(
        "--memory-timeout-sec",
        type=int,
        default=180,
        help="Max seconds to wait in memory gate when --memory-gate-mode wait (default: 180).",
    )
    parser.add_argument(
        "--memory-poll-sec",
        type=float,
        default=5.0,
        help="Polling interval for memory gate checks (default: 5.0).",
    )
    parser.add_argument(
        "--memory-relief-cmd",
        default="",
        help=(
            "Optional command to invoke once when memory is above threshold "
            "(e.g. 'bash /Users/pranjal/clawdGIT/scripts/purge_and_run.sh --relief-only')."
        ),
    )
    return parser.parse_args()


def _parse_vm_stat_pages(raw: str) -> tuple[int, dict[str, int]]:
    page_size_match = re.search(r"page size of (\d+) bytes", raw)
    page_size = int(page_size_match.group(1)) if page_size_match else 4096
    pages: dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        digits = re.sub(r"[^0-9]", "", val)
        if not digits:
            continue
        pages[key.strip().lower()] = int(digits)
    return page_size, pages


def active_wired_mb() -> int:
    try:
        raw = subprocess.check_output(["vm_stat"], text=True)
        page_size, pages = _parse_vm_stat_pages(raw)
        active = pages.get("pages active", 0)
        wired = pages.get("pages wired down", 0)
        used_bytes = (active + wired) * page_size
        return int(used_bytes / 1024 / 1024)
    except Exception:
        return 0


def maybe_cleanup_mps_cache() -> None:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


def enforce_memory_gate(
    threshold_mb: int,
    mode: str,
    timeout_sec: int,
    poll_sec: float,
    relief_cmd: str,
    stage: str,
) -> bool:
    if threshold_mb <= 0:
        return True

    start = time.monotonic()
    relief_invoked = False
    while True:
        used_mb = active_wired_mb()
        if used_mb <= threshold_mb:
            return True

        if relief_cmd and not relief_invoked:
            print(f"[MEM] {stage}: {used_mb}MB > {threshold_mb}MB. Running relief command...")
            subprocess.run(relief_cmd, shell=True, check=False)
            relief_invoked = True
            continue

        if mode == "skip":
            print(f"[MEM] {stage}: {used_mb}MB > {threshold_mb}MB. Skipping item.")
            return False
        if mode == "fail":
            print(f"[MEM] {stage}: {used_mb}MB > {threshold_mb}MB. Failing run.")
            return False

        # wait mode
        elapsed = time.monotonic() - start
        if elapsed >= max(0, timeout_sec):
            print(
                f"[MEM] {stage}: still {used_mb}MB after {int(elapsed)}s "
                f"(threshold {threshold_mb}MB)."
            )
            return False
        print(
            f"[MEM] {stage}: {used_mb}MB > {threshold_mb}MB. "
            f"Waiting {poll_sec:.1f}s..."
        )
        time.sleep(max(0.5, poll_sec))


def sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_duplicate(db: Database, file_hash: str) -> bool:
    rows = list(db["processed_images"].rows_where("file_hash = ?", [file_hash], limit=1))
    return bool(rows)


def get_existing_row(db: Database, file_hash: str) -> dict[str, Any] | None:
    rows = list(db["processed_images"].rows_where("file_hash = ?", [file_hash], limit=1))
    if not rows:
        return None
    return dict(rows[0])


def unique_dest(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def move_to(file_path: Path, directory: Path, prefix: str | None = None) -> Path:
    out_name = file_path.name if prefix is None else f"{prefix}_{file_path.name}"
    dst = unique_dest(directory, out_name)
    shutil.move(str(file_path), str(dst))
    return dst


def copy_to_obsidian(file_path: Path) -> Path:
    dst = unique_dest(OBSIDIAN_MEDIA_DIR, file_path.name)
    shutil.copy2(str(file_path), str(dst))
    return dst


def resolve_obsidian_path_for_reprocess(existing: dict[str, Any] | None, file_path: Path) -> Path:
    if existing:
        candidate = Path(str(existing.get("obsidian_path") or "")).expanduser()
        if str(candidate) and candidate.exists():
            return candidate
    return copy_to_obsidian(file_path)


def load_cg_image(image_path: Path) -> Any:
    from Cocoa import NSURL
    from Quartz import CGImageSourceCreateImageAtIndex, CGImageSourceCreateWithURL

    url = NSURL.fileURLWithPath_(str(image_path))
    src = CGImageSourceCreateWithURL(url, None)
    if src is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    cg_image = CGImageSourceCreateImageAtIndex(src, 0, None)
    if cg_image is None:
        raise RuntimeError(f"Cannot decode image: {image_path}")
    return cg_image


def run_ocr(image_path: Path) -> str:
    import Vision

    def score_text(text: str) -> float:
        if not text.strip():
            return -1e9
        chars = [c for c in text if not c.isspace()]
        total = max(1, len(chars))
        devanagari = sum(1 for c in chars if DEVANAGARI_RE.match(c))
        latin = sum(1 for c in chars if c.isascii() and c.isalpha())
        digits = sum(1 for c in chars if c.isdigit())
        allowed_punct = sum(1 for c in chars if c in ".,;:!?\"'()[]{}-_/+%&@")
        other = total - devanagari - latin - digits - allowed_punct
        words = len(re.findall(r"\S+", text))
        return (devanagari * 2.0) + (latin * 1.2) + (digits * 0.3) + (words * 1.0) - (other * 3.0)

    def recognize(languages: list[str] | None) -> str:
        request = Vision.VNRecognizeTextRequest.alloc().init()
        if hasattr(request, "setRecognitionLevel_"):
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        if hasattr(request, "setUsesLanguageCorrection_"):
            request.setUsesLanguageCorrection_(True)
        if languages and hasattr(request, "setRecognitionLanguages_"):
            try:
                request.setRecognitionLanguages_(languages)
            except Exception:
                pass

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(load_cg_image(image_path), None)
        result = handler.performRequests_error_([request], None)
        if isinstance(result, tuple):
            ok, err = result
        else:
            ok, err = bool(result), None
        if not ok:
            raise RuntimeError(f"Vision OCR failed: {err}")

        lines: list[str] = []
        for obs in request.results() or []:
            candidates = obs.topCandidates_(1)
            if candidates and len(candidates) > 0:
                text = str(candidates[0].string()).strip()
                if text:
                    lines.append(text)
        return "\n".join(lines)

    # Two-pass OCR: compare language-hinted and default decode, keep higher-quality text.
    hinted = recognize(["hi-IN", "en-US"])
    default = recognize(None)
    return hinted if score_text(hinted) >= score_text(default) else default


def normalize_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.strip("\"'`")
    return text[:220]


def dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def normalize_tags(tags: list[str]) -> list[str]:
    cleaned: list[str] = []
    for tag in tags:
        token = str(tag).lower().strip()
        token = token.replace("_", " ")
        token = re.sub(r"[^\w\s&+/-]", "", token)
        token = re.sub(r"\s+", " ", token).strip()
        if len(token) < 2 or token in STOPWORDS:
            continue
        if token.replace(" ", "").isdigit():
            continue
        cleaned.append(token)
    return dedupe_keep_order(cleaned)


def extract_keyword_tags(text: str, limit: int = TAG_COUNT) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,24}", (text or "").lower())
    out: list[str] = []
    for token in tokens:
        if token in STOPWORDS or token.isdigit():
            continue
        out.append(token)
    return dedupe_keep_order(out)[:limit]


def extract_month_tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    out: set[str] = set()
    for month in EN_MONTHS:
        if re.search(rf"\b{re.escape(month)}\b", lowered):
            out.add(month)
    for month in HI_MONTHS:
        if month in (text or ""):
            out.add(month)
    return out


def strip_unverified_months(caption: str, allowed_months: set[str]) -> str:
    caption_out = caption
    all_months = extract_month_tokens(caption)
    bad_months = all_months - allowed_months
    for month in sorted(bad_months, key=len, reverse=True):
        if month.isascii():
            caption_out = re.sub(rf"(?i)\b{re.escape(month)}\b", "", caption_out)
        else:
            caption_out = caption_out.replace(month, "")
    caption_out = re.sub(r"\s{2,}", " ", caption_out).strip(" ,.;:-")
    caption_out = normalize_caption(caption_out)
    return caption_out


def strip_unverified_month_like_dates(caption: str, ocr_text: str) -> str:
    # If OCR has no month token, avoid fabricated middle tokens in "DD <month-like> YYYY".
    if extract_month_tokens(ocr_text):
        return caption

    def _repl(match: re.Match[str]) -> str:
        day, token, year = match.group(1), match.group(2), match.group(3)
        # Keep known months; strip unknown month-like words.
        if token.lower() in EN_MONTHS or token in HI_MONTHS:
            return f"{day} {year}"
        return f"{day} {year}"

    cleaned = DATE_PATTERN.sub(_repl, caption)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;:-")
    return normalize_caption(cleaned)


def derive_ocr_signal_tags(ocr_text: str) -> list[str]:
    text = ocr_text or ""
    tags: list[str] = []
    has_devanagari = bool(DEVANAGARI_RE.search(text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_devanagari:
        tags.append("devanagari text")
    if has_devanagari and has_latin:
        tags.append("hindi english text")
    if re.search(r"\b\d{1,2}\s+\d{4}\b", text):
        tags.append("date text")
    if re.search(r"[\"“”'‘’]", text):
        tags.append("quoted text")
    return tags


def fallback_caption(file_path: Path, ocr_text: str) -> str:
    best_line = ""
    best_score = -10_000
    for raw in (ocr_text or "").splitlines():
        line = normalize_caption(raw)
        if len(line) < 6:
            continue
        alpha = sum(1 for ch in line if ch.isalpha())
        digits = sum(1 for ch in line if ch.isdigit())
        words = len(line.split())
        score = (alpha * 2) + words - (digits * 2)
        if score > best_score:
            best_score = score
            best_line = line
    if best_line:
        return best_line
    stem = re.sub(r"[_-]+", " ", file_path.stem).strip()
    if stem:
        return normalize_caption(stem)
    return "untitled image"


def build_ocr_snippet(ocr_text: str, max_lines: int = 14, max_chars: int = 1200) -> str:
    lines: list[str] = []
    used = 0
    for raw in (ocr_text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def ocr_preview(ocr_text: str, max_chars: int = 260) -> str:
    compact = re.sub(r"\s+", " ", (ocr_text or "")).strip()
    if not compact:
        return "(empty)"
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def print_extracted_fields(file_path: Path, caption: str, tags: list[str], ocr_text: str) -> None:
    print(f"[FIELDS] {file_path.name}")
    print(f"  caption: {caption or '(empty)'}")
    print(f"  tags: {', '.join(tags) if tags else '(empty)'}")
    print(f"  ocr: {ocr_preview(ocr_text)}")


def finalize_caption_tags(file_path: Path, ocr_text: str, caption: str, tags: list[str]) -> tuple[str, list[str]]:
    caption = normalize_caption(caption)
    tags = normalize_tags(tags)
    ocr_tags = extract_keyword_tags(ocr_text, limit=TAG_COUNT)
    ocr_signal_tags = normalize_tags(derive_ocr_signal_tags(ocr_text))

    if not caption:
        caption = fallback_caption(file_path, ocr_text)

    uncertain = re.search(r"\b(maybe|probably|likely|appears?|seems?|possibly)\b", caption.lower())
    if uncertain and ocr_text.strip() and len(caption.split()) < 5:
        caption = fallback_caption(file_path, ocr_text)

    # Guardrail: do not allow month names in caption unless OCR also contains the same month.
    if ocr_text.strip():
        ocr_months = extract_month_tokens(ocr_text)
        caption_months = extract_month_tokens(caption)
        if caption_months and not caption_months.issubset(ocr_months):
            stripped = strip_unverified_months(caption, ocr_months)
            if stripped:
                caption = stripped
            else:
                caption = fallback_caption(file_path, ocr_text)
        caption = strip_unverified_month_like_dates(caption, ocr_text)

    merged = list(ocr_signal_tags) + list(tags)
    if ocr_tags:
        if not any(tag in ocr_tags for tag in merged):
            merged.append(ocr_tags[0])
        merged.extend(tag for tag in ocr_tags if tag not in merged)

    if len(merged) < TAG_COUNT:
        caption_tags = extract_keyword_tags(caption, limit=TAG_COUNT)
        merged.extend(tag for tag in caption_tags if tag not in merged)

    if len(merged) < TAG_COUNT:
        defaults = ["document", "screenshot", "note", "image", "photo"]
        merged.extend(tag for tag in defaults if tag not in merged)

    return caption, merged[:TAG_COUNT]


def parse_vlm_output(text: str) -> tuple[str, list[str], str]:
    text = text.strip()

    caption = ""
    tags: list[str] = []

    # 1) Try strict JSON extraction first.
    candidates: list[str] = []
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE):
        candidates.append(m.group(1))
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in dedupe_keep_order(candidates):
        try:
            blob = json.loads(candidate)
            caption = str(blob.get("caption", "")).strip()
            parsed_tags = blob.get("tags", [])
            if isinstance(parsed_tags, list):
                tags = [str(t).strip().lower() for t in parsed_tags if str(t).strip()]
            elif isinstance(parsed_tags, str):
                tags = [t.strip().lower() for t in re.split(r"[,;|]+", parsed_tags) if t.strip()]
            if caption or tags:
                break
        except Exception:
            continue

    # 1b) Recover from partial JSON-like fragments when full JSON parse fails.
    if not caption:
        partial_cap = re.search(r'"caption"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if partial_cap:
            caption = partial_cap.group(1).strip()
    if not tags:
        partial_tags = re.search(r'"tags"\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE | re.DOTALL)
        if partial_tags:
            raw_tags = re.findall(r'"([^"]+)"', partial_tags.group(1))
            tags = [t.strip().lower() for t in raw_tags if t.strip()]

    # 2) Heuristic extraction when JSON is missing or malformed.
    if not caption:
        cap_match = re.search(r"(?i)caption:\s*(.*)", text)
        if cap_match:
            caption = cap_match.group(1).split("\n")[0].strip()
    
    if not tags:
        tags_match = re.search(r"(?i)tags:\s*(.*)", text)
        if tags_match:
            tags_line = tags_match.group(1).split("\n")[0].strip()
            tags_line = re.sub(r'[\[\]"\'\.]', "", tags_line)
            tags = [t.strip().lower() for t in re.split(r"[,;|\s]+", tags_line) if t.strip() and len(t) > 1]

    # 3) Final fallback from generated text.
    if not caption:
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("{")]
        caption = lines[0][:300] if lines else "Untitled Image"
    
    if not tags:
        potential = [t.strip().lower() for t in re.split(r"[\s,.]+", caption) if len(t) > 3]
        tags = potential[:5] if potential else ["processed"]

    caption = normalize_caption(caption.replace("...", ""))
    tags = normalize_tags([t.replace("...", "").strip() for t in tags])

    return caption, tags, text

def generate_caption_tags(image_path: Path, ocr_text: str, model: Any, processor: Any) -> tuple[str, list[str], str]:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_image

    try:
        ocr_snippet = build_ocr_snippet(ocr_text) or "(none)"
        if ocr_text.strip():
            user_prompt = (
                "You are labeling one image for a local search index.\n"
                'Return only valid JSON: {"caption":"", "tags":["","","","",""]}\n'
                "Rules:\n"
                "- Use only information visible in the image or OCR text below.\n"
                "- Do not guess brand, location, person identity, date, or amount unless clearly visible.\n"
                "- If a date appears, copy it exactly from OCR text; never infer or translate month/day.\n"
                "- caption must be factual and concise.\n"
                "- tags must be lowercase and factual keywords.\n"
                f"OCR text:\n{ocr_snippet}\n"
            )
        else:
            user_prompt = (
                "You are labeling one photo for search.\n"
                'Return only valid JSON: {"caption":"", "tags":["","","","",""]}\n'
                "Rules:\n"
                "- Describe only clearly visible objects.\n"
                "- If uncertain, use generic nouns (food, dish, plate, bowl, table, phone, document, screen).\n"
                "- Do not mention animals, people, bed, sofa, or room context unless clearly visible.\n"
                "- Keep caption short and factual.\n"
            )

        prompt = apply_chat_template(processor, model.config, user_prompt, num_images=1)
        
        img = load_image(str(image_path))
        raw = generate(
            model,
            processor,
            image=[img],
            prompt=prompt,
            max_tokens=220,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
        )
        text = raw.text if hasattr(raw, "text") else str(raw)
        caption, tags, _ = parse_vlm_output(text)
        if not caption or not any(tags):
            fallback_user_prompt = (
                "Return strict JSON with non-empty fields:\n"
                '{"caption":"", "tags":["","","","",""]}\n'
                "Use only visible objects. Avoid guessing hidden context.\n"
            )
            fallback_prompt = apply_chat_template(processor, model.config, fallback_user_prompt, num_images=1)
            raw2 = generate(
                model,
                processor,
                image=[img],
                prompt=fallback_prompt,
                max_tokens=180,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.05,
            )
            text2 = raw2.text if hasattr(raw2, "text") else str(raw2)
            cap2, tags2, _ = parse_vlm_output(text2)
            if cap2 or tags2:
                caption, tags, text = cap2, tags2, text2
        caption, tags = finalize_caption_tags(image_path, ocr_text, caption, tags)
        return caption, tags, text

    except Exception as e:
        print(f"[DEBUG] VLM error: {e}")
        raise


def requires_trust_remote_code(model_id: str) -> bool:
    lowered = (model_id or "").lower()
    return "nomic-ai/nomic-embed-text" in lowered


def prepare_embed_input(text: str, model_id: str, is_query: bool) -> str:
    payload = text.strip()
    lowered = (model_id or "").lower()
    if "nomic-embed-text" in lowered:
        prefix = "search_query: " if is_query else "search_document: "
        return f"{prefix}{payload}"
    if "e5" in lowered:
        prefix = "query: " if is_query else "passage: "
        return f"{prefix}{payload}"
    return payload


def table_for_embed_model(model_id: str) -> str:
    if model_id == DEFAULT_EMBED_MODEL_ID:
        return LANCEDB_TABLE
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    slug = slug[:64] if slug else "custom"
    return f"{LANCEDB_TABLE}__{slug}"


def embed_text(text: str, embedder: Any, embed_model_id: str, is_query: bool = False) -> list[float]:
    payload = prepare_embed_input(text, embed_model_id, is_query=is_query)
    vector = embedder.encode(payload, normalize_embeddings=True)
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if isinstance(vector, list) and vector and isinstance(vector[0], list):
        vector = vector[0]
    return [float(x) for x in vector]


def write_vector_row(row: dict[str, Any], table_name: str, upsert_by_hash: bool = False) -> None:
    import lancedb

    db = lancedb.connect(str(LANCEDB_PATH))
    try:
        table = db.open_table(table_name)
    except Exception:
        db.create_table(table_name, data=[row])
        return

    if upsert_by_hash:
        file_hash = str(row.get("file_hash") or "").replace("'", "''")
        if file_hash:
            try:
                table.delete(f"file_hash = '{file_hash}'")
            except Exception:
                pass
    table.add([row])


def upsert_metadata_row(db: Database, row: dict[str, Any]) -> None:
    db.conn.execute(
        """
        INSERT INTO processed_images
        (filename, file_hash, tags, ocr_text, caption, processed_at, obsidian_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_hash) DO UPDATE SET
            filename=excluded.filename,
            tags=excluded.tags,
            ocr_text=excluded.ocr_text,
            caption=excluded.caption,
            processed_at=excluded.processed_at,
            obsidian_path=excluded.obsidian_path
        """,
        (
            row["filename"],
            row["file_hash"],
            row["tags"],
            row["ocr_text"],
            row["caption"],
            row["processed_at"],
            row["obsidian_path"],
        ),
    )
    db.conn.commit()


def process_one_image(
    db: Database,
    file_path: Path,
    vlm_model: Any,
    vlm_processor: Any,
    embedder: Any,
    embed_model_id: str,
    vector_table_name: str,
    safe_reprocess: bool = False,
    print_fields: bool = True,
) -> None:
    file_hash = sha256_file(file_path)
    existing = get_existing_row(db, file_hash)
    if not safe_reprocess and existing is not None:
        archived = move_to(file_path, PROCESSED_DIR, prefix="duplicate")
        print(f"[SKIP] duplicate hash={file_hash[:12]} archived={archived}")
        return

    ocr_text = run_ocr(file_path)
    caption, tags, vlm_raw = generate_caption_tags(file_path, ocr_text, vlm_model, vlm_processor)
    if print_fields:
        print_extracted_fields(file_path, caption, tags, ocr_text)
    embed_payload = "\n".join(
        s
        for s in (
            file_path.name,
            caption,
            ", ".join(tags),
            ocr_text,
            vlm_raw,
        )
        if s
    )
    vector = embed_text(embed_payload, embedder, embed_model_id, is_query=False)
    now = datetime.now(timezone.utc).isoformat()

    obsidian_copy = (
        resolve_obsidian_path_for_reprocess(existing, file_path)
        if safe_reprocess
        else copy_to_obsidian(file_path)
    )
    metadata_row = {
        "filename": file_path.name,
        "file_hash": file_hash,
        "tags": json.dumps(tags, ensure_ascii=True),
        "ocr_text": ocr_text,
        "caption": caption,
        "processed_at": now,
        "obsidian_path": str(obsidian_copy),
    }
    if safe_reprocess:
        upsert_metadata_row(db, metadata_row)
    else:
        db["processed_images"].insert(metadata_row, alter=True)

    write_vector_row(
        {
            "id": file_hash,
            "file_hash": file_hash,
            "filename": file_path.name,
            "embedding": vector,
            "text": embed_payload,
            "processed_at": now,
            "obsidian_path": str(obsidian_copy),
        },
        table_name=vector_table_name,
        upsert_by_hash=safe_reprocess,
    )

    if safe_reprocess:
        print(f"[REPROCESS-OK] {file_path.name} hash={file_hash[:12]}")
        return

    archived = move_to(file_path, PROCESSED_DIR)
    print(f"[OK] {file_path.name} -> {archived}")


def iter_images(directory: Path, limit: int = 0) -> list[Path]:
    files: list[Path] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
            if limit > 0 and len(files) >= limit:
                break
    return files


def main() -> None:
    args = parse_args()
    ensure_dirs()
    db = build_sqlite()
    source_dir = PROCESSED_DIR if args.safe_reprocess else INBOX_DIR
    files = iter_images(source_dir, limit=max(0, args.limit))

    if not files:
        mode = "safe reprocess" if args.safe_reprocess else "ingest"
        print(f"[INFO] No images for {mode} in {source_dir}")
        return

    from mlx_vlm import load
    from sentence_transformers import SentenceTransformer

    if args.memory_threshold_mb > 0:
        used = active_wired_mb()
        print(f"[INFO] Memory guard: used={used}MB threshold={args.memory_threshold_mb}MB mode={args.memory_gate_mode}")
        ok = enforce_memory_gate(
            threshold_mb=args.memory_threshold_mb,
            mode=args.memory_gate_mode,
            timeout_sec=args.memory_timeout_sec,
            poll_sec=args.memory_poll_sec,
            relief_cmd=args.memory_relief_cmd,
            stage="before model load",
        )
        if not ok:
            raise SystemExit("[ERROR] Memory gate blocked model load.")

    print(f"[INFO] Loading VLM model: {args.vlm_model}")
    model, processor = load(args.vlm_model)
    vector_table_name = table_for_embed_model(args.embed_model)
    print(f"[INFO] Loading embedding model: {args.embed_model}")
    print(f"[INFO] Using vector table: {vector_table_name}")
    embedder = SentenceTransformer(
        args.embed_model,
        trust_remote_code=requires_trust_remote_code(args.embed_model),
    )

    try:
        for file_path in files:
            try:
                process_one_image(
                    db,
                    file_path,
                    model,
                    processor,
                    embedder,
                    embed_model_id=args.embed_model,
                    vector_table_name=vector_table_name,
                    safe_reprocess=args.safe_reprocess,
                    print_fields=not args.no_print_fields,
                )
            except Exception as exc:
                print(f"[FAIL] {file_path.name}: {exc}")
                traceback.print_exc()
                if file_path.exists() and not args.safe_reprocess:
                    failed_dst = move_to(file_path, FAILED_DIR)
                    print(f"[FAIL] moved to {failed_dst}")
            finally:
                maybe_cleanup_mps_cache()
    finally:
        del model
        del processor
        del embedder
        maybe_cleanup_mps_cache()


if __name__ == "__main__":
    main()
