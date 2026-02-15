from __future__ import annotations

import re
from pathlib import Path

from .models import OCRBlock


LIST_PATTERN = re.compile(r"^\s*([\-\*\u2022]|\d+[\.)])\s+")
TABLE_PATTERN = re.compile(r"\b\d+\b.*\b\d+\b")


def _classify_block_type(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "text"
    if LIST_PATTERN.search(t):
        return "list"
    if "|" in t or "\t" in t:
        return "table"
    if TABLE_PATTERN.search(t) and len(t.split()) >= 4 and "  " in t:
        return "table"
    if len(t.split()) <= 10 and (t.isupper() or t.endswith(":")):
        return "title"
    return "text"


def _load_cg_image(image_path: Path):
    from Cocoa import NSURL
    from Quartz import CGImageSourceCreateImageAtIndex, CGImageSourceCreateWithURL

    url = NSURL.fileURLWithPath_(str(image_path))
    src = CGImageSourceCreateWithURL(url, None)
    if src is None:
        raise RuntimeError(f"Cannot read image for OCR: {image_path}")
    cg_image = CGImageSourceCreateImageAtIndex(src, 0, None)
    if cg_image is None:
        raise RuntimeError(f"Cannot decode image for OCR: {image_path}")
    return cg_image


def extract_ocr_structured(image_path: Path, width: int, height: int) -> tuple[list[OCRBlock], float]:
    import Vision

    request = Vision.VNRecognizeTextRequest.alloc().init()
    if hasattr(request, "setRecognitionLevel_"):
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    if hasattr(request, "setUsesLanguageCorrection_"):
        request.setUsesLanguageCorrection_(True)
    if hasattr(request, "setRecognitionLanguages_"):
        try:
            request.setRecognitionLanguages_(["en-US", "hi-IN"])
        except Exception:
            pass

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(_load_cg_image(image_path), None)
    result = handler.performRequests_error_([request], None)
    if isinstance(result, tuple):
        ok, err = result
    else:
        ok, err = bool(result), None
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {err}")

    blocks: list[OCRBlock] = []
    confidences: list[float] = []

    for obs in request.results() or []:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        cand = candidates[0]
        text = str(cand.string() or "").strip()
        if not text:
            continue

        try:
            conf = float(cand.confidence())
        except Exception:
            conf = 0.0

        bbox = obs.boundingBox()
        x = float(getattr(bbox, "origin").x)
        y = float(getattr(bbox, "origin").y)
        w = float(getattr(bbox, "size").width)
        h = float(getattr(bbox, "size").height)

        # Vision uses normalized bottom-left coordinates.
        x1 = x * width
        y1 = (1.0 - (y + h)) * height
        x2 = (x + w) * width
        y2 = (1.0 - y) * height

        blocks.append(
            OCRBlock(
                block_type=_classify_block_type(text),
                text=text,
                bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                confidence=round(conf, 4),
            )
        )
        confidences.append(conf)

    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    return blocks, avg_conf
