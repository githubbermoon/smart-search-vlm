from __future__ import annotations

import io
from pathlib import Path

from .config import StackConfig
from .models import PreparedImage
from .utils import sha256_bytes


def _load_pillow():
    try:
        from PIL import Image, ImageOps

        return Image, ImageOps
    except Exception as exc:
        raise RuntimeError("Pillow is required for preprocessing. Install with: uv pip install pillow") from exc


def preprocess_image(image_path: Path, cfg: StackConfig) -> PreparedImage:
    Image, ImageOps = _load_pillow()

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        max_dim = max(w, h)
        if max_dim > cfg.max_image_dim:
            scale = cfg.max_image_dim / float(max_dim)
            resized_w = max(1, int(round(w * scale)))
            resized_h = max(1, int(round(h * scale)))
            img = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        else:
            resized_w, resized_h = w, h

        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95, optimize=True)
        normalized_bytes = buffer.getvalue()

    sha = sha256_bytes(normalized_bytes)
    cfg.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = cfg.preprocessed_dir / f"{sha}.jpg"
    if not normalized_path.exists():
        normalized_path.write_bytes(normalized_bytes)

    return PreparedImage(
        source_path=image_path,
        normalized_path=normalized_path,
        sha256_hash=sha,
        width=resized_w,
        height=resized_h,
    )
