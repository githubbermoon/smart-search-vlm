from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRBlock:
    block_type: str
    text: str
    bbox: list[float]
    confidence: float


@dataclass
class PreparedImage:
    source_path: Path
    normalized_path: Path
    sha256_hash: str
    width: int
    height: int


@dataclass
class VLMOutput:
    caption: str
    summary: str
    category: str
    tags: list[str]
    raw_output: str


@dataclass
class SearchResult:
    image_id: str
    file_path: str
    caption: str
    summary: str
    tags: list[str]
    score: float
    routing_source: str
