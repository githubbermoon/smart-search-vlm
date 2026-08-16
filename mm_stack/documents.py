from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class DocumentSection:
    label: str
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    section_label: str
    text: str
    chunk_index: int


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag.lower() in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag.lower() in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> list[DocumentSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires the free 'pypdf' package") from exc

    reader = PdfReader(str(path))
    sections: list[DocumentSection] = []
    for index, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            sections.append(DocumentSection(f"Page {index}", text))
    return sections


def _extract_docx(path: Path) -> list[DocumentSection]:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs: list[str] = []
    for paragraph in root.iter():
        if paragraph.tag.endswith("}p"):
            text = "".join(node.text or "" for node in paragraph.iter() if node.tag.endswith("}t"))
            if text.strip():
                paragraphs.append(text.strip())
    text = _clean_text("\n".join(paragraphs))
    return [DocumentSection("Document", text)] if text else []


def _numeric_xml_sort(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)\.xml$", name)
    return (int(match.group(1)) if match else 0, name)


def _extract_pptx(path: Path) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            (name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
            key=_numeric_xml_sort,
        )
        for index, name in enumerate(names, start=1):
            root = ET.fromstring(archive.read(name))
            text = _clean_text("\n".join(node.text or "" for node in root.iter() if node.tag.endswith("}t")))
            if text:
                sections.append(DocumentSection(f"Slide {index}", text))
    return sections


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter() if node.tag.endswith("}t"))
        for item in root
    ]


def _extract_xlsx(path: Path) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    with zipfile.ZipFile(path) as archive:
        shared = _xlsx_shared_strings(archive)
        names = sorted(
            (name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)),
            key=_numeric_xml_sort,
        )
        for index, name in enumerate(names, start=1):
            root = ET.fromstring(archive.read(name))
            rows: list[str] = []
            for row in (node for node in root.iter() if node.tag.endswith("}row")):
                values: list[str] = []
                for cell in (node for node in row if node.tag.endswith("}c")):
                    cell_type = cell.attrib.get("t", "")
                    value_node = next((node for node in cell if node.tag.endswith("}v")), None)
                    inline = "".join(node.text or "" for node in cell.iter() if node.tag.endswith("}t"))
                    value = inline if cell_type == "inlineStr" else (value_node.text if value_node is not None else "")
                    if cell_type == "s" and value:
                        try:
                            value = shared[int(value)]
                        except (ValueError, IndexError):
                            pass
                    values.append(value or "")
                if any(value.strip() for value in values):
                    rows.append(" | ".join(values))
            text = _clean_text("\n".join(rows))
            if text:
                sections.append(DocumentSection(f"Sheet {index}", text))
    return sections


def extract_document(path: str | Path) -> list[DocumentSection]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        sections = _extract_pdf(path)
    elif suffix == ".docx":
        sections = _extract_docx(path)
    elif suffix == ".pptx":
        sections = _extract_pptx(path)
    elif suffix == ".xlsx":
        sections = _extract_xlsx(path)
    elif suffix in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(_read_text(path))
        text = _clean_text("".join(parser.parts))
        sections = [DocumentSection("Document", text)] if text else []
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows = csv.reader(io.StringIO(_read_text(path)), delimiter=delimiter)
        text = _clean_text("\n".join(" | ".join(cell.strip() for cell in row) for row in rows))
        sections = [DocumentSection("Table", text)] if text else []
    elif suffix == ".json":
        raw = _read_text(path)
        try:
            raw = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
        text = _clean_text(raw)
        sections = [DocumentSection("Document", text)] if text else []
    else:
        text = _clean_text(_read_text(path))
        sections = [DocumentSection("Document", text)] if text else []

    if not sections:
        raise ValueError("document contains no extractable text")
    return sections


def chunk_sections(
    sections: list[DocumentSection],
    *,
    chunk_size: int = 1200,
    overlap: int = 180,
) -> list[DocumentChunk]:
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section in sections:
        text = _clean_text(section.text)
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                boundary = max(text.rfind("\n", start + chunk_size // 2, end), text.rfind(" ", start + chunk_size // 2, end))
                if boundary > start:
                    end = boundary
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(section.label, chunk_text, chunk_index))
                chunk_index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
