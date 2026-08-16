from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from mm_stack.config import StackConfig
from mm_stack.db import connect_sqlite, ensure_schema, get_image_by_id, upsert_image_metadata
from mm_stack.documents import chunk_sections, extract_document
from mm_stack.ingestion import MultimodalIngestor


class DocumentExtractionTests(unittest.TestCase):
    def test_plain_text_is_cleaned_and_chunked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.md"
            path.write_text("Heading\n\n" + "searchable words " * 180, encoding="utf-8")
            sections = extract_document(path)
            chunks = chunk_sections(sections, chunk_size=300, overlap=40)
            self.assertGreater(len(chunks), 2)
            self.assertEqual(chunks[0].section_label, "Document")
            self.assertTrue(all(0 < len(chunk.text) <= 300 for chunk in chunks))

    def test_docx_open_xml_is_read_without_office_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.docx"
            xml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>Quarterly launch plan</w:t></w:r></w:p></w:body></w:document>'
            )
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", xml)
            sections = extract_document(path)
            self.assertEqual(sections[0].text, "Quarterly launch plan")

    def test_xlsx_shared_string_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.xlsx"
            shared = (
                '<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<si><t>Revenue</t></si></sst>'
            )
            sheet = (
                '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>42</v></c></row></sheetData>'
                '</worksheet>'
            )
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("xl/sharedStrings.xml", shared)
                archive.writestr("xl/worksheets/sheet1.xml", sheet)
            sections = extract_document(path)
            self.assertIn("Revenue | 42", sections[0].text)


class DocumentIngestionTests(unittest.TestCase):
    def test_document_dispatch_builds_nonvisual_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "manual.txt"
            path.write_text("The local search manual explains document ingestion.", encoding="utf-8")
            cfg = StackConfig(
                sqlite_path=root / "stack.db",
                lancedb_path=root / "vectors.lance",
                preprocessed_dir=root / "preprocessed",
            )
            ingestor = MultimodalIngestor(cfg)
            captured = []

            def process(candidates, safe_reprocess):
                captured.extend(candidates)
                return {"ingested": len(candidates), "failed": [], "skipped_duplicates": 0}

            ingestor._process_candidates = process  # type: ignore[method-assign]
            result = ingestor.ingest_path(path)
            self.assertEqual(result["document_files"], 1)
            self.assertEqual(result["document_chunks"], 1)
            self.assertEqual(captured[0].content_type, "document")
            self.assertFalse(captured[0].is_visual)

    def test_schema_persists_document_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = StackConfig(sqlite_path=root / "stack.db", lancedb_path=root / "vectors.lance")
            conn = connect_sqlite(cfg)
            ensure_schema(conn)
            upsert_image_metadata(
                conn,
                {
                    "id": "doc-1", "file_path": "/tmp/manual.pdf", "sha256_hash": "hash-1",
                    "width": 0, "height": 0, "caption": "manual.pdf — Page 2",
                    "summary": "setup instructions", "tags": ["document", "pdf"],
                    "schema_version": cfg.schema_version,
                    "embedding_model_clip": cfg.clip_model_name, "embedding_model_text": cfg.text_model_name,
                    "embedding_dimension_clip": cfg.clip_dimension, "embedding_dimension_text": cfg.text_dimension,
                    "embedding_schema_version_clip": cfg.clip_schema_version,
                    "embedding_schema_version_text": cfg.text_schema_version,
                    "content_type": "document", "section_label": "Page 2", "chunk_index": 3,
                },
            )
            conn.commit()
            row = get_image_by_id(conn, "doc-1")
            self.assertEqual(row["content_type"], "document")
            self.assertEqual(row["section_label"], "Page 2")
            self.assertEqual(row["chunk_index"], 3)
            conn.close()


if __name__ == "__main__":
    unittest.main()
