import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from ingestion.document_model import (
    DocumentElement,
    DocumentFormat,
    ElementType,
    ParseStatus,
    ParsedDocument,
    ParserProvenance,
    SourceDocument,
    config_fingerprint,
    sha256_file,
    stable_document_id,
    stable_element_id,
)


class TestDocumentModel(unittest.TestCase):
    def test_stable_ids_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "示例.txt"
            path.write_text("同一内容", encoding="utf-8")
            digest = sha256_file(path)

        document_id = stable_document_id(digest)
        first = stable_element_id(document_id, ElementType.PARAGRAPH, 0, "line:1")
        second = stable_element_id(document_id, ElementType.PARAGRAPH, 0, "line:1")
        self.assertEqual(first, second)
        self.assertTrue(document_id.startswith("doc_"))

    def test_config_fingerprint_ignores_key_order(self):
        self.assertEqual(
            config_fingerprint({"model": "vlm", "ocr": False}),
            config_fingerprint({"ocr": False, "model": "vlm"}),
        )

    def test_document_rejects_missing_parent(self):
        source = SourceDocument(
            document_id="doc_abc",
            filename="demo.txt",
            document_format=DocumentFormat.TXT,
            media_type="text/plain",
            size_bytes=4,
            sha256="abc",
        )
        parser = ParserProvenance(
            parser_name="local_text",
            config_fingerprint=config_fingerprint({}),
        )
        element = DocumentElement(
            element_id="el_1",
            element_type=ElementType.PARAGRAPH,
            order=0,
            text="测试",
            parent_id="el_missing",
        )

        with self.assertRaises(ValidationError):
            ParsedDocument(
                source=source,
                status=ParseStatus.SUCCESS,
                parser=parser,
                elements=[element],
                root_element_ids=["el_1"],
            )

    def test_success_document_must_have_elements(self):
        source = SourceDocument(
            document_id="doc_abc",
            filename="demo.txt",
            document_format=DocumentFormat.TXT,
            media_type="text/plain",
            size_bytes=0,
            sha256="abc",
        )
        parser = ParserProvenance(
            parser_name="local_text",
            config_fingerprint=config_fingerprint({}),
        )
        with self.assertRaises(ValidationError):
            ParsedDocument(source=source, status=ParseStatus.SUCCESS, parser=parser)


if __name__ == "__main__":
    unittest.main()
