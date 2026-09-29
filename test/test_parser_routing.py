import tempfile
import unittest
from pathlib import Path

from pydantic import SecretStr

from ingestion.document_model import DocumentFormat
from ingestion.parsers.mineru_contracts import MinerUSettings
from ingestion.parsers.routing import (
    ParserRoute,
    build_parse_batches,
    detect_document_format,
    plan_document,
)


class TestParserRouting(unittest.TestCase):
    def test_detects_supported_formats(self):
        self.assertEqual(detect_document_format("a.pdf"), DocumentFormat.PDF)
        self.assertEqual(detect_document_format("a.HTML"), DocumentFormat.HTML)
        self.assertEqual(detect_document_format("a.md"), DocumentFormat.MARKDOWN)
        self.assertEqual(detect_document_format("a.jpeg"), DocumentFormat.IMAGE)

    def test_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            detect_document_format("archive.zip")

    def test_html_is_separated_from_document_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / "a.pdf", root / "b.html", root / "c.md"]
            for path in paths:
                path.write_text(path.name, encoding="utf-8")
            planned = [plan_document(path) for path in paths]

        batches = build_parse_batches(planned)
        routes = {(batch.route, batch.model_version) for batch in batches}
        self.assertEqual(
            routes,
            {
                (ParserRoute.MINERU_DOCUMENT, "vlm"),
                (ParserRoute.MINERU_HTML, "MinerU-HTML"),
                (ParserRoute.LOCAL_TEXT, None),
            },
        )

    def test_batches_never_exceed_mineru_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index in range(51):
                path = root / f"{index}.pdf"
                path.write_bytes(str(index).encode("ascii"))
                paths.append(path)
            planned = [plan_document(path) for path in paths]

        batches = build_parse_batches(planned)
        self.assertEqual([len(batch.documents) for batch in batches], [50, 1])

    def test_token_is_not_exposed_by_settings_repr(self):
        settings = MinerUSettings(api_token=SecretStr("unit-test-token"))
        self.assertTrue(settings.api_token.get_secret_value().strip())
        self.assertNotIn(settings.api_token.get_secret_value(), repr(settings))


if __name__ == "__main__":
    unittest.main()
