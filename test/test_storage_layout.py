from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from ingestion.parsers.execution import ParserExecutionService
from ingestion.storage import StorageLayout


class TestStorageLayout(unittest.TestCase):
    def test_formal_parse_persists_source_and_unified_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "sample.md"
            source.parent.mkdir()
            source.write_text("# 标题\n\n正式知识库内容。", encoding="utf-8")
            layout = StorageLayout(root / "storage")
            service = ParserExecutionService(storage_layout=layout)

            documents = asyncio.run(service.parse_paths([source]))

            self.assertEqual(len(documents), 1)
            document = documents[0]
            stored_source = layout.source_dir(document.source.document_id) / source.name
            parsed = layout.parsed_dir(document.source.document_id) / "parsed_document.json"
            self.assertTrue(stored_source.is_file())
            self.assertTrue(parsed.is_file())
            self.assertEqual(document.source.source_uri, str(stored_source.resolve()))
            self.assertTrue(layout.assets.is_dir())
            self.assertTrue(layout.chunks.is_dir())
            self.assertTrue(layout.indexes.is_dir())

    def test_evaluation_namespace_is_separate_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = StorageLayout(Path(directory) / "storage")
            evaluation = layout.evaluation("zh_public_minibench_v0.1")
            evaluation.ensure()
            self.assertTrue(evaluation.parsed.is_dir())
            self.assertTrue(evaluation.chunks.is_dir())
            self.assertNotEqual(evaluation.parsed, layout.parsed)
            with self.assertRaises(ValueError):
                layout.evaluation("../escape")


if __name__ == "__main__":
    unittest.main()
