import unittest

from pydantic import ValidationError

from ingestion.chunk_models import ChunkType
from ingestion.chunk_registry import STRATEGIES, build_chunker, recommend_strategy
from ingestion.chunking import ChunkingConfig, FixedTokenChunker, StructureAwareChunker, TokenCounter
from ingestion.document_model import (
    DocumentElement,
    DocumentFormat,
    ElementType,
    ParseStatus,
    ParsedDocument,
    ParserProvenance,
    SourceDocument,
    SourceLocation,
    TableCell,
    TableData,
)


def make_document(elements: list[DocumentElement], document_id: str = "doc_test") -> ParsedDocument:
    return ParsedDocument(
        source=SourceDocument(
            document_id=document_id,
            filename="测试.md",
            document_format=DocumentFormat.MARKDOWN,
            media_type="text/markdown",
            size_bytes=100,
            sha256="a" * 64,
        ),
        status=ParseStatus.SUCCESS,
        parser=ParserProvenance(parser_name="test", config_fingerprint="test"),
        elements=elements,
        root_element_ids=[element.element_id for element in elements],
    )


class TestChunking(unittest.TestCase):
    def test_config_rejects_inconsistent_ranges(self):
        with self.assertRaises(ValidationError):
            ChunkingConfig(target_tokens=700, max_tokens=600)
        with self.assertRaises(ValidationError):
            ChunkingConfig(max_tokens=100, overlap_tokens=100)

    def test_structure_chunk_keeps_heading_and_source_evidence(self):
        elements = [
            DocumentElement(
                element_id="heading", element_type=ElementType.HEADING, order=0,
                text="第一章 总则", level=1, source_location=SourceLocation(line_start=1, line_end=1),
            ),
            DocumentElement(
                element_id="paragraph", element_type=ElementType.PARAGRAPH, order=1,
                text="这是正文内容。" * 20, source_location=SourceLocation(line_start=2, line_end=2),
            ),
        ]
        chunks = StructureAwareChunker(ChunkingConfig(target_tokens=64, max_tokens=96, overlap_tokens=10)).chunk(
            make_document(elements)
        )
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.text.startswith("第一章 总则") for chunk in chunks))
        self.assertTrue(all("paragraph" in chunk.element_ids for chunk in chunks))
        self.assertTrue(all(chunk.source_locations for chunk in chunks))
        self.assertTrue(all(chunk.metadata["config_hash"] for chunk in chunks))

    def test_table_chunks_repeat_header_and_keep_formula(self):
        table = TableData(
            row_count=4,
            column_count=2,
            cells=[
                TableCell(row=0, column=0, text="项目", is_header=True),
                TableCell(row=0, column=1, text="金额", is_header=True),
                TableCell(row=1, column=0, text="甲"),
                TableCell(row=1, column=1, text="10", formula="=5+5"),
                TableCell(row=2, column=0, text="乙"),
                TableCell(row=2, column=1, text="20"),
                TableCell(row=3, column=0, text="丙"),
                TableCell(row=3, column=1, text="30"),
            ],
        )
        element = DocumentElement(
            element_id="table", element_type=ElementType.TABLE, order=0, text="预算表", table=table,
            source_location=SourceLocation(sheet_name="预算", cell_range="A1:B4"),
        )
        chunks = StructureAwareChunker(ChunkingConfig(table_rows_per_chunk=1)).chunk(make_document([element]))
        self.assertEqual(3, len(chunks))
        self.assertTrue(all(chunk.chunk_type == ChunkType.TABLE for chunk in chunks))
        self.assertTrue(all("项目 | 金额" in chunk.text for chunk in chunks))
        self.assertIn("公式：=5+5", chunks[0].text)

    def test_fixed_split_never_exceeds_max_for_long_unbroken_text(self):
        element = DocumentElement(
            element_id="text", element_type=ElementType.PARAGRAPH, order=0, text="知" * 3000,
        )
        chunks = FixedTokenChunker(ChunkingConfig(target_tokens=64, max_tokens=96, overlap_tokens=10)).chunk(
            make_document([element])
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 96 for chunk in chunks))

    def test_registry_preserves_legal_strategy(self):
        self.assertIn("legal_article_v1", STRATEGIES)
        document = make_document([
            DocumentElement(
                element_id="law", element_type=ElementType.PARAGRAPH, order=0,
                text="第一条 甲。\n第二条 乙。\n第三条 丙。",
            )
        ])
        self.assertEqual("legal_article_v1", recommend_strategy(document))
        self.assertEqual("legal_article_v1", build_chunker("legal_article_v1").strategy)

    def test_chunk_ids_are_stable(self):
        document = make_document([
            DocumentElement(element_id="p", element_type=ElementType.PARAGRAPH, order=0, text="稳定内容。")
        ])
        chunker = StructureAwareChunker()
        first = [chunk.chunk_id for chunk in chunker.chunk(document)]
        second = [chunk.chunk_id for chunk in chunker.chunk(document)]
        self.assertEqual(first, second)

    def test_token_counter_can_split_with_or_without_tiktoken(self):
        counter = TokenCounter()
        pieces = counter.split("测试内容" * 1000, 50)
        self.assertTrue(pieces)
        self.assertTrue(all(counter.count(piece) <= 50 for piece in pieces))


if __name__ == "__main__":
    unittest.main()
