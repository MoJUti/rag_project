"""邻接证据的预算、章节边界、去重与来源校验。"""

import unittest

from langchain_core.documents import Document

from ingestion.chunk_models import KnowledgeChunk, ChunkType
from ingestion.document_model import SourceLocation
from retrieval.evidence_expansion import NeighborhoodExpander
from retrieval.pipeline import RetrievalPipeline


def chunk(key: str, text: str, section: str = "甲", document_id: str = "doc-a") -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=key, document_id=document_id, strategy="test",
                          chunk_type=ChunkType.TEXT, text=text, token_count=len(text),
                          section_path=[section], source_locations=[SourceLocation(line_start=1, line_end=2)])


def seed(row: KnowledgeChunk) -> Document:
    return Document(page_content=row.text, metadata={"chunk_id": row.chunk_id,
                                                    "document_id": row.document_id, "source": "原文.md"})


class EvidenceExpansionTest(unittest.TestCase):
    def test_original_order_not_chunk_id_order_and_own_source_is_kept(self) -> None:
        rows = [chunk("z", "前文"), chunk("a", "命中"), chunk("y", "后文")]
        rows[0].source_locations = [SourceLocation(line_start=10, line_end=11)]
        expander = NeighborhoodExpander(rows, count_tokens=len)
        result = expander("问题", [seed(rows[1])])
        self.assertEqual([doc.metadata["chunk_id"] for doc in result], ["a", "z", "y"])
        self.assertEqual(result[1].metadata["source"], "原文.md")
        self.assertIn('"line_start": 10', result[1].metadata["source_locations_json"])
        self.assertEqual(result[1].metadata["expanded_from_chunk_id"], "a")

    def test_budget_keeps_seeds_and_can_skip_large_neighbor(self) -> None:
        rows = [chunk("before", "超长前文"), chunk("hit", "很长的命中块"), chunk("after", "短")]
        result = NeighborhoodExpander(rows, extra_token_budget=1, count_tokens=len)("问题", [seed(rows[1])])
        self.assertEqual([doc.metadata["chunk_id"] for doc in result], ["hit", "after"])

    def test_cannot_cross_document_or_intervening_section(self) -> None:
        rows = [chunk("a", "命中"), chunk("b", "别章", section="乙"), chunk("c", "又是甲章"),
                chunk("other", "另一文档", document_id="doc-b")]
        result = NeighborhoodExpander(rows, radius=2, count_tokens=len)("问题", [seed(rows[0])])
        self.assertEqual([doc.metadata["chunk_id"] for doc in result], ["a"])

    def test_neighbor_already_recalled_is_not_added_twice(self) -> None:
        rows = [chunk("a", "甲"), chunk("b", "乙"), chunk("c", "丙")]
        result = NeighborhoodExpander(rows, count_tokens=len)("问题", [seed(rows[0]), seed(rows[1])])
        self.assertEqual([doc.metadata["chunk_id"] for doc in result], ["a", "b", "c"])

    def test_stale_index_and_unknown_hit_fail_clearly(self) -> None:
        row = chunk("a", "旧文")
        expander = NeighborhoodExpander([row], count_tokens=len)
        changed = seed(row)
        changed.page_content = "新文"
        with self.assertRaisesRegex(ValueError, "索引内容不一致"):
            expander("问题", [changed])
        with self.assertRaisesRegex(ValueError, "不在证据邻接表"):
            expander("问题", [seed(chunk("unknown", "内容"))])

    def test_deleted_neighbor_requires_index_resynchronization(self) -> None:
        rows = [chunk("a", "仍在库中"), chunk("b", "已删除")]
        expander = NeighborhoodExpander(rows, count_tokens=len)
        with self.assertRaisesRegex(ValueError, "切块集合不一致"):
            expander.validate_index([seed(rows[0])])

    def test_distinct_tables_in_same_section_are_not_joined(self) -> None:
        rows = [chunk("a", "表一"), chunk("b", "表二")]
        for index, row in enumerate(rows):
            row.chunk_type = ChunkType.TABLE
            row.element_ids = [f"table-{index}"]
        result = NeighborhoodExpander(rows, count_tokens=len)("问题", [seed(rows[0])])
        self.assertEqual(len(result), 1)

    def test_pipeline_trace_distinguishes_added_context(self) -> None:
        rows = [chunk("a", "甲"), chunk("b", "乙")]

        class Retriever:
            def get_candidates(self, query: str, limit: int) -> list[Document]:
                return [seed(rows[0])]

        result = RetrievalPipeline(Retriever(), final_k=1,
                                   evidence_expander=NeighborhoodExpander(rows, count_tokens=len)).run("问题")
        self.assertEqual(result.trace[-1].added_chunk_ids, ["b"])


if __name__ == "__main__":
    unittest.main()
