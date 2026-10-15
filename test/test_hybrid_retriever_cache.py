"""混合检索缓存和候选去重的回归测试。"""

import unittest

from langchain_core.documents import Document

from retrieval.hybrid_retriever import HybridRetrieverService, rrf_fuse


class TestHybridRetrieverCache(unittest.TestCase):
    def test_reuses_bm25_until_document_count_changes(self) -> None:
        documents = [Document(page_content="甲乙", metadata={"chunk_id": "c1"})]
        reads = 0

        def get_all() -> list[Document]:
            nonlocal reads
            reads += 1
            return list(documents)

        retriever = HybridRetrieverService(
            get_vector_docs=lambda _query, _k: [],
            get_all_docs=get_all,
            get_document_count=lambda: len(documents),
        )
        retriever.retrieve("甲")
        retriever.retrieve("甲")
        self.assertEqual(1, reads)

        documents.append(Document(page_content="丙丁", metadata={"chunk_id": "c2"}))
        retriever.retrieve("丙")
        self.assertEqual(2, reads)

    def test_fusion_uses_stable_chunk_id_for_same_evidence(self) -> None:
        dense = Document(page_content="同一证据", metadata={"chunk_id": "c1", "source": "文档"})
        lexical = Document(page_content="同一证据", metadata={"chunk_id": "c1", "source": "文档"})
        self.assertEqual(1, len(rrf_fuse([[dense], [lexical]], top_k=10)))

    def test_dense_strategy_does_not_build_bm25(self) -> None:
        retriever = HybridRetrieverService(
            get_vector_docs=lambda _query, _k: [Document(page_content="向量结果")],
            get_all_docs=lambda: self.fail("仅向量策略不应读取全库文档"),
            strategy_id="dense",
        )
        self.assertEqual("向量结果", retriever.retrieve("问题")[0].page_content)

    def test_unknown_strategy_reports_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持的召回策略"):
            HybridRetrieverService(
                get_vector_docs=lambda _query, _k: [],
                get_all_docs=lambda: [],
                strategy_id="unknown",
            )

    def test_same_count_replacement_can_invalidate_cache(self) -> None:
        docs = [Document(page_content="旧内容", metadata={"chunk_id": "old"})]
        retriever = HybridRetrieverService(lambda _query, _k: [], lambda: docs,
                                           get_document_count=lambda: len(docs), strategy_id="bm25")
        self.assertEqual(retriever.retrieve("旧")[0].metadata["chunk_id"], "old")
        docs[0] = Document(page_content="新内容", metadata={"chunk_id": "new"})
        retriever.invalidate()
        self.assertEqual(retriever.retrieve("新")[0].metadata["chunk_id"], "new")

    def test_duplicate_in_one_ranking_does_not_inflate_fusion_score(self) -> None:
        a, b = [Document(page_content=key, metadata={"chunk_id": key}) for key in ("a", "b")]
        ranked = rrf_fuse([[a, a, a], [b], [b]], top_k=2)
        self.assertEqual(ranked[0].metadata["chunk_id"], "b")


if __name__ == "__main__":
    unittest.main()
