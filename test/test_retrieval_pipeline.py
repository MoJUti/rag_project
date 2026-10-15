"""检索插件组合、候选预算、来源追踪及故障的行为测试。"""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from retrieval.pipeline import RERANKER_FACTORIES, RetrievalPipeline
from retrieval.qwen_reranker import RerankResult


def documents(count: int) -> list[Document]:
    return [Document(page_content=f"原始证据{i}", metadata={"chunk_id": f"c{i}"})
            for i in range(count)]


class FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.calls: list[tuple[str, int]] = []

    def get_candidates(self, query: str, limit: int) -> list[Document]:
        self.calls.append((query, limit))
        return self.docs[:limit]


class ReverseReranker:
    def __init__(self) -> None:
        self.closed = False

    def rerank(self, query: str, docs: list[str]) -> RerankResult:
        return RerankResult(list(reversed(range(len(docs)))), 123, "req-test")

    def close(self) -> None:
        self.closed = True


class RetrievalPipelineTest(unittest.TestCase):
    def test_default_final_budget_is_ten(self) -> None:
        retriever = FakeRetriever(documents(12))
        result = RetrievalPipeline(retriever).run("问题")
        self.assertEqual(retriever.calls, [("问题", 10)])
        self.assertEqual(len(result.documents), 10)

    def test_disabled_rerank_preserves_order_and_budget(self) -> None:
        retriever = FakeRetriever(documents(8))
        pipeline = RetrievalPipeline(retriever, final_k=2, candidate_k=8)
        result = pipeline.run("问题")
        self.assertEqual(retriever.calls, [("问题", 2)])
        self.assertEqual([doc.metadata["chunk_id"] for doc in result.documents], ["c0", "c1"])
        self.assertNotIn("rerank", [stage.stage for stage in result.trace])

    def test_rerank_sees_candidates_before_final_cut_and_keeps_sources(self) -> None:
        retriever = FakeRetriever(documents(8))
        reranker = ReverseReranker()
        with patch.dict(RERANKER_FACTORIES, {"test": lambda: reranker}):
            result = RetrievalPipeline(retriever, final_k=2, candidate_k=8,
                                       rerank_strategy="test").run("问题")
        self.assertEqual(retriever.calls, [("问题", 8)])
        self.assertEqual([doc.metadata["chunk_id"] for doc in result.documents], ["c7", "c6"])
        self.assertEqual(result.documents[0].page_content, "原始证据7")
        stage = next(stage for stage in result.trace if stage.stage == "rerank")
        self.assertEqual(stage.total_tokens, 123)
        self.assertEqual(stage.request_id, "req-test")
        self.assertTrue(reranker.closed)

    def test_empty_recall_never_calls_paid_plugin(self) -> None:
        with patch.dict(RERANKER_FACTORIES, {"test": lambda: self.fail("空候选不能调用模型")}):
            result = RetrievalPipeline(FakeRetriever([]), rerank_strategy="test").run("问题")
        self.assertEqual(result.documents, [])

    def test_multiquery_fuses_and_deduplicates(self) -> None:
        retriever = FakeRetriever(documents(3))
        pipeline = RetrievalPipeline(retriever, final_k=2,
                                     query_transform=lambda _: ["甲", "乙", "甲"])
        result = pipeline.run("复合问题")
        self.assertEqual(retriever.calls, [("甲", 2), ("乙", 2)])
        self.assertEqual(len(result.documents), 2)
        self.assertEqual(result.trace[0].queries, ["甲", "乙"])

    def test_invalid_plugin_and_budget_fail_at_initialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持的重排策略"):
            RetrievalPipeline(FakeRetriever([]), rerank_strategy="unknown")
        with self.assertRaisesRegex(ValueError, "候选数量"):
            RetrievalPipeline(FakeRetriever([]), final_k=7, candidate_k=6)

    def test_failure_reports_stage_and_closes_client(self) -> None:
        reranker = ReverseReranker()
        with patch.object(reranker, "rerank", side_effect=RuntimeError("HTTP 429")):
            with patch.dict(RERANKER_FACTORIES, {"test": lambda: reranker}):
                with self.assertRaisesRegex(RuntimeError, "重排阶段失败.*HTTP 429"):
                    RetrievalPipeline(FakeRetriever(documents(2)), rerank_strategy="test").run("问题")
        self.assertTrue(reranker.closed)


if __name__ == "__main__":
    unittest.main()
