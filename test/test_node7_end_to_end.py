"""端到端评测必须执行真实检索接口、隔离会话，并区分召回与回答成绩。"""

import unittest
from unittest.mock import patch

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from core import config
from generation.prompts import NO_EVIDENCE_ANSWER
from scripts.run_node7_end_to_end import ObservedRagService, check_failure_limit, evaluation_model, retrieval_metrics, run_case, summarize_run
from test.test_grounded_answers import FakeVectorService, make_document


class EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.histories = {}
        for key, value in {"retrieval_strategy": "dense", "rerank_strategy": "none", "evidence_expansion": "none"}.items():
            item = patch.object(config, key, value)
            item.start()
            self.addCleanup(item.stop)
        self.doc = make_document()
        self.catalog = {"chunk-1": {"document_id": "doc-1", "text": self.doc.page_content}}
        self.question = {"question_id": "test", "query": "预计财务费用？", "answerable": True,
                         "evaluation_suite": "retrieval_55", "relevant_document_ids": ["item"],
                         "evidence": [{"document_id": "item", "quote": "47百万元"}]}

    def service(self, vector: FakeVectorService, model: object) -> ObservedRagService:
        return ObservedRagService(vector_service=vector, chat_model=model,
                                  history_factory=lambda key: self.histories.setdefault(key, InMemoryChatMessageHistory()))

    def test_real_chain_retrieves_once_and_records_actual_evidence(self) -> None:
        vector = FakeVectorService([self.doc])
        with patch.object(vector, "get_vector_docs", wraps=vector.get_vector_docs) as search:
            rag = self.service(vector, FakeListChatModel(responses=["47百万元。[S1]", "47百万元。[S1]"]))
            first = run_case(rag, self.question, self.catalog, {"item": "doc-1"}, "first")
            self.assertEqual(search.call_count, 1)
            second = run_case(rag, self.question, self.catalog, {"item": "doc-1"}, "second")
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["ranked_chunk_ids"], ["chunk-1"])
        self.assertEqual(first["retrieval_metrics"]["recall_at_10"], 1)
        self.assertIn("报告.md", first["display_answer"])
        self.assertEqual(first["evidence"], second["evidence"])
        self.assertEqual(len(self.histories["first"].messages), 2)
        self.assertEqual(len(self.histories["second"].messages), 2)
        self.assertEqual(first["review_status"], "awaiting_source_based_review")

    def test_refusal_never_receives_recall_or_oracle_evidence(self) -> None:
        question = {**self.question, "answerable": False, "scope_source_path": "missing.md"}
        rag = self.service(FakeVectorService([self.doc]), FakeListChatModel(responses=["无法确定。[S1]"]))
        row = run_case(rag, question, self.catalog, {}, "refusal")
        self.assertIsNone(row["retrieval_metrics"])
        self.assertFalse(row["scope_document_present"])
        self.assertEqual(row["evidence"][0]["source"], "报告.md")

    def test_generation_failure_keeps_retrieval_and_next_failure_does_not_reuse_trace(self) -> None:
        def fail(_: object) -> str:
            raise RuntimeError("模拟模型超时")

        vector = FakeVectorService([self.doc])
        rag = self.service(vector, RunnableLambda(fail))
        row = run_case(rag, self.question, self.catalog, {"item": "doc-1"}, "failed")
        self.assertEqual(row["stage"], "generation")
        self.assertEqual(row["ranked_chunk_ids"], ["chunk-1"])
        with patch.object(rag.retrieval_pipeline, "run", side_effect=RuntimeError("模拟召回失败")):
            row = run_case(rag, self.question, self.catalog, {}, "failed-retrieval")
        self.assertEqual(row["stage"], "retrieval")
        self.assertNotIn("evidence", row)

    def test_empty_evidence_bypasses_model_without_fake_semantic_score(self) -> None:
        def fail(_: object) -> str:
            self.fail("无证据不应调用模型")

        rag = self.service(FakeVectorService([]), RunnableLambda(fail))
        row = run_case(rag, self.question, self.catalog, {"item": "doc-1"}, "empty")
        self.assertEqual(row["raw_answer"], NO_EVIDENCE_ANSWER)
        self.assertFalse(row["model_called"])
        self.assertTrue(row["generation_complete"])
        summary = summarize_run({"test": row}, 78)
        self.assertEqual(summary["expected"], 78)
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(summary["semantic_review"], "pending")
        self.assertNotIn("accuracy", summary)

    def test_acceptance_metric_retains_equivalent_groups(self) -> None:
        question = {**self.question, "evaluation_suite": "acceptance_23",
                    "relevant_document_groups": [["item", "copy"]],
                    "evidence": [{"document_ids": ["item", "copy"], "quote": "47百万元",
                                  "locator": {"type": "line"}}]}
        score = retrieval_metrics(question, ["chunk-1"], self.catalog, {"item": "doc-1", "copy": "doc-1"})
        self.assertEqual(score["document_recall_at_6"], 1)
        self.assertEqual(score["complete_evidence_at_6"], 1)

    def test_ten_block_metrics_are_not_mislabeled_as_six(self) -> None:
        score = retrieval_metrics(self.question, ["chunk-1"], self.catalog, {"item": "doc-1"}, final_k=10)
        self.assertEqual(score["recall_at_10"], 1)
        self.assertNotIn("recall_at_6", score)

    def test_application_passes_ten_blocks_to_generation(self) -> None:
        docs = [make_document() for _ in range(12)]
        for index, doc in enumerate(docs):
            doc.metadata["chunk_id"] = f"chunk-{index}"
        rag = self.service(FakeVectorService(docs), FakeListChatModel(responses=["有依据。[S10]"]))
        result = rag.retrieve_with_trace("测试默认块数")
        self.assertEqual(config.hybrid_final_k, 10)
        self.assertEqual(len(result.documents), 10)
        self.assertEqual(rag.hybrid_retriever.final_k, 10)

    def test_batch_stops_after_three_failures_and_success_resets_streak(self) -> None:
        failed = {"status": "failed", "error": "HTTP 403"}
        self.assertEqual(check_failure_limit(failed, 0), 1)
        self.assertEqual(check_failure_limit(failed, 1), 2)
        self.assertEqual(check_failure_limit({"status": "success"}, 2), 0)
        with self.assertRaisesRegex(RuntimeError, "连续失败3题.*HTTP 403"):
            check_failure_limit(failed, 2)

    def test_explicit_glm_evaluation_is_isolated_from_qwen(self) -> None:
        with patch.dict("os.environ", {"ZHIPU_API_KEY": "test-key"}):
            with patch("scripts.run_node7_end_to_end.build_chat_model") as qwen:
                identity, pool = evaluation_model("zhipu")
        qwen.assert_not_called()
        self.assertEqual(identity["model"], config.zhipu_chat_model)
        self.assertTrue(identity["enable_thinking"])
        self.assertEqual(identity["reasoning_effort"], "low")
        self.assertEqual([name for name, _ in pool.members], ["zhipu"])
        self.assertNotIn("test-key", str(identity))

    def test_unknown_evaluation_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持的评测供应商"):
            evaluation_model("unknown")
