"""离线检索对照的缓存边界、来源约束与上下文计数测试。"""

import copy
import unittest

from scripts.compare_node7_retrieval_gaps import measure_context, verified_rerank


class RetrievalGapComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = {"trace": [
            {"stage": "query", "queries": ["问题"]},
            {"stage": "recall", "chunk_ids": ["a", "b"]},
            {"stage": "rerank", "chunk_ids": ["b", "a"]},
        ]}

    def test_matching_cache_replays_full_permutation(self) -> None:
        self.assertEqual(verified_rerank(self.cache, "问题", ["a", "b"]), ["b", "a"])

    def test_missing_stage_has_explicit_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少查询"):
            verified_rerank({"trace": []}, "问题", ["a", "b"])

    def test_changed_query_or_input_order_is_rejected(self) -> None:
        for query, pool in (("其他问题", ["a", "b"]), ("问题", ["b", "a"])):
            with self.assertRaisesRegex(ValueError, "查询或有序候选"):
                verified_rerank(self.cache, query, pool)

    def test_duplicate_or_incomplete_ranking_is_rejected(self) -> None:
        for ids in (["a", "a"], ["a"], ["a", "c"]):
            row = copy.deepcopy(self.cache)
            row["trace"][-1]["chunk_ids"] = ids
            with self.assertRaisesRegex(ValueError, "完整候选"):
                verified_rerank(row, "问题", ["a", "b"])

    def test_expanded_context_does_not_claim_seed6_recall(self) -> None:
        question = {"question_id": "q", "evaluation_suite": "acceptance_23", "answerable": True,
                    "relevant_document_groups": [["source"]],
                    "evidence": [{"document_ids": ["source"], "quote": "金额800万元",
                                  "locator": {"type": "table"}}]}
        catalog = {"a": {"document_id": "d", "text": "其他内容"},
                   "b": {"document_id": "d", "text": "金额800万元"},
                   "c": {"document_id": "wrong", "text": "金额800万元"}}
        requirements = [{"id": "amount", "document_ids": ["source"],
                         "alternatives": [{"quote": "金额800万元"}]}]
        args = (catalog, {"source": "d"}, {"a": 10, "b": 20, "c": 20}, requirements)
        wrong_source = measure_context(question, ["a", "c"], ["a"], *args)
        self.assertFalse(wrong_source["diagnostic_complete"])
        result = measure_context(question, ["a", "b"], ["a"], *args)
        self.assertTrue(result["diagnostic_complete"])
        self.assertEqual(result["added_estimated_tokens"], 20)
        self.assertEqual(result["total_estimated_tokens"], 30)
        self.assertIsNone(result["document_recall_seed6"])

    def test_duplicate_context_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "切块重复"):
            measure_context({"question_id": "q"}, ["a", "a"], ["a"], {"a": {}}, {}, {}, [])


if __name__ == "__main__":
    unittest.main()
