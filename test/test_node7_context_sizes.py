"""上下文块数对照的同池截断、来源与统计边界测试。"""

import unittest

from scripts.compare_node7_context_sizes import context_metrics, prefixes, summarize


class ContextSizeTests(unittest.TestCase):
    def test_sizes_are_nested_prefixes_of_one_pool(self) -> None:
        pool = [str(i) for i in range(30)]
        selections = prefixes(pool, dict.fromkeys(pool))
        self.assertEqual([len(selections[k]) for k in (6, 10, 20)], [6, 10, 20])
        self.assertEqual(selections[6], selections[20][:6])
        self.assertEqual(selections[10], selections[20][:10])

    def test_invalid_pools_are_rejected(self) -> None:
        pool = [str(i) for i in range(20)]
        for ids, catalog in ((pool[:19], dict.fromkeys(pool)),
                             (pool + [pool[0]], dict.fromkeys(pool)), (pool, {})):
            with self.assertRaisesRegex(ValueError, "候选不足20条"):
                prefixes(ids, catalog)

    def test_acceptance_scoring_uses_actual_context_not_hardcoded_six(self) -> None:
        question = {"question_id": "q", "evaluation_suite": "acceptance_23", "answerable": True,
                    "relevant_document_ids": ["source"], "relevant_document_groups": [["source"]],
                    "evidence": [{"document_ids": ["source"], "quote": "金额800万元",
                                  "locator": {"type": "table"}}]}
        catalog = {str(i): {"document_id": "wrong", "text": "其他内容"} for i in range(10)}
        catalog["9"] = {"document_id": "correct", "text": "金额800万元"}
        tokens = dict.fromkeys(catalog, 10)
        requirements = [{"id": "amount", "document_ids": ["source"],
                         "alternatives": [{"quote": "金额800万元"}]}]
        six = context_metrics(question, list(catalog)[:6], catalog, {"source": "correct"}, tokens, requirements)
        ten = context_metrics(question, list(catalog), catalog, {"source": "correct"}, tokens, requirements)
        self.assertEqual(six["context_evidence"]["complete_evidence"], 0)
        self.assertFalse(six["diagnostic_complete"])
        self.assertEqual(ten["context_evidence"]["complete_evidence"], 1)
        self.assertTrue(ten["diagnostic_complete"])
        self.assertEqual(ten["estimated_tokens"], 100)
        self.assertEqual(ten["relevant_source_blocks"], 1)
        self.assertIsNone(ten["legacy_single_chunk_evidence"])

    def test_duplicate_or_unknown_context_has_explicit_error(self) -> None:
        for ids in (["a", "a"], ["unknown"], []):
            with self.assertRaisesRegex(ValueError, "上下文为空"):
                context_metrics({"question_id": "q"}, ids, {"a": {}}, {}, {}, [])

    def test_empty_summary_has_explicit_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "统计题组不能为空"):
            summarize({}, [])


if __name__ == "__main__":
    unittest.main()
