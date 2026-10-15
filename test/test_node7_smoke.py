"""生成小批验证必须区分检索遗漏、指定原文对照和实际回答。"""

import unittest
from langchain_core.documents import Document
from scripts.run_node7_answer_smoke import choose_fixture, scope_is_present
from generation.prompts import SYSTEM_PROMPT


class Node7SmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {"question_id": "r02", "answerable": False,
                         "scope_source_path": "documents/md/target.md"}
        self.docs = {"wrong": Document(page_content="干扰内容", metadata={"source": "other.md"}),
                     "target": Document(page_content="数值为破折号", metadata={"source": "target.md"})}

    def test_cached_mode_keeps_retrieval_miss_visible(self) -> None:
        result = choose_fixture(self.question, ["wrong"], self.docs, "cached")
        self.assertEqual(result, [self.docs["wrong"]])
        self.assertFalse(scope_is_present(self.question, result))

    def test_controlled_mode_uses_target_and_requires_complete_small_source(self) -> None:
        result = choose_fixture(self.question, ["wrong"], self.docs, "scope_controlled")
        self.assertEqual(result, [self.docs["target"]])
        self.assertTrue(scope_is_present(self.question, result))
        with self.assertRaisesRegex(ValueError, "完整切块"):
            choose_fixture(self.question, ["wrong"], {"wrong": self.docs["wrong"]}, "scope_controlled")
        oversized = {str(index): self.docs["target"] for index in range(7)}
        with self.assertRaisesRegex(ValueError, "完整切块"):
            choose_fixture(self.question, ["wrong"], oversized, "scope_controlled")

    def test_answerable_questions_never_get_oracle_evidence(self) -> None:
        question = {"answerable": True}
        self.assertEqual(choose_fixture(question, ["wrong"], self.docs, "scope_controlled"),
                         [self.docs["wrong"]])
        self.assertIsNone(scope_is_present(question, [self.docs["wrong"]]))

    def test_prompt_preserves_missing_units_and_internal_conflicts(self) -> None:
        self.assertIn("原文未注明单位", SYSTEM_PROMPT)
        self.assertIn("正文与备注或不同来源冲突", SYSTEM_PROMPT)
        self.assertIn("对象有歧义时分开说明或请求澄清", SYSTEM_PROMPT)
