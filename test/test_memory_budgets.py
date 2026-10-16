"""Token触发与原文保留实验不应退化为固定轮数截断。"""

import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from core import config
from memory.session_context import build_context, estimate_tokens, validate_summary
from scripts.compare_memory_budgets import budget_association_ids


class DefaultMemoryBudgetTest(unittest.TestCase):
    def test_selected_defaults_keep_originals_then_compress_within_budget(self) -> None:
        self.assertEqual(config.memory_history_max_tokens, 32768)
        self.assertEqual(config.memory_summary_max_tokens, 4096)
        self.assertEqual(config.memory_compression_trigger, "tokens")
        summarize = Mock(return_value="保留用户已确认的条件")
        short = [HumanMessage(content="用户条件"), AIMessage(content="收到")] * 10
        context, _ = build_context(short, {}, summarize)
        self.assertEqual(context, short)
        summarize.assert_not_called()
        long = [HumanMessage(content="中文条件" * 1000), AIMessage(content="收到")] * 20
        self.assertGreater(estimate_tokens(long), 32768)
        context, state = build_context(long, {}, summarize)
        summarize.assert_called_once()
        self.assertEqual(state["status"], "ready")
        self.assertLessEqual(estimate_tokens(context), 32768)
        self.assertEqual(context[-1], long[-1])


class MemoryBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        values = {"memory_compression_trigger": "tokens", "memory_summary_max_tokens": 60,
                  "memory_history_max_tokens": 1000, "memory_summary_enabled": True}
        for name, value in values.items():
            item = patch.object(config, name, value)
            item.start()
            self.addCleanup(item.stop)
        self.messages = [message for i in range(20) for message in (
            HumanMessage(content=f"第{i}轮条件：" + "中文" * 15), AIMessage(content="确认条件"))]

    def test_under_budget_keeps_more_than_three_rounds_without_summary(self) -> None:
        summarize = Mock(return_value="摘要")
        with patch.object(config, "memory_history_max_tokens", 10000):
            context, state = build_context(self.messages, {}, summarize)
        self.assertEqual(context, self.messages)
        self.assertEqual(state["covered_messages"], 0)
        summarize.assert_not_called()

    def test_over_budget_retains_raw_suffix_and_avoids_duplicate_calls(self) -> None:
        summarize = Mock(return_value="保留必要条件")
        with patch.object(config, "memory_history_max_tokens", 600):
            context, state = build_context(self.messages, {}, summarize)
            repeated, _ = build_context(self.messages, state, summarize)
        summarize.assert_called_once()
        self.assertEqual(context, repeated)
        self.assertGreater(sum(m.type == "human" for m in context), 3)
        self.assertLessEqual(estimate_tokens(context), 600)
        self.assertEqual(context[-1], self.messages[-1])

    def test_summary_token_cap_replaces_character_cap_only_when_explicit(self) -> None:
        with patch.object(config, "memory_summary_max_chars", 2):
            validate_summary("超过两个字但没有超过60Token")
        with patch.object(config, "memory_summary_max_tokens", 1):
            with self.assertRaisesRegex(ValueError, "Token上限"):
                validate_summary("很长的中文摘要")

    def test_invalid_token_policy_is_rejected(self) -> None:
        for value in (None, 0, 1001):
            with patch.object(config, "memory_summary_max_tokens", value):
                with self.assertRaisesRegex(ValueError, "摘要Token上限"):
                    build_context(self.messages, {}, Mock())

    def test_association_check_does_not_cross_other_project_or_partial_number(self) -> None:
        probes = [{"group": "P001", "amount": 7001}]
        self.assertEqual(budget_association_ids("P001 预算7001元", probes), ["P001"])
        self.assertEqual(budget_association_ids("P001 未知；P002 预算7001元", probes), [])
        self.assertEqual(budget_association_ids("P001 预算70010元", probes), [])


if __name__ == "__main__":
    unittest.main()
