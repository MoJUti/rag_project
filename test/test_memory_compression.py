import os
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core import config
from memory import history_store as hs


class TestMemoryCompression(unittest.TestCase):
    def setUp(self):
        self._backup = {
            "memory_keep_recent_rounds": config.memory_keep_recent_rounds,
            "memory_summary_trigger_rounds": config.memory_summary_trigger_rounds,
            "memory_history_max_tokens": config.memory_history_max_tokens,
            "memory_summary_max_chars": config.memory_summary_max_chars,
            "memory_summary_enabled": config.memory_summary_enabled,
            "memory_summary_tag": config.memory_summary_tag,
        }

        config.memory_keep_recent_rounds = 4
        config.memory_summary_trigger_rounds = 6
        config.memory_history_max_tokens = 1500
        config.memory_summary_max_chars = 800
        config.memory_summary_enabled = True
        config.memory_summary_tag = "[SESSION_SUMMARY]"

    def tearDown(self):
        for k, v in self._backup.items():
            setattr(config, k, v)

    def _build_round_messages(self, rounds: int, chars_per_msg: int = 20):
        messages = []
        for idx in range(1, rounds + 1):
            user_text = f"用户第{idx}轮:" + ("U" * chars_per_msg)
            ai_text = f"助手第{idx}轮:" + ("A" * chars_per_msg)
            messages.append(HumanMessage(content=user_text))
            messages.append(AIMessage(content=ai_text))
        return messages

    def _count_human_rounds(self, messages):
        return sum(1 for m in messages if m.type == "human")

    def test_no_summary_when_rounds_below_trigger(self):
        source = self._build_round_messages(rounds=3)
        with patch("memory.history_store._summarize_messages") as mock_summary:
            compressed = hs._compress_messages(source)

        self.assertFalse(any(hs._is_summary_message(m) for m in compressed))
        self.assertEqual(self._count_human_rounds(compressed), 3)
        mock_summary.assert_not_called()

    def test_summary_generated_when_rounds_reach_trigger(self):
        source = self._build_round_messages(rounds=8)
        with patch("memory.history_store._summarize_messages", return_value="这是摘要") as mock_summary:
            compressed = hs._compress_messages(source)

        self.assertTrue(any(hs._is_summary_message(m) for m in compressed))
        self.assertEqual(self._count_human_rounds(compressed), config.memory_keep_recent_rounds)
        mock_summary.assert_called_once()

    def test_token_budget_prunes_oldest_recent_round(self):
        config.memory_history_max_tokens = 180
        source = self._build_round_messages(rounds=10, chars_per_msg=120)

        with patch("memory.history_store._summarize_messages", return_value="短摘要"):
            compressed = hs._compress_messages(source)

        total_tokens = hs._estimate_messages_tokens(compressed)
        human_rounds = self._count_human_rounds(compressed)

        self.assertGreaterEqual(human_rounds, 1)
        self.assertLessEqual(human_rounds, config.memory_keep_recent_rounds)
        if total_tokens > config.memory_history_max_tokens:
            # 极端情况下允许剩余 1 轮仍超预算，这是当前实现的边界行为。
            self.assertEqual(human_rounds, 1)

    def test_summary_fallback_when_model_fails(self):
        source = self._build_round_messages(rounds=2, chars_per_msg=30)

        class BrokenModel:
            def invoke(self, _):
                raise RuntimeError("mock failure")

        with patch("memory.history_store._build_light_model", return_value=BrokenModel()):
            summary = hs._summarize_messages(source, "旧摘要")

        self.assertTrue(len(summary) > 0)
        self.assertLessEqual(len(summary), config.memory_summary_max_chars)

    def test_summary_tag_recognition(self):
        yes_msg = SystemMessage(content=f"{config.memory_summary_tag}\nabc")
        no_msg_system = SystemMessage(content="普通系统消息")
        no_msg_human = HumanMessage(content=f"{config.memory_summary_tag} 但这不是系统消息")

        self.assertTrue(hs._is_summary_message(yes_msg))
        self.assertFalse(hs._is_summary_message(no_msg_system))
        self.assertFalse(hs._is_summary_message(no_msg_human))

    def test_idempotent_compression_result(self):
        source = self._build_round_messages(rounds=9, chars_per_msg=25)

        with patch("memory.history_store._summarize_messages", return_value="稳定摘要"):
            once = hs._compress_messages(source)
            twice = hs._compress_messages(once)

        once_summary_count = sum(1 for m in once if hs._is_summary_message(m))
        twice_summary_count = sum(1 for m in twice if hs._is_summary_message(m))
        once_rounds = self._count_human_rounds(once)
        twice_rounds = self._count_human_rounds(twice)

        self.assertEqual(once_summary_count, 1)
        self.assertEqual(twice_summary_count, 1)
        self.assertEqual(once_rounds, twice_rounds)

        # 语义幂等：再次压缩后体积不应明显膨胀。
        once_tokens = hs._estimate_messages_tokens(once)
        twice_tokens = hs._estimate_messages_tokens(twice)
        self.assertLessEqual(twice_tokens, once_tokens + 20)


if __name__ == "__main__":
    unittest.main()
