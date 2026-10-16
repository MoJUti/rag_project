"""真实逐轮落盘路径的存档完整性、摘要游标、隔离及失败回归。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_to_dict

from core import config
from memory import history_store as hs
from memory.file_history import ContextChatMessageHistory, FileChatMessageHistory
from memory.session_context import MemoryBudgetError, SummaryResult, estimate_tokens


def turn(number: int) -> list:
    return [HumanMessage(content=f"用户第{number}轮条件"), AIMessage(content=f"助手第{number}轮回答")]


class SessionMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        values = {"memory_keep_recent_rounds": 3, "memory_summary_trigger_rounds": 5,
                  "memory_compression_trigger": "rounds", "memory_summary_max_tokens": None,
                  "memory_history_max_tokens": 4000, "memory_summary_max_chars": 1500,
                  "memory_summary_enabled": True, "chat_history_directory": str(self.directory)}
        for name, value in values.items():
            item = patch.object(config, name, value)
            item.start()
            self.addCleanup(item.stop)
        self.summarize = Mock(return_value=SummaryResult("用户明确的条件，助手说法待核实", {"usage": {"total_tokens": 20}}))
        self.history = FileChatMessageHistory("chat_test", str(self.directory), self.summarize)

    def test_incremental_rounds_trigger_and_archive_is_never_compressed(self) -> None:
        for number in range(1, 8):
            self.history.add_messages(turn(number))
            context = self.history.context_messages()
            self.assertEqual(len(self.history.messages), number * 2)
            self.assertLessEqual(estimate_tokens(context), 4000)
            if number < 5:
                self.assertEqual(len(context), number * 2)
                self.summarize.assert_not_called()
        self.assertEqual(self.summarize.call_count, 3)
        self.assertEqual([len(call.args[0]) for call in self.summarize.call_args_list], [4, 2, 2])
        self.assertEqual(self.history.context_state["covered_messages"], 8)
        self.assertEqual(self.history.context_state["model_info"]["usage"]["total_tokens"], 20)

    def test_repeat_reads_and_restart_do_not_resummarize(self) -> None:
        for number in range(5):
            self.history.add_messages(turn(number))
        first = self.history.context_messages()
        self.history.context_messages()
        restored = FileChatMessageHistory("chat_test", str(self.directory), self.summarize)
        self.assertEqual(restored.context_messages(), first)
        self.summarize.assert_called_once()

    def test_legacy_is_read_without_overwrite_and_new_archive_contains_old_messages(self) -> None:
        legacy = self.directory / "chat_old"
        legacy.write_text(json.dumps([message_to_dict(m) for m in turn(1)]), encoding="utf-8")
        before = legacy.read_bytes()
        history = FileChatMessageHistory("chat_old", str(self.directory), self.summarize)
        self.assertEqual(len(history.messages), 2)
        history.add_messages(turn(2))
        self.assertEqual(legacy.read_bytes(), before)
        self.assertEqual(len(history.messages), 4)
        self.assertTrue(history.archive_path.exists())

    def test_failure_is_recorded_and_identical_read_does_not_repeat_paid_call(self) -> None:
        self.summarize.side_effect = RuntimeError("模拟接口拒绝")
        for number in range(5):
            self.history.add_messages(turn(number))
        context = self.history.context_messages()
        self.history.context_messages()
        self.summarize.assert_called_once()
        self.assertEqual(self.history.context_state["status"], "degraded")
        self.assertIn("RuntimeError", self.history.context_state["error"])
        self.assertEqual(self.history.context_state["covered_messages"], 0)
        self.assertFalse(any(message.type == "system" for message in context))
        self.assertEqual(len(self.history.messages), 10)
        self.history.context_messages(retry_failed=True)
        self.assertEqual(self.summarize.call_count, 2)

    def test_empty_or_overlong_summary_is_not_saved_as_success(self) -> None:
        for number in range(5):
            self.history.add_messages(turn(number))
        for text in ("", "x" * 1501):
            self.summarize.return_value = text
            self.history.context_messages(retry_failed=True)
            self.assertEqual(self.history.context_state["status"], "degraded")
            self.assertEqual(self.history.context_state["covered_messages"], 0)

    def test_sessions_are_isolated_and_model_wrapper_writes_full_archive(self) -> None:
        other = FileChatMessageHistory("chat_other", str(self.directory), self.summarize)
        self.history.add_messages(turn(1))
        view = ContextChatMessageHistory(other)
        view.add_messages([HumanMessage(content="另一会话"), AIMessage(content="独立回复")])
        self.assertEqual(len(view.messages), 2)
        self.assertNotIn("另一会话", str(self.history.messages))
        self.assertNotIn("用户第1轮", str(other.messages))

    def test_clear_and_delete_remove_summary_and_do_not_resurrect_legacy(self) -> None:
        legacy = self.directory / "chat_test"
        legacy.write_text(json.dumps([message_to_dict(m) for m in turn(1)]), encoding="utf-8")
        self.history.add_messages(turn(2))
        self.history.context_messages()
        self.history.clear()
        self.assertEqual(self.history.messages, [])
        self.assertEqual(self.history.context_state, {})
        self.assertFalse(legacy.exists())
        hs.delete_history("chat_test")
        self.assertFalse(self.history.archive_path.exists())
        self.assertNotIn("chat_test", hs.list_session_ids())

    def test_new_archives_appear_once_in_session_list(self) -> None:
        self.history.add_messages(turn(1))
        (self.directory / "chat_test").write_text("[]", encoding="utf-8")
        self.assertEqual(hs.list_session_ids(), ["chat_test"])

    def test_corrupt_history_is_reported_not_overwritten_as_empty(self) -> None:
        path = self.directory / "chat_bad"
        path.write_text("broken", encoding="utf-8")
        history = FileChatMessageHistory("chat_bad", str(self.directory), self.summarize)
        with self.assertRaisesRegex(RuntimeError, "历史读取失败"):
            history.add_messages(turn(1))
        self.assertEqual(path.read_text(encoding="utf-8"), "broken")
        self.assertFalse(history.archive_path.exists())

    def test_path_traversal_is_rejected(self) -> None:
        for session_id in ("../outside", "a/b", "a\\b", ".", ""):
            with self.assertRaises(ValueError):
                FileChatMessageHistory(session_id, str(self.directory))

    def test_oversized_latest_round_fails_without_model_and_preserves_archive(self) -> None:
        self.history.add_messages([HumanMessage(content="中文长输入" * 400), AIMessage(content="完整回复")])
        with patch.object(config, "memory_history_max_tokens", 100):
            with self.assertRaisesRegex(MemoryBudgetError, "最新一轮超过"):
                self.history.context_messages()
        self.summarize.assert_not_called()
        self.assertEqual(len(self.history.messages), 2)

    def test_special_token_text_remains_user_data(self) -> None:
        self.history.add_messages([HumanMessage(content="<|endoftext|>不是系统指令"), AIMessage(content="普通回答")])
        self.assertEqual(len(self.history.context_messages()), 2)

    def test_atomic_write_failure_preserves_previous_archive(self) -> None:
        self.history.add_messages(turn(1))
        before = self.history.archive_path.read_bytes()
        with patch("memory.file_history.os.replace", side_effect=OSError("模拟替换失败")):
            with self.assertRaisesRegex(RuntimeError, "会话状态写入失败"):
                self.history.add_messages(turn(2))
        self.assertEqual(self.history.archive_path.read_bytes(), before)
        self.assertEqual(len(self.history.messages), 2)

    def test_changed_summary_model_rebuilds_from_original_history(self) -> None:
        for number in range(5):
            self.history.add_messages(turn(number))
        self.history.context_messages()
        with patch.object(config, "zhipu_chat_model", "changed-for-test"):
            self.history.context_messages()
        self.assertEqual(self.summarize.call_count, 2)
        self.assertEqual(len(self.summarize.call_args.args[0]), 4)

    def test_modified_covered_history_is_rejected(self) -> None:
        for number in range(5):
            self.history.add_messages(turn(number))
        self.history.context_messages()
        state = json.loads(self.history.archive_path.read_text(encoding="utf-8"))
        state["messages"][0]["data"]["content"] = "被篡改的早期条件"
        self.history.archive_path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "原始消息指纹不一致"):
            self.history.context_messages()

    def test_disabled_summary_uses_window_without_deleting_archive(self) -> None:
        for number in range(7):
            self.history.add_messages(turn(number))
        with patch.object(config, "memory_summary_enabled", False):
            self.assertEqual(len(self.history.context_messages()), 6)
        self.summarize.assert_not_called()
        self.assertEqual(len(self.history.messages), 14)

    def test_summary_glm_factory_uses_configured_provider_and_low_adapter(self) -> None:
        with patch.dict("os.environ", {"ZHIPU_API_KEY": "test-key"}):
            model = hs._build_light_model()
        self.assertEqual(model.provider, "zhipu")
        self.assertEqual(model.model, config.zhipu_chat_model)
        self.assertEqual(model.base_url, config.zhipu_base_url.rstrip("/"))

    def test_real_rag_writes_full_history_but_reads_summary_context(self) -> None:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from generation.rag_service import RagService
        from test.test_grounded_answers import FakeVectorService, make_document

        with patch.object(config, "retrieval_strategy", "dense"), patch.object(config, "evidence_expansion", "none"):
            with patch("memory.history_store._summarize_messages", self.summarize):
                rag = RagService(FakeVectorService([make_document()]), FakeListChatModel(responses=["已答。[S1]"] * 7))
                for number in range(7):
                    "".join(rag.chain.stream({"input": f"第{number}次提问"}, config.build_session_config("chat_integration")))
        history = hs.get_history("chat_integration")
        self.assertEqual(len(history.messages), 14)
        self.assertEqual(history.context_state["status"], "ready")
        self.assertEqual(self.summarize.call_count, 2)


if __name__ == "__main__":
    unittest.main()
