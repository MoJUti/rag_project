"""逐请求可观测性：不重复执行、异常保留、元数据隔离及界面无付费请求验证。"""

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core import config
from generation.observability import RequestTraceRecorder, settings_snapshot, stream_with_trace
from generation.rag_service import RagService
from test.test_grounded_answers import FakeVectorService, make_document


class ObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        for key, value in {"evidence_expansion": "none", "query_resolution_strategy": "original", "retrieval_strategy": "dense"}.items():
            item = patch.object(config, key, value)
            item.start()
            self.addCleanup(item.stop)

    def service(self) -> RagService:
        archive = InMemoryChatMessageHistory()
        return RagService(vector_service=FakeVectorService([make_document()]),
            chat_model=FakeListChatModel(responses=["预计财务费用为47百万元。[S1]"]), history_factory=lambda _: archive)

    def test_stream_unchanged_and_retrieval_only_once(self) -> None:
        service = self.service()
        spy = Mock(wraps=service.retrieve_with_trace)
        service.retrieve_with_trace = spy
        report, run_config = {}, config.build_session_config("trace_a")
        text = "".join(stream_with_trace(service, "预计财务费用？", run_config, report))
        self.assertIn("47百万元。[S1]", text)
        self.assertEqual(text.count("报告.md"), 1)
        spy.assert_called_once_with("预计财务费用？")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["prepared"]["evidence"]), 1)
        self.assertEqual(report["prepared"]["stages"][0]["stage"], "history_query")
        self.assertEqual(report["prepared"]["history"]["message_count"], 0)
        self.assertEqual(report["model_calls"][0]["phase"], "answer")
        self.assertIsNotNone(report["first_text_seconds"])
        self.assertNotIn("callbacks", run_config)

    def test_partial_failure_is_saved_and_original_error_propagates(self) -> None:
        def chunks(*args, **kwargs):
            yield "部分正文"
            raise RuntimeError("测试中断")
        service = SimpleNamespace(chain=SimpleNamespace(stream=chunks))
        report = {}
        with self.assertRaisesRegex(RuntimeError, "测试中断"):
            list(stream_with_trace(service, "问题", {}, report))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["output_characters"], 4)
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertIsNone(report["prepared"])

    def test_close_stream_is_interruption_not_success(self) -> None:
        service = SimpleNamespace(chain=SimpleNamespace(stream=lambda *a, **k: iter(["甲", "乙"])))
        report = {}
        stream = stream_with_trace(service, "问题", {}, report)
        next(stream)
        stream.close()
        self.assertEqual(report["status"], "interrupted")

    def test_model_metadata_whitelist_and_per_request_isolation(self) -> None:
        first, second = RequestTraceRecorder(), RequestTraceRecorder()
        payload = {"evidence": [{"text": "资料甲"}]}
        first.on_custom_event("rag_prepared", payload)
        payload["evidence"][0]["text"] = "被修改"
        self.assertEqual(first.prepared["evidence"][0]["text"], "资料甲")
        self.assertIsNone(second.prepared)
        run_id = uuid4()
        generation = SimpleNamespace(message=None, generation_info={
            "model_name": "测试模型", "authorization": "敏感值不保存", "token_usage": {"prompt_tokens": 12}})
        first.on_llm_end(SimpleNamespace(generations=[[generation]]), run_id=run_id)
        self.assertNotIn("authorization", first.model_calls[str(run_id)])
        self.assertEqual(first.model_calls[str(run_id)]["usage"]["prompt_tokens"], 12)

    def test_settings_do_not_include_keys(self) -> None:
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "secret-for-test"}):
            snapshot = settings_snapshot()
        self.assertNotIn("API_KEY", str(snapshot))
        self.assertNotIn("secret-for-test", str(snapshot))

    def test_prepared_evidence_survives_generation_failure(self) -> None:
        from langchain_core.runnables import RunnableLambda

        def fail(value: object) -> str:
            raise RuntimeError("生成失败")

        service = self.service()
        service.chat_model = RunnableLambda(fail)
        service.chain = service._build_chain()
        report = {}
        with self.assertRaisesRegex(RuntimeError, "生成失败"):
            list(stream_with_trace(service, "问题", config.build_session_config("failure"), report))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["prepared"]["evidence"]), 1)
        self.assertIsNone(report["first_text_seconds"])

    def test_clarification_report_contains_no_answer_model_call(self) -> None:
        from retrieval.query_resolution import QueryResolution
        service = self.service()
        service.query_resolver = Mock()
        service.query_resolver.resolve.return_value = QueryResolution("它是谁？", "", "clarify", "请明确公司。")
        service.retrieve_with_trace = Mock(side_effect=AssertionError("不应检索"))
        report = {}
        text = "".join(stream_with_trace(service, "它是谁？", config.build_session_config("clarify"), report))
        self.assertEqual(text, "请明确公司。")
        self.assertEqual(report["model_calls"], [])
        self.assertEqual(report["prepared"]["query_resolution"]["action"], "clarify")
        self.assertEqual(report["prepared"]["evidence"], [])


class TracePanelAppTest(unittest.TestCase):
    def setUp(self) -> None:
        # AppTest自身也会创建临时目录，限定到工作区以避免Windows沙箱清理权限问题。
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        item = patch.object(tempfile, "tempdir", str(root))
        item.start()
        self.addCleanup(item.stop)

    def test_chat_page_shows_trace_and_rerun_does_not_repeat_request(self) -> None:
        from streamlit.testing.v1 import AppTest
        archive = InMemoryChatMessageHistory()
        with patch.multiple(config, evidence_expansion="none", query_resolution_strategy="original", retrieval_strategy="dense"):
            service = RagService(vector_service=FakeVectorService([make_document()]),
                chat_model=FakeListChatModel(responses=["预计财务费用为47百万元。[S1]"]), history_factory=lambda _: archive)
        spy = Mock(wraps=service.retrieve_with_trace)
        service.retrieve_with_trace = spy
        path = Path(__file__).resolve().parents[1] / "app/streamlit_chat.py"
        with patch("generation.rag_service.RagService", return_value=service), \
             patch("memory.history_store.list_session_ids", return_value=["ui_test"]), \
             patch("memory.history_store.get_history", return_value=InMemoryChatMessageHistory()), \
             patch("memory.history_store.is_session_pinned", return_value=False), \
             patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-placeholder"}):
            page = AppTest.from_file(str(path), default_timeout=30).run()
            self.assertEqual(len(page.exception), 0)
            page.chat_input[0].set_value("预计财务费用？").run()
            self.assertEqual(len(page.exception), 0)
            self.assertEqual(page.session_state["request_traces"]["ui_test"]["status"], "completed")
            self.assertTrue(any(e.label == "最近一次问答流程" for e in page.expander))
            self.assertTrue(any(m.label == "本次总耗时" for m in page.metric))
            spy.assert_called_once()
            page.run()
            spy.assert_called_once()
