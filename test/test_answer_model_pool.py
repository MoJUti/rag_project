"""回答模型池的切换边界、接口协议和配置回归测试。"""

import json
import unittest
from unittest.mock import patch

import requests
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import Runnable

from core import config
from generation.model import ModelRequestError, build_chat_model
from generation.model_pool import AnswerModelPool, CompatibleChatModel
from scripts.run_node7_answer_smoke import UsageRecorder


class StubModel(Runnable):
    def __init__(self, events: list) -> None:
        self.events = events
        self.calls = 0

    def invoke(self, input, config=None, **kwargs):
        return list(self.stream(input, config, **kwargs))

    def stream(self, input, config=None, **kwargs):
        self.calls += 1
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield AIMessageChunk(content=event)


class AnswerModelPoolTest(unittest.TestCase):
    def pool(self, primary: StubModel, backup: StubModel) -> AnswerModelPool:
        return AnswerModelPool([("qwen", lambda: primary), ("zhipu", lambda: backup)])

    def test_normal_answer_and_refusal_do_not_use_backup(self) -> None:
        for answer in ("答案[S1]", "资料不足，无法回答"):
            primary, backup = StubModel([answer]), StubModel(["备用"])
            self.assertEqual(self.pool(primary, backup).invoke("问题").content, answer)
            self.assertEqual(backup.calls, 0)

    def test_transient_failure_before_content_uses_backup(self) -> None:
        primary = StubModel(["", ModelRequestError("限流", retryable=True)])
        backup = StubModel(["备用", "答案"])
        with self.assertLogs("generation.model_pool", level="WARNING"):
            self.assertEqual(self.pool(primary, backup).invoke("问题").content, "备用答案")
        self.assertEqual(backup.calls, 1)

    def test_no_switch_after_first_content(self) -> None:
        primary = StubModel(["半截", ModelRequestError("断线", retryable=True)])
        backup = StubModel(["备用"])
        stream = self.pool(primary, backup).stream("问题")
        self.assertEqual(next(stream).content, "半截")
        with self.assertRaisesRegex(ModelRequestError, "qwen.*断线"):
            list(stream)
        self.assertEqual(backup.calls, 0)

    def test_auth_and_empty_answer_do_not_switch(self) -> None:
        for events in ([ModelRequestError("HTTP 401")], [""], []):
            primary, backup = StubModel(events), StubModel(["备用"])
            with self.assertRaises(ModelRequestError):
                self.pool(primary, backup).invoke("问题")
            self.assertEqual(backup.calls, 0)

    def test_all_members_fail_without_looping(self) -> None:
        primary = StubModel([ModelRequestError("超时", True)])
        backup = StubModel([ModelRequestError("服务故障", True)])
        with self.assertLogs("generation.model_pool", level="WARNING"):
            with self.assertRaisesRegex(ModelRequestError, "zhipu.*服务故障"):
                self.pool(primary, backup).invoke("问题")
        self.assertEqual((primary.calls, backup.calls), (1, 1))

    def test_next_request_still_prefers_qwen(self) -> None:
        primary, backup = StubModel([ModelRequestError("超时", True)]), StubModel(["备用"])
        pool = self.pool(primary, backup)
        with self.assertLogs("generation.model_pool", level="WARNING"):
            pool.invoke("问题")
        primary.events = ["千问恢复"]
        self.assertEqual(pool.invoke("新问题").content, "千问恢复")
        self.assertEqual(backup.calls, 1)

    def test_factory_keeps_qwen_default_and_validates_backup(self) -> None:
        with patch.object(config, "chat_fallback_provider", "none"):
            self.assertEqual([name for name, _ in build_chat_model().members], ["qwen"])
        with patch.object(config, "chat_fallback_provider", "zhipu"), patch.dict(
            "os.environ", {"ZHIPU_API_KEY": ""}
        ):
            with self.assertRaisesRegex(ValueError, "未配置完整"):
                build_chat_model()

    def test_compatible_stream_parameters_and_resource_cleanup(self) -> None:
        for provider in ("qwen", "zhipu"):
            model = CompatibleChatModel("test-model", "https://example.com/v1", "secret", provider)
            event = {"choices": [{"delta": {"content": "答案[S1]"}, "finish_reason": "stop"}]}
            with patch("generation.model_pool.requests.Session") as sessions:
                session = sessions.return_value.__enter__.return_value
                response = session.post.return_value.__enter__.return_value
                response.iter_lines.return_value = ["data: " + json.dumps(event), "data: [DONE]"]
                answer = model.invoke([HumanMessage(content="问题")], max_tokens=100)
            self.assertEqual(answer.content, "答案[S1]")
            body = session.post.call_args.kwargs["json"]
            if provider == "qwen":
                self.assertIs(body["enable_thinking"], False)
                self.assertNotIn("thinking", body)
            else:
                self.assertEqual(body["thinking"], {"type": "enabled"})
                self.assertEqual(body["reasoning_effort"], "low")
                self.assertNotIn("enable_thinking", body)
            self.assertEqual(body["max_tokens"], 100)
            self.assertFalse(session.trust_env)
            self.assertEqual(answer.response_metadata["provider"], provider)
            sessions.return_value.__exit__.assert_called_once()

    def test_http_status_controls_retryability_without_leaking_secret(self) -> None:
        model = CompatibleChatModel("test", "https://example.com/v1", "secret")
        for status, retryable in ((401, False), (400, False), (404, False), (429, True), (503, True)):
            with patch("generation.model_pool.requests.Session") as sessions:
                response = sessions.return_value.__enter__.return_value.post.return_value.__enter__.return_value
                actual_response = requests.Response()
                actual_response.status_code = status
                response.raise_for_status.side_effect = requests.HTTPError("secret", response=actual_response)
                with self.assertRaises(ModelRequestError) as error:
                    model.invoke([HumanMessage(content="问题")])
            self.assertEqual(error.exception.retryable, retryable)
            self.assertNotIn("secret", str(error.exception))

    def test_broken_stream_and_invalid_json_are_not_success(self) -> None:
        model = CompatibleChatModel("test", "https://example.com/v1", "secret")
        for lines in ([], ["data: broken"]):
            with patch("generation.model_pool.requests.Session") as sessions:
                response = sessions.return_value.__enter__.return_value.post.return_value.__enter__.return_value
                response.iter_lines.return_value = lines
                with self.assertRaises(ModelRequestError):
                    model.invoke([HumanMessage(content="问题")])

    def test_callback_keeps_final_answer_model_and_usage(self) -> None:
        model = CompatibleChatModel("test", "https://example.com/v1", "secret", "qwen")
        events = [
            {"id": "request-1", "choices": [{"delta": {"content": "答案"}}]},
            {"id": "request-1", "choices": [{"delta": {"content": "[S1]"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        ]
        recorder = UsageRecorder()
        with patch("generation.model_pool.requests.Session") as sessions:
            response = sessions.return_value.__enter__.return_value.post.return_value.__enter__.return_value
            response.iter_lines.return_value = ["data: " + json.dumps(event) for event in events] + ["data: [DONE]"]
            result = list(AnswerModelPool([("qwen", lambda: model)]).stream(
                "问题", {"callbacks": [recorder]}))
        self.assertEqual("".join(chunk.content for chunk in result), "答案[S1]")
        self.assertEqual(recorder.body, "答案[S1]")
        self.assertEqual(recorder.info["provider"], "qwen")
        self.assertEqual(recorder.info["token_usage"]["total_tokens"], 15)

    def test_truncated_or_filtered_generation_is_not_success(self) -> None:
        model = CompatibleChatModel("test", "https://example.com/v1", "secret", "qwen")
        for reason in ("length", "content_filter"):
            event = {"choices": [{"delta": {"content": "半截答案"}, "finish_reason": reason}]}
            with patch("generation.model_pool.requests.Session") as sessions:
                response = sessions.return_value.__enter__.return_value.post.return_value.__enter__.return_value
                response.iter_lines.return_value = ["data: " + json.dumps(event), "data: [DONE]"]
                with self.assertRaisesRegex(ModelRequestError, "未完整生成"):
                    list(model.stream("问题"))


if __name__ == "__main__":
    unittest.main()
