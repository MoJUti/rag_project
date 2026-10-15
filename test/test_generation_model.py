"""回答端平台地址、关闭思考和传输会话生命周期的回归测试。"""

import unittest
from unittest.mock import patch
import requests

from langchain_core.messages import HumanMessage

from core import config
from generation.model import build_qwen_model as build_chat_model


def response(content: str, finish_reason: str = "stop") -> dict:
    return {"output": {"choices": [{"message": {"role": "assistant", "content": content},
                                      "finish_reason": finish_reason}]},
            "request_id": "test-request", "usage": {"input_tokens": 10, "output_tokens": 5}}


class GenerationModelTest(unittest.TestCase):
    def setUp(self) -> None:
        key = patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"})
        key.start()
        self.addCleanup(key.stop)

    def test_configured_model_platform_and_non_thinking_are_forwarded(self) -> None:
        model = build_chat_model()
        self.assertEqual(model.model_name, config.chat_model_name)
        self.assertEqual(model.model_kwargs["base_address"], config.dashscope_http_api_base_url)
        self.assertFalse(model.model_kwargs["enable_thinking"])

    def test_invoke_uses_direct_session_and_closes_it(self) -> None:
        model = build_chat_model()
        model.streaming = False
        with patch("generation.model.requests.Session") as sessions:
            session = sessions.return_value.__enter__.return_value
            with patch.object(type(model), "completion_with_retry", return_value=response("47[S1]")) as request:
                result = model.invoke([HumanMessage(content="问题")])
        self.assertEqual(result.content, "47[S1]")
        self.assertIs(request.call_args.kwargs["session"], session)
        self.assertFalse(session.trust_env)
        sessions.return_value.__exit__.assert_called_once()

    def test_stream_yields_model_chunks_and_closes_session(self) -> None:
        model = build_chat_model()
        with patch("generation.model.requests.Session") as sessions:
            with patch.object(type(model), "stream_completion_with_retry", return_value=iter([
                response("预计", "null"), response("47[S1]"),
            ])):
                result = list(model.stream([HumanMessage(content="问题")]))
        self.assertEqual("".join(item.content for item in result), "预计47[S1]")
        sessions.return_value.__exit__.assert_called_once()

    def test_unavailable_model_has_clear_error_and_closes_session(self) -> None:
        model = build_chat_model()
        with patch("generation.model.requests.Session") as sessions:
            with patch.object(type(model), "stream_completion_with_retry",
                              side_effect=requests.HTTPError("status_code: 404 Model not exist")):
                with self.assertRaisesRegex(RuntimeError, "当前平台不可用.*404"):
                    list(model.stream([HumanMessage(content="问题")]))
        sessions.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
