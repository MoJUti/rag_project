"""新Web后端的真实路由、SSE、会话隔离和异常测试，模型及向量库使用替身。"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from api.main import SessionStore, create_app
from core import config
from generation.rag_service import RagService
from test.test_grounded_answers import FakeVectorService, make_document


def read_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.split("\n\n"):
        lines = frame.splitlines()
        if lines and lines[0].startswith("event: "):
            events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)
        self.providers = []
        self.model = FakeListChatModel(responses=["财务费用为47百万元。[S1]"])
        for key, value in {"retrieval_strategy": "dense", "evidence_expansion": "none", "query_resolution_strategy": "original", "api_chat_provider": "zhipu"}.items():
            item = patch.object(config, key, value)
            item.start()
            self.addCleanup(item.stop)

        def factory(provider, history_factory):
            self.providers.append(provider)
            return RagService(vector_service=FakeVectorService([make_document()]), chat_model=self.model, history_factory=history_factory)

        self.client = TestClient(create_app(factory, SessionStore(self.directory.name)), base_url="http://localhost")
        self.addCleanup(self.client.close)

    def create(self) -> str:
        response = self.client.post("/api/sessions")
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def test_health_and_configuration_do_not_call_model(self) -> None:
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        configuration = self.client.get("/api/config").json()
        self.assertEqual(configuration["default_provider"], "zhipu")
        self.assertEqual(self.providers, [])
        self.assertNotIn("API_KEY", json.dumps(configuration))

    def test_stream_history_and_provider_selection(self) -> None:
        session = self.create()
        events = read_events(self.client.post(f"/api/sessions/{session}/chat", json={"query": "费用多少？"}).text)
        self.assertEqual(events[0][0], "start")
        self.assertIn("prepared", [name for name, _ in events])
        self.assertEqual(events[-1][0], "done")
        text = "".join(data["text"] for name, data in events if name == "token")
        self.assertIn("47百万元。[S1]", text)
        messages = self.client.get(f"/api/sessions/{session}/messages").json()
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["content"], text)
        self.assertEqual(self.providers, ["zhipu"])
        second = self.create()
        self.client.post(f"/api/sessions/{second}/chat", json={"query": "费用多少？", "provider": "qwen"})
        self.assertEqual(self.providers, ["zhipu", "qwen"])
        self.assertEqual(len(self.client.get(f"/api/sessions/{session}/messages").json()), 2)

    def test_invalid_input_unknown_session_and_external_origin(self) -> None:
        self.assertEqual(self.client.get("/api/sessions/bad!/messages").status_code, 422)
        self.assertEqual(self.client.get("/api/sessions/chat_missing/messages").status_code, 404)
        session = self.create()
        self.assertEqual(self.client.post(f"/api/sessions/{session}/chat", json={"query": " "}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/sessions/{session}/chat", json={"query": "问题", "provider": "other"}).status_code, 422)
        self.assertEqual(self.client.get("/api/config", headers={"Origin": "https://evil.example"}).status_code, 403)

    def test_generation_error_is_terminal_event_not_success(self) -> None:
        def fail(_):
            raise RuntimeError("测试模型连接失败")
        self.model = RunnableLambda(fail)
        session = self.create()
        events = read_events(self.client.post(f"/api/sessions/{session}/chat", json={"query": "费用？"}).text)
        self.assertEqual(events[-1][0], "error")
        self.assertIn("测试模型连接失败", events[-1][1]["message"])
        self.assertFalse(any(name == "done" for name, _ in events))
        self.assertEqual(self.client.get(f"/api/sessions/{session}/messages").json(), [])

    def test_same_session_rejects_overlapping_requests(self) -> None:
        entered, release = Event(), Event()

        def wait(_):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("测试释放等待超时")
            return "47百万元。[S1]"

        self.model = RunnableLambda(wait)
        session = self.create()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.client.post, f"/api/sessions/{session}/chat", json={"query": "费用？"})
            try:
                self.assertTrue(entered.wait(8))
                response = self.client.post(f"/api/sessions/{session}/chat", json={"query": "重复问题"})
                self.assertEqual(response.status_code, 409)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=10).status_code, 200)

    def test_initialization_failure_has_safe_trace_and_redacts_key(self) -> None:
        def fail_factory(*args):
            raise ValueError("连接失败 secret-local-api-key")

        with patch.dict("os.environ", {"ZHIPU_API_KEY": "secret-local-api-key"}):
            client = TestClient(create_app(fail_factory, SessionStore(self.directory.name)), base_url="http://localhost")
            with client:
                session = client.post("/api/sessions").json()["id"]
                response = client.post(f"/api/sessions/{session}/chat", json={"query": "问题"})
                events = read_events(response.text)
        error = events[-1][1]
        self.assertNotIn("secret-local-api-key", response.text)
        self.assertEqual(error["trace"]["model_calls"], [])
        self.assertEqual(error["trace"]["status"], "failed")

    def test_malformed_origin_is_rejected(self) -> None:
        self.assertEqual(self.client.get("/api/config", headers={"Origin": "http://["}).status_code, 403)
