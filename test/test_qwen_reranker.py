"""重排接口的响应、异常与输入校验。"""

import unittest

import httpx

from retrieval.qwen_reranker import QwenTextReranker


class QwenRerankerTest(unittest.TestCase):
    def test_rerank_preserves_original_indices_and_usage(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            self.assertEqual(request.read().count(b"top_n"), 1)
            return httpx.Response(200, json={
                "output": {"results": [{"index": 1}, {"index": 0}]},
                "usage": {"total_tokens": 123}, "request_id": "req-1",
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            service = QwenTextReranker(api_key="test-key", client=client)
            result = service.rerank("问题", ["文档甲", "文档乙"])
        self.assertEqual(result.indices, [1, 0])
        self.assertEqual(result.total_tokens, 123)

    def test_incomplete_indices_fail_clearly(self) -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
            "output": {"results": [{"index": 0}, {"index": 0}]},
            "usage": {"total_tokens": 12},
        }))
        with httpx.Client(transport=transport) as client:
            service = QwenTextReranker(api_key="test-key", client=client)
            with self.assertRaisesRegex(RuntimeError, "下标不完整或有重复"):
                service.rerank("问题", ["文档甲", "文档乙"])

    def test_http_error_reports_status_and_code(self) -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(401, json={"code": "InvalidApiKey"}))
        with httpx.Client(transport=transport) as client:
            service = QwenTextReranker(api_key="test-key", client=client)
            with self.assertRaisesRegex(RuntimeError, "HTTP 401，错误码 InvalidApiKey"):
                service.rerank("问题", ["文档甲"])


if __name__ == "__main__":
    unittest.main()
