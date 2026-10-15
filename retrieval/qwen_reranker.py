"""千问文本重排适配器；不改变现有召回默认路径。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


DEFAULT_RERANK_URL = (
    "https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank"
)


@dataclass(frozen=True)
class RerankResult:
    """按相关性排序的原输入下标和本次调用消耗。"""

    indices: list[int]
    total_tokens: int
    request_id: str


class QwenTextReranker:
    """调用文本排序接口并严格校验返回的候选下标。"""

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        model: str = "qwen3.7-text-rerank",
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，无法调用重排模型")
        self.url = url or os.getenv("RERANK_API_URL", DEFAULT_RERANK_URL).strip()
        self.model = model
        # 与嵌入客户端保持一致，避免继承系统代理后在TLS握手阶段失败。
        self.client = client or httpx.Client(trust_env=False, timeout=httpx.Timeout(90.0, connect=20.0))
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> QwenTextReranker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def rerank(self, query: str, documents: list[str]) -> RerankResult:
        if not query.strip() or not documents or any(not item.strip() for item in documents):
            raise ValueError("重排查询与候选文档均不能为空")
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": len(documents)},
        }
        try:
            response = self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"千问重排连接失败：{type(exc).__name__}") from exc
        if response.is_error:
            try:
                error_code = str(response.json().get("code", "unknown"))
            except (ValueError, AttributeError):
                error_code = "unknown"
            raise RuntimeError(f"千问重排请求失败：HTTP {response.status_code}，错误码 {error_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("千问重排响应不是有效JSON") from exc
        try:
            results = body["output"]["results"]
            indices = [int(item["index"]) for item in results]
            total_tokens = int(body["usage"]["total_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("千问重排响应缺少排序结果或用量") from exc
        if len(indices) != len(documents) or set(indices) != set(range(len(documents))):
            raise RuntimeError("千问重排返回的候选下标不完整或有重复")
        return RerankResult(indices, total_tokens, str(body.get("request_id", "")))
