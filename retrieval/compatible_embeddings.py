"""固定模型与维度的 OpenAI 兼容 Embedding 客户端。"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings


class CompatibleEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Embedding base_url 不能为空")
        if not api_key.strip():
            raise ValueError("Embedding api_key 不能为空")
        if dimensions < 1:
            raise ValueError("Embedding dimensions 必须为正整数")
        self.endpoint = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.client = http_client or httpx.Client(
            trust_env=False,
            timeout=httpx.Timeout(90.0, connect=30.0),
        )
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def embed(
        self,
        texts: list[str],
        max_attempts: int = 5,
    ) -> tuple[list[list[float]], int | None]:
        if not texts:
            return [], 0
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": texts,
                        "dimensions": self.dimensions,
                        "encoding_format": "float",
                    },
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(f"Embedding 服务暂时不可用: HTTP {response.status_code}")
                if response.status_code != 200:
                    payload = _safe_json(response)
                    message = (
                        (payload.get("error") or {}).get("message")
                        or payload.get("message")
                        or response.text[:300]
                    )
                    raise ValueError(
                        f"Embedding 请求失败: HTTP {response.status_code}: {message}"
                    )
                payload = response.json()
                data = sorted(payload.get("data", []), key=lambda row: row.get("index", 0))
                vectors = [row.get("embedding", []) for row in data]
                if len(vectors) != len(texts):
                    raise ValueError(
                        f"Embedding 返回数量异常: {len(vectors)}，预期 {len(texts)}"
                    )
                for vector in vectors:
                    if len(vector) != self.dimensions:
                        raise ValueError(
                            f"Embedding 返回维度异常: {len(vector)}，预期 {self.dimensions}"
                        )
                usage = payload.get("usage") or {}
                return vectors, usage.get("total_tokens") or usage.get("prompt_tokens")
            except ValueError:
                raise
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                time.sleep(min(20, 2 ** attempt))
        raise RuntimeError(f"Embedding 请求连续失败 {max_attempts} 次") from last_error


class CompatibleEmbeddings(Embeddings):
    """LangChain Embeddings 实现，显式传递 text-embedding-v4 的维度。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        batch_size: int = 10,
        client: CompatibleEmbeddingClient | None = None,
    ) -> None:
        if not 1 <= batch_size <= 10:
            raise ValueError("text-embedding-v4 同步接口 batch_size 必须在 1..10")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.client = client or CompatibleEmbeddingClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            dimensions=dimensions,
        )

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        batch_size: int,
    ) -> "CompatibleEmbeddings":
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        return cls(base_url, api_key, model, dimensions, batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch_vectors, _ = self.client.embed(texts[start:start + self.batch_size])
            vectors.extend(batch_vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors, _ = self.client.embed([text])
        return vectors[0]

    def close(self) -> None:
        self.client.close()


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}
