"""MinerU v4 精准解析 API 客户端。"""

from __future__ import annotations

import asyncio
import io
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from ingestion.parsers.mineru_contracts import (
    MinerUBatchUploadRequest,
    MinerUExtractResult,
    MinerUSettings,
)
from ingestion.parsers.routing import ParseBatch


class MinerUApiError(RuntimeError):
    pass


class MinerUClient:
    def __init__(self, settings: MinerUSettings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self._session = session or requests.Session()

    async def submit_batch(self, batch: ParseBatch) -> tuple[str, list[str]]:
        request = MinerUBatchUploadRequest(
            files=[
                {
                    "name": document.source_path.name,
                    "data_id": document.document_id,
                    "is_ocr": document.document_format.value == "image",
                }
                for document in batch.documents
            ],
            model_version=batch.model_version or self.settings.default_model,
            language=self.settings.language,
            enable_table=self.settings.enable_table,
            enable_formula=self.settings.enable_formula,
        )
        payload = await asyncio.to_thread(
            self._request_json,
            "POST",
            "/file-urls/batch",
            json=request.api_payload(),
            timeout=self.settings.request_timeout_seconds,
        )
        data = _api_data(payload)
        batch_id = str(data.get("batch_id") or "")
        file_urls = [str(value) for value in data.get("file_urls", [])]
        if not batch_id or len(file_urls) != len(batch.documents):
            raise MinerUApiError("MinerU 未返回有效 batch_id 或签名上传 URL 数量不匹配")
        return batch_id, file_urls

    async def upload_batch(self, batch: ParseBatch, file_urls: list[str]) -> None:
        async def upload(path: Path, url: str) -> None:
            await asyncio.to_thread(self._upload_file, path, url)

        await asyncio.gather(
            *(upload(document.source_path, url) for document, url in zip(batch.documents, file_urls, strict=True))
        )

    async def wait_for_batch(self, batch_id: str) -> list[MinerUExtractResult]:
        deadline = time.monotonic() + self.settings.poll_timeout_seconds
        while time.monotonic() < deadline:
            payload = await asyncio.to_thread(
                self._request_json,
                "GET",
                f"/extract-results/batch/{batch_id}",
                timeout=self.settings.request_timeout_seconds,
            )
            data = _api_data(payload)
            raw_results = data.get("extract_result") or data.get("extract_results") or data.get("results") or []
            results = [MinerUExtractResult.model_validate(item) for item in raw_results]
            if results and all(result.state in {"done", "failed"} for result in results):
                return results
            await asyncio.sleep(self.settings.poll_interval_seconds)
        raise TimeoutError(f"MinerU 批次 {batch_id} 在限定时间内未完成")

    async def download_and_extract(self, result: MinerUExtractResult, destination: Path) -> Path:
        if result.state != "done" or result.full_zip_url is None:
            raise MinerUApiError(f"文件 {result.file_name} 未成功解析: {result.err_msg or result.state}")
        response = await asyncio.to_thread(
            self._session.get,
            str(result.full_zip_url),
            timeout=self.settings.upload_timeout_seconds,
        )
        response.raise_for_status()
        destination.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(response.content, destination)
        return destination

    def _upload_file(self, path: Path, url: str) -> None:
        with path.open("rb") as stream:
            response = self._session.put(
                url,
                data=stream,
                timeout=self.settings.upload_timeout_seconds,
                headers={"Content-Type": "application/octet-stream"},
            )
        response.raise_for_status()

    def _request_json(self, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{str(self.settings.api_base_url).rstrip('/')}{endpoint}"
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self.settings.api_token.get_secret_value()}"
        headers.setdefault("Content-Type", "application/json")
        response = self._session.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        payload = response.json()
        code = payload.get("code", 0)
        if code not in {0, "0", None}:
            raise MinerUApiError(f"MinerU API 错误 {code}: {payload.get('msg') or payload.get('message')}")
        return payload


def _api_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise MinerUApiError("MinerU API 返回的 data 不是对象")
    return data


def _safe_extract_zip(content: bytes, destination: Path, max_uncompressed_bytes: int = 2_000_000_000) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        infos = archive.infolist()
        if len(infos) > 20_000:
            raise MinerUApiError("MinerU ZIP 文件条目过多")
        if sum(info.file_size for info in infos) > max_uncompressed_bytes:
            raise MinerUApiError("MinerU ZIP 解压后体积超过安全限制")
        for info in infos:
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise MinerUApiError(f"MinerU ZIP 包含不安全路径: {info.filename}")
            mode = info.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise MinerUApiError(f"MinerU ZIP 包含符号链接: {info.filename}")
            target = (root / Path(*pure.parts)).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise MinerUApiError(f"MinerU ZIP 路径越界: {info.filename}") from exc
        archive.extractall(root)


__all__ = ["MinerUApiError", "MinerUClient"]
