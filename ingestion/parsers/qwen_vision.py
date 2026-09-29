"""千问 3.8 异常页复核客户端；文档提取强制关闭思考。"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, field_validator


class QwenVisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: HttpUrl = HttpUrl("https://maas.qianwenaiapi.com/compatible-mode/v1")
    api_key: SecretStr
    model: str = "qwen3.8-max"
    enable_thinking: bool = False
    timeout_seconds: float = Field(default=300.0, gt=0)
    max_images: int = Field(default=8, ge=1, le=32)

    @field_validator("api_key")
    @classmethod
    def key_must_not_be_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("DASHSCOPE_API_KEY 未配置")
        return value

    @field_validator("enable_thinking")
    @classmethod
    def thinking_must_be_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("文档结构化提取禁止开启千问思考模式")
        return value

    @classmethod
    def from_env(cls) -> "QwenVisionSettings":
        return cls(
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://maas.qianwenaiapi.com/compatible-mode/v1"),
            api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY", "")),
            model=os.getenv("QWEN_VL_MODEL", "qwen3.8-max"),
            enable_thinking=_env_bool("QWEN_VL_ENABLE_THINKING", False),
        )


class QwenVisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    content: dict[str, Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    raw_text: str


class QwenVisionClient:
    def __init__(self, settings: QwenVisionSettings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self._session = session or requests.Session()

    def extract(self, images: list[Path], prompt: str) -> QwenVisionResult:
        if not images or len(images) > self.settings.max_images:
            raise ValueError(f"图片数量必须在 1 到 {self.settings.max_images} 之间")
        content: list[dict[str, Any]] = [_image_part(path) for path in images]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "enable_thinking": False,
        }
        response = self._session.post(
            f"{str(self.settings.base_url).rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        raw_text = str(data["choices"][0]["message"].get("content", ""))
        parsed = _parse_json_object(raw_text)
        return QwenVisionResult(
            model=str(data.get("model") or self.settings.model),
            content=parsed,
            usage=data.get("usage") or {},
            raw_text=raw_text,
        )


def _image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.S | re.I)
    if fenced:
        stripped = fenced.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("千问返回内容不是有效 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("千问返回 JSON 必须是对象")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是布尔值")


__all__ = ["QwenVisionClient", "QwenVisionResult", "QwenVisionSettings"]
