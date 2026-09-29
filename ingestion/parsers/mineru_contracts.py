"""MinerU v4 精准解析 API 的配置和传输模型，不包含网络调用。"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, field_validator


MinerUModelVersion = Literal["pipeline", "vlm", "MinerU-HTML"]


class MinerUSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_base_url: HttpUrl = HttpUrl("https://mineru.net/api/v4")
    api_token: SecretStr
    default_model: Literal["pipeline", "vlm"] = "vlm"
    html_model: Literal["MinerU-HTML"] = "MinerU-HTML"
    language: str = "ch"
    enable_table: bool = True
    enable_formula: bool = True
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    upload_timeout_seconds: float = Field(default=300.0, gt=0)
    poll_interval_seconds: float = Field(default=3.0, gt=0)
    poll_timeout_seconds: float = Field(default=1800.0, gt=0)
    max_batch_size: int = Field(default=50, ge=1, le=50)

    @field_validator("api_token")
    @classmethod
    def token_must_not_be_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("MINERU_API_TOKEN 未配置")
        return value

    @classmethod
    def from_env(cls) -> "MinerUSettings":
        return cls(
            api_base_url=os.getenv("MINERU_API_BASE_URL", "https://mineru.net/api/v4"),
            api_token=SecretStr(os.getenv("MINERU_API_TOKEN", "")),
            default_model=os.getenv("MINERU_DEFAULT_MODEL", "vlm"),
            html_model=os.getenv("MINERU_HTML_MODEL", "MinerU-HTML"),
            language=os.getenv("MINERU_LANGUAGE", "ch"),
            enable_table=_env_bool("MINERU_ENABLE_TABLE", True),
            enable_formula=_env_bool("MINERU_ENABLE_FORMULA", True),
            request_timeout_seconds=float(os.getenv("MINERU_REQUEST_TIMEOUT_SECONDS", "30")),
            upload_timeout_seconds=float(os.getenv("MINERU_UPLOAD_TIMEOUT_SECONDS", "300")),
            poll_interval_seconds=float(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "3")),
            poll_timeout_seconds=float(os.getenv("MINERU_POLL_TIMEOUT_SECONDS", "1800")),
            max_batch_size=int(os.getenv("MINERU_MAX_BATCH_SIZE", "50")),
        )


class MinerUFileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    data_id: str
    is_ocr: bool = False
    page_ranges: str | None = None


class MinerUBatchUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[MinerUFileRequest] = Field(min_length=1, max_length=50)
    model_version: MinerUModelVersion
    language: str = "ch"
    enable_table: bool = True
    enable_formula: bool = True
    extra_formats: list[Literal["docx", "html", "latex"]] = Field(default_factory=list)

    def api_payload(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)


class MinerUBatchAccepted(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_id: str
    file_urls: list[HttpUrl]


class MinerUExtractResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_name: str
    state: Literal["waiting-file", "pending", "converting", "running", "done", "failed"]
    full_zip_url: HttpUrl | None = None
    err_msg: str = ""
    data_id: str | None = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false、1/0、yes/no 或 on/off")


__all__ = [
    "MinerUBatchAccepted",
    "MinerUBatchUploadRequest",
    "MinerUExtractResult",
    "MinerUFileRequest",
    "MinerUModelVersion",
    "MinerUSettings",
]
