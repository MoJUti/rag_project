"""沿用千问模型，通过明确的平台地址调用现有DashScope适配器。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator
import re

import requests
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from core import config

if TYPE_CHECKING:
    from generation.model_pool import AnswerModelPool


class ModelRequestError(RuntimeError):
    """携带可恢复标记，避免模型池通过报错文字猜测切换条件。"""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def generation_error(model: str, error: requests.RequestException) -> ModelRequestError:
    """对外提供可定位的错误，不复述请求头或完整响应正文。"""
    response_status = error.response.status_code if error.response is not None else None
    status = re.search(r"status_code:\s*(\d+)", str(error))
    code = response_status or (int(status.group(1)) if status else None)
    if code == 404:
        return ModelRequestError(f"回答模型 {model} 在当前平台不可用（HTTP 404），请检查模型名称及权限")
    if code:
        return ModelRequestError(f"回答模型 {model} 请求失败（HTTP {code}）", code == 429 or code >= 500)
    return ModelRequestError(f"回答模型 {model} 连接失败：{type(error).__name__}",
                             isinstance(error, (requests.Timeout, requests.ConnectionError)))


class PlatformChatModel(ChatTongyi):
    """每次调用独立管理直连会话，避免代理故障和跨请求共享传输状态。"""

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            # 非流式入口可聚合流式结果，连接由下面的流式实现统一关闭。
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        try:
            with requests.Session() as session:
                session.trust_env = False
                kwargs["session"] = session
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except requests.RequestException as exc:
            raise generation_error(self.model_name, exc) from exc

    def _stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            with requests.Session() as session:
                session.trust_env = False
                kwargs["session"] = session
                yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        except requests.RequestException as exc:
            raise generation_error(self.model_name, exc) from exc


def build_qwen_model() -> PlatformChatModel:
    """模型名称保持配置值；SDK原生地址与兼容接口地址分别配置。"""
    return PlatformChatModel(model=config.chat_model_name, streaming=True, max_retries=2, model_kwargs={
        "base_address": config.dashscope_http_api_base_url,
        "enable_thinking": False,
        "request_timeout": 45,
    })


def build_chat_model() -> AnswerModelPool:
    """回答入口统一走模型池；未配置智谱时仍只有千问一个成员。"""
    import os
    from generation.model_pool import CompatibleChatModel, build_model_pool

    return build_model_pool(lambda: CompatibleChatModel(
        config.chat_model_name, config.embedding_base_url,
        os.getenv("DASHSCOPE_API_KEY", "").strip(), provider="qwen",
    ))
