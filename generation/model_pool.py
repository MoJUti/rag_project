"""回答模型池：按明确顺序调用，仅在输出前处理可恢复故障。"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlsplit

import requests
from langchain_core.messages import AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel, generate_from_stream
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import Field, SecretStr

from core import config
from generation.model import ModelRequestError, generation_error

logger = logging.getLogger(__name__)


class CompatibleChatModel(BaseChatModel):
    """统一兼容接口；千问关闭思考，智谱使用用户确认的低档思考。"""

    model: str
    base_url: str
    provider: str
    api_key: SecretStr = Field(exclude=True, repr=False)

    def __init__(self, model: str, base_url: str, api_key: str, provider: str = "zhipu") -> None:
        if not model or not api_key:
            raise ValueError(f"{provider} 回答模型未配置完整：请填写对应模型名称和 API 密钥")
        url = urlsplit(base_url)
        if url.scheme != "https" or not url.netloc or url.query or url.fragment or url.username:
            raise ValueError("回答模型 BASE_URL 必须为不含认证信息、查询参数的 HTTPS 接口根地址")
        if provider not in {"qwen", "zhipu"}:
            raise ValueError("回答供应商仅支持 qwen 或 zhipu")
        super().__init__(model=model, base_url=base_url.rstrip("/"), provider=provider,
                         api_key=SecretStr(api_key))

    @property
    def _llm_type(self) -> str:
        return "knowledge-base-compatible-chat"

    def _generate(self, messages: list, stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return generate_from_stream(self._stream(messages, stop=stop, run_manager=run_manager, **kwargs))

    def _stream(self, messages: list, stop: list[str] | None = None,
                run_manager: Any = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        roles = {"human": "user", "ai": "assistant", "system": "system"}
        if any(message.type not in roles for message in messages):
            raise ValueError("回答接口仅接受用户、助手和系统消息")
        body = {"model": self.model, "messages": [
            {"role": roles[message.type], "content": message.content} for message in messages
        ], "stream": True}
        body.update({"enable_thinking": False} if self.provider == "qwen"
                    else {"thinking": {"type": "enabled"}, "reasoning_effort": "low"})
        supported = {"stop", "max_tokens", "temperature", "top_p"}
        if set(kwargs) - supported:
            raise ValueError("回答接口收到未支持的生成参数")
        body.update(kwargs)
        if stop is not None:
            body["stop"] = stop
        finished = False
        identified = False
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.post(
                    f"{self.base_url}/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
                    stream=True, timeout=(10, 45),
                ) as response:
                    response.raise_for_status()
                    response.encoding = "utf-8"
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            finished = True
                            break
                        event = json.loads(data)
                        if event.get("error"):
                            raise ModelRequestError(f"回答模型 {self.model} 返回流内错误")
                        choices = event.get("choices") or []
                        choice = choices[0] if choices else {}
                        if choice.get("finish_reason") in {"length", "content_filter"}:
                            raise ModelRequestError(f"回答模型 {self.model} 未完整生成：{choice['finish_reason']}")
                        finished = finished or choice.get("finish_reason") is not None
                        content = choice.get("delta", {}).get("content") or ""
                        if not isinstance(content, str):
                            raise ValueError("回答流的正文不是文本")
                        # 固定标识只发一次，避免消息块合并时重复拼接供应商名称。
                        metadata = {} if identified else {
                            "provider": self.provider, "model_name": self.model,
                            "request_id": event.get("id"),
                        }
                        identified = True
                        if event.get("usage") is not None:
                            metadata["usage"] = event["usage"]
                        if choice.get("finish_reason") is not None:
                            metadata["finish_reason"] = choice["finish_reason"]
                        info = dict(metadata)
                        if "usage" in info:
                            info["token_usage"] = info.pop("usage")
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=content, response_metadata=metadata),
                            generation_info=info,
                        )
            if not finished:
                raise ModelRequestError(f"回答模型 {self.model} 的输出意外中断", retryable=True)
        except requests.RequestException as exc:
            raise generation_error(self.model, exc) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelRequestError(f"回答模型 {self.model} 返回格式异常") from exc


def _collect(chunks: Iterator[AIMessageChunk]) -> AIMessageChunk:
    """非流式调用聚合相同流，避免维护两套故障切换逻辑。"""
    result = None
    for chunk in chunks:
        result = chunk if result is None else result + chunk
    if result is None:
        raise ModelRequestError("回答模型没有返回消息")
    return result


class AnswerModelPool(Runnable):
    """工厂延迟创建模型；每次调用独立保存状态，不跨会话切换默认值。"""

    def __init__(self, members: list[tuple[str, Callable[[], Runnable]]]) -> None:
        if not members:
            raise ValueError("回答模型池不能为空")
        self.members = tuple(members)

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> AIMessageChunk:
        return _collect(self.stream(input, config, **kwargs))

    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[AIMessageChunk]:
        for index, (name, factory) in enumerate(self.members):
            emitted = False
            pending = []
            try:
                for chunk in factory().stream(input, config, **kwargs):
                    # 先缓冲空块；正文一旦对外输出，就不能再接另一模型的回答。
                    if not emitted and not chunk.content:
                        pending.append(chunk)
                        continue
                    emitted = True
                    for buffered in pending:
                        yield buffered
                    pending.clear()
                    yield chunk
                if not emitted:
                    raise ModelRequestError(f"回答模型 {name} 未返回正文")
                logger.info("回答模型池完成：provider=%s", name)
                return
            except ModelRequestError as exc:
                if emitted or not exc.retryable or index == len(self.members) - 1:
                    raise ModelRequestError(f"回答模型池在 {name} 失败：{exc}") from exc
                logger.warning("回答模型 %s 发生可恢复故障，切换至 %s：%s",
                               name, self.members[index + 1][0], exc)


def build_model_pool(qwen_factory: Callable[[], Runnable]) -> AnswerModelPool:
    """千问固定优先；备用策略只有显式启用且配置完整才生效。"""
    members = [("qwen", qwen_factory)]
    if config.chat_fallback_provider not in {"none", "zhipu"}:
        raise ValueError("CHAT_FALLBACK_PROVIDER 仅支持 none 或 zhipu")
    if config.chat_fallback_provider == "zhipu":
        model = CompatibleChatModel(config.zhipu_chat_model, config.zhipu_base_url,
                               os.getenv("ZHIPU_API_KEY", "").strip())
        members.append(("zhipu", lambda: model))
    return AnswerModelPool(members)

