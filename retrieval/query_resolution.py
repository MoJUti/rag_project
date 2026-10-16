"""可插拔的历史查询消歧；结果按请求返回，不在插件中保存会话状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
from typing import Callable, Protocol, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from core import config
from core.prompt_loader import load_prompt
from generation.model_pool import CompatibleChatModel


@dataclass
class QueryResolution:
    original_query: str
    query: str
    action: str = "original"
    clarification: str = ""
    anchors: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0
    model_info: dict = field(default_factory=dict)


class QueryResolver(Protocol):
    def resolve(self, query: str, history: list[BaseMessage]) -> QueryResolution: ...


class OriginalQueryResolver:
    """默认插件不调用模型，完全保留原问题检索。"""

    def resolve(self, query: str, history: list[BaseMessage]) -> QueryResolution:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询消歧阶段的问题不能为空")
        return QueryResolution(query, query)


def parse_resolution(text: str, query: str, history: list[BaseMessage]) -> QueryResolution:
    """严格校验协议与逐字锚点；锚点存在不等于模型语义一定正确，仍需评测。"""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("查询消歧返回无效JSON") from exc
    if not isinstance(value, dict) or set(value) != {"action", "query", "clarification", "anchors"}:
        raise ValueError("查询消歧返回字段不完整或存在多余字段")
    action, resolved, clarification, anchors = (value[k] for k in ("action", "query", "clarification", "anchors"))
    if not isinstance(action, str) or action not in {"original", "rewrite", "clarify"} or not isinstance(resolved, str) or not isinstance(clarification, str) or not isinstance(anchors, list):
        raise ValueError("查询消歧返回了非法动作或字段类型")
    if action == "original" and (resolved != query or clarification or anchors):
        raise ValueError("原问题策略必须保持问题不变，且不附加锚点或澄清")
    if action == "clarify" and (resolved or not clarification.strip() or anchors):
        raise ValueError("澄清策略必须停止查询并提供明确澄清问题")
    if action == "rewrite" and (not resolved.strip() or clarification or not anchors):
        raise ValueError("查询改写必须提供完整问题与历史锚点，且不同时澄清")
    for anchor in anchors:
        if not isinstance(anchor, dict) or set(anchor) != {"message_index", "quote"}:
            raise ValueError("查询消歧的历史锚点字段错误")
        index, quote = anchor["message_index"], anchor["quote"]
        if type(index) is not int or not 0 <= index < len(history):
            raise ValueError("查询消歧引用了不存在的历史消息")
        if history[index].type not in {"human", "system"} or not isinstance(quote, str) or not quote.strip() or quote not in str(history[index].content):
            raise ValueError("查询消歧锚点必须逐字来自用户历史或会话摘要")
    return QueryResolution(query, resolved, action, clarification, anchors)


def _glm_model() -> CompatibleChatModel:
    return CompatibleChatModel(config.zhipu_chat_model, config.zhipu_base_url,
                               os.getenv("ZHIPU_API_KEY", "").strip(), provider="zhipu")


class HistoryQueryResolver(OriginalQueryResolver):
    """显式启用后使用已配置GLM、low档；故障明确报错，不悄悄猜测或降级查询。"""

    def __init__(self, model_factory: Callable[[], Any] = _glm_model) -> None:
        self.model_factory = model_factory

    def resolve(self, query: str, history: list[BaseMessage]) -> QueryResolution:
        super().resolve(query, history)
        started = time.perf_counter()
        payload = {"query": query, "history": [{"index": i, "role": m.type, "content": m.content}
                                               for i, m in enumerate(history)]}
        try:
            response = self.model_factory().invoke([SystemMessage(content=load_prompt("retrieval/history_query.md")),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False))], max_tokens=800)
            result = parse_resolution(str(response.content), query, history)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(f"历史查询消歧失败：{exc}") from exc
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        result.model_info = dict(response.response_metadata)
        return result


QUERY_RESOLVER_FACTORIES: dict[str, Callable[[], QueryResolver]] = {
    "original": OriginalQueryResolver,
    "glm_history": HistoryQueryResolver,
}


def create_query_resolver(strategy: str) -> QueryResolver:
    try:
        factory = QUERY_RESOLVER_FACTORIES[strategy]
    except KeyError as exc:
        raise ValueError(f"不支持的历史查询消歧策略：{strategy}") from exc
    return factory()
