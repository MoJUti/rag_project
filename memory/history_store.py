"""会话历史入口：前端读取完整存档，回答链读取受预算约束的上下文视图。"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from core import config
from core.prompt_loader import render_prompt
from generation.model_pool import CompatibleChatModel
from memory.file_history import ContextChatMessageHistory, FileChatMessageHistory, session_paths
from memory.session_context import (
    SummaryResult, build_context, estimate_message_tokens, estimate_tokens, is_summary, split_rounds, validate_summary,
)

logger = logging.getLogger(__name__)


def _build_light_model() -> CompatibleChatModel:
    """摘要只使用用户确认的GLM，适配器沿用low档思考。"""
    return CompatibleChatModel(config.zhipu_chat_model, config.zhipu_base_url,
                               os.getenv("ZHIPU_API_KEY", "").strip(), provider="zhipu")


def _estimate_message_tokens(message: BaseMessage) -> int:
    return estimate_message_tokens(message)


def _estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    return estimate_tokens(messages)


def _is_summary_message(message: BaseMessage) -> bool:
    return is_summary(message)


def _build_summary_prompt(existing_summary: str, transcript: str) -> str:
    """原文只作为摘要数据传入，不将历史指令提升为系统规则。"""
    return render_prompt("memory/conversation_summary.md", {
        "existing_summary": existing_summary or "（无）", "transcript": transcript or "（无）",
        "summary_limit": (f"最多{config.memory_summary_max_tokens}估算Token" if config.memory_summary_max_tokens is not None
                          else f"最多{config.memory_summary_max_chars}字符"),
    })


def _summarize_messages(messages_to_summarize: list[BaseMessage], existing_summary_text: str) -> SummaryResult:
    """模型失败、空摘要、过长摘要均明确失败，由上下文构建器记录降级。"""
    roles = {"human": "用户", "ai": "助手"}
    transcript = "\n".join(f"[{roles.get(message.type, '历史其他消息')}] {message.content}"
                           for message in messages_to_summarize)
    prompt = _build_summary_prompt(existing_summary_text, transcript)
    try:
        response = _build_light_model().invoke([HumanMessage(content=prompt)])
        text = str(response.content).strip()
        validate_summary(text)
        return SummaryResult(text, dict(getattr(response, "response_metadata", {})))
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"GLM会话摘要失败：{type(exc).__name__}") from exc


def _split_rounds(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    return split_rounds(messages)


def _flatten_rounds(rounds: list[list[BaseMessage]]) -> list[BaseMessage]:
    return [message for row in rounds for message in row]


def _compress_messages(all_messages: list[BaseMessage]) -> list[BaseMessage]:
    """保留旧诊断脚本入口；正式增量缓存由FileChatMessageHistory管理。"""
    context, state = build_context(all_messages, {}, _summarize_messages)
    if state["status"] == "degraded":
        logger.warning("记忆压缩降级：%s", state["error"])
    return context


def get_history(session_id: str) -> FileChatMessageHistory:
    """UI使用完整历史，不因模型上下文压缩而丢失可回放的消息。"""
    return FileChatMessageHistory(session_id, config.chat_history_directory)


def get_context_history(session_id: str) -> ContextChatMessageHistory:
    """回答链使用压缩视图，新增消息仍写入同一会话完整存档。"""
    return ContextChatMessageHistory(get_history(session_id))


def _history_dir() -> Path:
    """获取历史目录路径，不存在则自动创建。"""
    path = Path(config.chat_history_directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _meta_file() -> Path:
    """返回会话元数据文件路径。"""
    return _history_dir() / ".session_meta.json"


def _load_meta() -> dict[str, Any]:
    """加载会话元数据，异常时返回空字典。"""
    file_path = _meta_file()
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # data: 元数据 JSON 解析结果。
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _save_meta(meta: dict[str, Any]) -> None:
    """保存会话元数据到磁盘。"""
    with open(_meta_file(), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def list_session_ids() -> list[str]:
    """列出全部会话 ID，并按置顶、更新时间、字母序排序。"""
    directory = _history_dir()
    meta = _load_meta()
    # sessions: 收集到的会话文件名列表。
    sessions: set[str] = set()
    for file in directory.iterdir():
        if file.is_file() and file.name.startswith("chat_"):
            sessions.add(file.name)
    archive_directory = directory / ".memory_v2"
    if archive_directory.exists():
        sessions.update(file.stem for file in archive_directory.glob("chat_*.json") if file.is_file())

    def sort_key(session_id: str):
        """构造排序键：置顶优先，其次最近修改时间，再按 ID。"""
        # pinned: 当前会话是否置顶。
        pinned = bool(meta.get(session_id, {}).get("pinned", False))
        # file_path: 会话文件路径。
        legacy, archive = session_paths(str(directory), session_id)
        file_path = archive if archive.exists() else legacy
        # mtime: 文件最后修改时间戳。
        mtime = file_path.stat().st_mtime if file_path.exists() else 0
        return (0 if pinned else 1, -mtime, session_id.lower())

    return sorted(sessions, key=sort_key)


def delete_history(session_id: str) -> None:
    """删除指定会话历史及其元数据。"""
    history = get_history(session_id)
    # 先写空存档，确保删除旧文件中断时也不会重新读到旧对话。
    history.clear()
    if history.archive_path.exists():
        history.archive_path.unlink()

    meta = _load_meta()
    if session_id in meta:
        meta.pop(session_id, None)
        _save_meta(meta)


def is_session_pinned(session_id: str) -> bool:
    """查询会话是否被置顶。"""
    meta = _load_meta()
    return bool(meta.get(session_id, {}).get("pinned", False))


def set_session_pinned(session_id: str, pinned: bool) -> None:
    """设置会话置顶状态。"""
    meta = _load_meta()
    # session_meta: 指定会话的元数据字典。
    session_meta = meta.get(session_id, {})
    session_meta["pinned"] = pinned
    meta[session_id] = session_meta
    _save_meta(meta)


def toggle_session_pinned(session_id: str) -> bool:
    """切换会话置顶状态并返回新状态。"""
    new_state = not is_session_pinned(session_id)
    set_session_pinned(session_id, new_state)
    return new_state
