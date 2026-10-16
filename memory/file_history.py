"""完整会话存档与压缩视图分离，旧版文件仅作兼容输入。"""

from __future__ import annotations

from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Callable, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from memory.session_context import build_context

logger = logging.getLogger(__name__)
_LOCKS = defaultdict(threading.RLock)


def session_paths(storage_path: str, session_id: str) -> tuple[Path, Path]:
    """限制会话标识为单个文件名，防止读写越出指定历史目录。"""
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        raise ValueError("会话ID只能包含字母、数字、下划线或短横线，长度1至128")
    root = Path(storage_path).resolve()
    legacy = root / session_id
    archive = root / ".memory_v2" / f"{session_id}.json"
    if not legacy.resolve().is_relative_to(root) or not archive.resolve().is_relative_to(root):
        raise ValueError("会话路径越出历史存储目录")
    return legacy, archive


def atomic_json(path: Path, value: object) -> None:
    """同目录原子替换单个状态文件，原文和摘要游标不会分别落盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            pending = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    except OSError as exc:
        raise RuntimeError(f"会话状态写入失败：{path.name}（{type(exc).__name__}），旧存档未主动删除") from exc
    finally:
        if pending is not None and pending.exists():
            pending.unlink()


class FileChatMessageHistory(BaseChatMessageHistory):
    """messages始终读取完整存档，context_messages才构建有预算的模型输入。"""

    def __init__(self, session_id: str, storage_path: str, summarizer: Callable | None = None) -> None:
        self.session_id = session_id
        self.storage_path = storage_path
        self.legacy_path, self.archive_path = session_paths(storage_path, session_id)
        self.file_path = str(self.archive_path)
        self.summarizer = summarizer
        self.lock = _LOCKS[str(self.archive_path)]

    def _read(self) -> dict:
        path = self.archive_path if self.archive_path.exists() else self.legacy_path
        if not path.exists():
            return {"schema_version": 2, "messages": [], "context_state": {}}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if path == self.legacy_path:
                if not isinstance(value, list):
                    raise ValueError("旧历史不是消息列表")
                value = {"schema_version": 2, "messages": value, "context_state": {}, "imported_from_legacy": True}
            if not isinstance(value, dict) or value.get("schema_version") != 2 or not isinstance(value.get("messages"), list):
                raise ValueError("历史结构或版本错误")
            if not isinstance(value.get("context_state"), dict):
                raise ValueError("上下文状态不是对象")
            messages_from_dict(value["messages"])
            return value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(f"会话 {self.session_id} 历史读取失败：{type(exc).__name__}，未覆盖原文件") from exc

    @property
    def messages(self) -> list[BaseMessage]:
        with self.lock:
            return messages_from_dict(self._read()["messages"])

    @property
    def context_state(self) -> dict:
        with self.lock:
            return dict(self._read()["context_state"])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        with self.lock:
            state = self._read()
            state["messages"].extend(message_to_dict(message) for message in messages)
            atomic_json(self.archive_path, state)

    def context_messages(self, retry_failed: bool = False) -> list[BaseMessage]:
        with self.lock:
            state = self._read()
            summarizer = self.summarizer
            if summarizer is None:
                from memory.history_store import _summarize_messages
                summarizer = _summarize_messages
            context, details = build_context(messages_from_dict(state["messages"]), state["context_state"],
                                             summarizer, retry_failed=retry_failed)
            if details != state["context_state"]:
                state["context_state"] = details
                atomic_json(self.archive_path, state)
            if details["status"] == "degraded":
                logger.warning("会话%s记忆降级：%s；完整历史仍保留", self.session_id, details["error"])
            return context

    def clear(self) -> None:
        """只有用户明确清空时才移除旧文件；空新版存档阻止旧数据重新出现。"""
        with self.lock:
            atomic_json(self.archive_path, {"schema_version": 2, "messages": [], "context_state": {}})
            if self.legacy_path.exists():
                self.legacy_path.unlink()


class ContextChatMessageHistory(BaseChatMessageHistory):
    """回答链读取压缩视图，写入仍追加到完整存档；前端不使用这个视图回放历史。"""

    def __init__(self, archive: FileChatMessageHistory) -> None:
        self.archive = archive

    @property
    def messages(self) -> list[BaseMessage]:
        return self.archive.context_messages()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.archive.add_messages(messages)

    def clear(self) -> None:
        self.archive.clear()
