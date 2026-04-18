import json
import os
from pathlib import Path
from typing import Any
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.chat_models import init_chat_model

from core import config


def _debug_log(message: str) -> None:
    """根据配置决定是否输出压缩调试日志。"""
    if bool(getattr(config, "memory_compression_debug", False)):
        print(f"[MEMORY_COMPRESS] {message}")


def _build_light_model():
    """构建轻量模型实例，专用于内部摘要任务，失败时自动回退。"""
    try:
        # model: 实际调用的轻量模型名称，例如 qwen-turbo。
        model = config.light_model_name
        return init_chat_model(
            model=model,
            model_provider="tongyi",
        )
    except Exception:
        # 回退策略：保持历史兼容，避免因为新版初始化失败导致主链路不可用。
        return ChatTongyi(model=config.light_model_name)


def _estimate_message_tokens(message: BaseMessage) -> int:
    """估算单条消息 token 数。

    说明：
    - 当前项目未强依赖专用 tokenizer，这里使用中文场景常见的经验值估算。
    - 估算目标是“稳定控制上限”，不追求逐 token 精确计数。
    """
    # content_text: 统一把消息内容转成字符串后再估算。
    content_text = str(message.content) if message.content is not None else ""
    # rough_tokens: 经验近似。中文通常 1 个字约等于 1~2 token，取 len/2 作为折中。
    rough_tokens = max(1, len(content_text) // 2)
    # role_overhead: 消息角色、结构化包装等固定开销。
    role_overhead = 6
    return rough_tokens + role_overhead


def _estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    """估算消息列表的总 token 数。"""
    return sum(_estimate_message_tokens(msg) for msg in messages)


def _is_summary_message(message: BaseMessage) -> bool:
    """判断消息是否为系统内部摘要消息。"""
    if message.type != "system":
        return False
    # summary_tag: 摘要标识前缀，用于区分普通 system 提示与内部摘要。
    summary_tag = config.memory_summary_tag
    return str(message.content).startswith(summary_tag)


def _build_summary_prompt(existing_summary: str, transcript: str) -> str:
    """构建摘要提示词，要求输出结构化且可持续追加的会话记忆。"""
    return (
        "你是对话记忆压缩器。请将以下历史对话压缩为简洁摘要，供后续多轮问答使用。\\n"
        "要求：\\n"
        "1) 保留关键事实、用户诉求、结论与未解决问题。\\n"
        "2) 删除寒暄、重复表达、无信息量句子。\\n"
        "3) 输出使用中文，控制在 6-12 条要点。\\n"
        "4) 若已有旧摘要，请在其基础上增量更新，而不是简单重复。\\n\\n"
        f"旧摘要：\\n{existing_summary or '（无）'}\\n\\n"
        f"待压缩对话：\\n{transcript or '（无）'}"
    )


def _fallback_summary(existing_summary: str, transcript: str) -> str:
    """轻量兜底摘要：当模型调用失败时，仍保证压缩流程可执行。"""
    # merged_text: 把旧摘要与新对话拼接，做最小可用截断。
    merged_text = "\\n".join(part for part in [existing_summary, transcript] if part).strip()
    if not merged_text:
        return ""
    # max_chars: 使用配置控制摘要最大字符，防止摘要本身无限增长。
    max_chars = config.memory_summary_max_chars
    return merged_text[:max_chars]


def _summarize_messages(
    messages_to_summarize: list[BaseMessage],
    existing_summary_text: str,
) -> str:
    """调用轻量模型对指定消息做增量摘要，失败则自动回退。"""
    # lines: 将历史消息转成可读 transcript，便于模型提炼关键信息。
    lines: list[str] = []
    for message in messages_to_summarize:
        if message.type == "human":
            role_label = "用户"
        elif message.type == "ai":
            role_label = "助手"
        else:
            role_label = "系统"
        lines.append(f"[{role_label}] {str(message.content)}")

    # transcript: 可被摘要模型消费的原始会话文本。
    transcript = "\\n".join(lines)
    # prompt_text: 摘要任务提示词。
    prompt_text = _build_summary_prompt(existing_summary_text, transcript)

    try:
        # light_model: 专用于低成本摘要压缩的小模型（qwen-turbo）。
        light_model = _build_light_model()
        # response: 模型返回对象，可能是 BaseMessage，也可能是字符串。
        response = light_model.invoke(prompt_text)
        if hasattr(response, "content"):
            summary_text = str(response.content).strip()
        else:
            summary_text = str(response).strip()
        # max_chars: 再次做输出上限保护。
        max_chars = config.memory_summary_max_chars
        clipped_summary = summary_text[:max_chars]
        _debug_log(
            "摘要生成成功: "
            f"输入消息={len(messages_to_summarize)}条, "
            f"旧摘要长度={len(existing_summary_text)}, "
            f"新摘要长度={len(clipped_summary)}"
        )
        return clipped_summary
    except Exception:
        _debug_log("摘要模型调用失败，进入 fallback 摘要逻辑")
        return _fallback_summary(existing_summary_text, transcript)


def _split_rounds(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """按“用户发言”为边界，把历史消息切分为会话轮次。"""
    rounds: list[list[BaseMessage]] = []
    # current_round: 正在构建的轮次缓存。
    current_round: list[BaseMessage] = []
    for message in messages:
        if message.type == "human" and current_round:
            rounds.append(current_round)
            current_round = [message]
        else:
            current_round.append(message)
    if current_round:
        rounds.append(current_round)
    return rounds


def _flatten_rounds(rounds: list[list[BaseMessage]]) -> list[BaseMessage]:
    """把轮次结构还原为消息列表。"""
    flattened: list[BaseMessage] = []
    for round_messages in rounds:
        flattened.extend(round_messages)
    return flattened


def _compress_messages(all_messages: list[BaseMessage]) -> list[BaseMessage]:
    """执行双轨压缩：摘要压缩 + 滑动窗口 + Token 预算裁剪。"""
    if not all_messages:
        _debug_log("输入为空，跳过压缩")
        return []

    input_tokens = _estimate_messages_tokens(all_messages)
    _debug_log(f"开始压缩: 输入消息={len(all_messages)}条, 估算token={input_tokens}")

    # 1) 拆分已有摘要消息和普通对话消息。
    summary_messages = [msg for msg in all_messages if _is_summary_message(msg)]
    normal_messages = [msg for msg in all_messages if not _is_summary_message(msg)]

    # old_summary_text: 只保留最新一条摘要作为增量基线。
    old_summary_text = ""
    if summary_messages:
        latest_summary = summary_messages[-1]
        latest_content = str(latest_summary.content)
        summary_tag = config.memory_summary_tag
        old_summary_text = latest_content.replace(summary_tag, "", 1).strip()

    # 2) 基于普通消息切分轮次，先做滑动窗口。
    rounds = _split_rounds(normal_messages)
    total_rounds = len(rounds)
    keep_recent_rounds = max(1, int(config.memory_keep_recent_rounds))
    summary_trigger_rounds = max(keep_recent_rounds + 1, int(config.memory_summary_trigger_rounds))

    # recent_rounds: 永远优先保留最近 N 轮，确保短期对话连续性。
    recent_rounds = rounds[-keep_recent_rounds:]
    # older_rounds: 可被摘要压缩的旧轮次。
    older_rounds = rounds[:-keep_recent_rounds]

    _debug_log(
        "轮次切分完成: "
        f"总轮次={total_rounds}, 保留最近轮次={len(recent_rounds)}, 可摘要旧轮次={len(older_rounds)}"
    )

    # 3) 满足触发条件时更新摘要。
    new_summary_text = old_summary_text
    should_refresh_summary = (
        bool(config.memory_summary_enabled)
        and total_rounds >= summary_trigger_rounds
        and len(older_rounds) > 0
    )
    if should_refresh_summary:
        messages_for_summary = _flatten_rounds(older_rounds)
        _debug_log(
            "触发摘要: "
            f"触发阈值={summary_trigger_rounds}, 待摘要消息={len(messages_for_summary)}条"
        )
        new_summary_text = _summarize_messages(messages_for_summary, old_summary_text)
    else:
        _debug_log(
            "未触发摘要: "
            f"summary_enabled={bool(config.memory_summary_enabled)}, "
            f"total_rounds={total_rounds}, trigger={summary_trigger_rounds}, older_rounds={len(older_rounds)}"
        )

    # 4) 重建压缩后的消息序列（摘要 + 最近轮次）。
    compressed_messages: list[BaseMessage] = []
    if new_summary_text:
        summary_tag = config.memory_summary_tag
        summary_message = SystemMessage(content=f"{summary_tag}\\n{new_summary_text}")
        compressed_messages.append(summary_message)
    compressed_messages.extend(_flatten_rounds(recent_rounds))

    # 5) 应用 Token 预算裁剪：优先保留摘要与最新轮次。
    max_tokens = max(200, int(config.memory_history_max_tokens))
    trim_count = 0
    current_tokens = _estimate_messages_tokens(compressed_messages)
    if current_tokens > max_tokens:
        _debug_log(
            "触发预算裁剪: "
            f"当前token={current_tokens}, 预算上限={max_tokens}, "
            f"将按轮次从旧到新删除最近轮次"
        )
    # while 条件里保底至少 1 轮对话，避免上下文被裁到只剩摘要。
    while _estimate_messages_tokens(compressed_messages) > max_tokens and len(recent_rounds) > 1:
        # 逐轮移除最旧轮次，直到满足预算。
        recent_rounds = recent_rounds[1:]
        trim_count += 1
        compressed_messages = []
        if new_summary_text:
            summary_tag = config.memory_summary_tag
            compressed_messages.append(SystemMessage(content=f"{summary_tag}\\n{new_summary_text}"))
        compressed_messages.extend(_flatten_rounds(recent_rounds))

    output_tokens = _estimate_messages_tokens(compressed_messages)
    summary_exists = any(_is_summary_message(m) for m in compressed_messages)
    _debug_log(
        "压缩完成: "
        f"输出消息={len(compressed_messages)}条, "
        f"输出token={output_tokens}/{max_tokens}, "
        f"是否含摘要={summary_exists}, "
        f"裁剪轮次数={trim_count}, "
        f"保留轮次={len(recent_rounds)}"
    )

    return compressed_messages


class FileChatMessageHistory(BaseChatMessageHistory):
    """基于本地 JSON 文件的会话历史实现，并内置压缩策略。"""

    def __init__(self, session_id: str, storage_path: str):
        """初始化单会话历史存储对象。"""
        # session_id: 会话唯一标识，用于生成会话文件名。
        self.session_id = session_id
        # storage_path: 历史文件目录，例如 chat_history 目录。
        self.storage_path = storage_path
        # file_path: 当前会话对应的具体文件路径。
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """追加消息并执行压缩后落盘。

        流程：
        1) 读取旧历史 + 新增消息
        2) 执行摘要/滑窗/Token 三层压缩
        3) 序列化写入本地文件
        """
        # all_messages: 本次写入前的完整消息序列。
        all_messages = list(self.messages)
        all_messages.extend(messages)
        # compressed_messages: 压缩后的可持久化消息序列。
        compressed_messages = _compress_messages(all_messages)
        # serialized: LangChain 标准消息字典格式，便于后续反序列化。
        serialized = [message_to_dict(message) for message in compressed_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f)

    @property
    def messages(self) -> list[BaseMessage]:
        """读取并反序列化会话消息。文件不存在或损坏时返回空列表。"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                # message_data: 原始 JSON 列表。
                message_data = json.load(f)
                return messages_from_dict(message_data)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def clear(self) -> None:
        """清空当前会话历史。"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)


def get_history(session_id: str):
    """按会话 ID 获取历史对象。"""
    return FileChatMessageHistory(session_id, config.chat_history_directory)


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
    sessions: list[str] = []
    for file in directory.iterdir():
        if file.is_file() and file.name.startswith("chat_"):
            sessions.append(file.name)

    def sort_key(session_id: str):
        """构造排序键：置顶优先，其次最近修改时间，再按 ID。"""
        # pinned: 当前会话是否置顶。
        pinned = bool(meta.get(session_id, {}).get("pinned", False))
        # file_path: 会话文件路径。
        file_path = directory / session_id
        # mtime: 文件最后修改时间戳。
        mtime = file_path.stat().st_mtime if file_path.exists() else 0
        return (0 if pinned else 1, -mtime, session_id.lower())

    return sorted(sessions, key=sort_key)


def delete_history(session_id: str) -> None:
    """删除指定会话历史及其元数据。"""
    file_path = _history_dir() / session_id
    if file_path.exists() and file_path.is_file():
        file_path.unlink()

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
