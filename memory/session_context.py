"""从完整会话生成可重建的上下文视图，不删除原始消息。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from typing import Callable

from langchain_core.messages import BaseMessage, SystemMessage, message_to_dict

from core import config
from core.prompt_loader import load_prompt


class MemoryBudgetError(ValueError):
    """最新一轮本身超过预算，不能悄悄截断后当作完整历史。"""


@dataclass
class SummaryResult:
    text: str
    metadata: dict = field(default_factory=dict)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def encoding() -> object | None:
    """优先沿用项目已安装的分词器；缺失时采用保守字节估计，不在线下载模型。"""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except (ImportError, OSError, ValueError):
        return None


def estimate_message_tokens(message: BaseMessage) -> int:
    return estimate_text_tokens(str(message.content)) + 6


def estimate_text_tokens(text: str) -> int:
    """不计消息角色开销的正文估算，用于统一摘要Token上限。"""
    tokenizer = encoding()
    return len(tokenizer.encode(text, disallowed_special=())) if tokenizer else len(text.encode("utf-8"))


def validate_summary(text: str) -> None:
    limit = getattr(config, "memory_summary_max_tokens", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("摘要为空或类型错误")
    if limit is not None:
        if int(limit) < 1 or estimate_text_tokens(text) > int(limit):
            raise ValueError("摘要超过Token上限或上限无效")
    elif len(text) > config.memory_summary_max_chars:
        raise ValueError("摘要超过字符上限")


def estimate_tokens(messages: list[BaseMessage]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def is_summary(message: BaseMessage) -> bool:
    return message.type == "system" and str(message.content).startswith(config.memory_summary_tag)


def split_rounds(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    rounds = []
    for message in messages:
        if message.type == "human" or not rounds:
            rounds.append([])
        rounds[-1].append(message)
    return rounds


def build_context(messages: list[BaseMessage], previous: dict, summarize: Callable,
                  retry_failed: bool = False) -> tuple[list[BaseMessage], dict]:
    """只摘要新进入较早区的消息；同一失败批次不在重复读取时反复付费。"""
    keep = int(config.memory_keep_recent_rounds)
    trigger = int(config.memory_summary_trigger_rounds)
    budget = int(config.memory_history_max_tokens)
    mode = getattr(config, "memory_compression_trigger", "rounds")
    if mode not in {"rounds", "tokens"}:
        raise ValueError("记忆触发方式只支持rounds或tokens")
    if keep < 1 or trigger <= keep or budget < 1 or int(config.memory_summary_max_chars) < 1:
        raise ValueError("记忆配置无效：保留轮数和预算须大于0，摘要触发轮数须大于保留轮数")
    policy = fingerprint({"keep": keep, "trigger": trigger, "budget": budget,
        "enabled": config.memory_summary_enabled, "summary_max_chars": config.memory_summary_max_chars,
        "summary_max_tokens": getattr(config, "memory_summary_max_tokens", None), "trigger_mode": mode,
        "model": config.zhipu_chat_model, "base_url": config.zhipu_base_url, "reasoning_effort": "low",
        "prompt": load_prompt("memory/conversation_summary.md", ("existing_summary", "transcript", "summary_limit"))})
    legacy = [str(message.content)[len(config.memory_summary_tag):].strip() for message in messages if is_summary(message)]
    normal = [message for message in messages if not is_summary(message)]
    state = dict(previous) if previous.get("policy") == policy else {
        "policy": policy, "summary": legacy[-1] if legacy else "", "covered_messages": 0}
    if not isinstance(state.get("summary"), str) or "covered_messages" not in state:
        raise ValueError("记忆摘要状态缺少文本或覆盖位置")
    covered = state["covered_messages"]
    if not isinstance(covered, int) or covered < 0 or covered > len(normal):
        raise ValueError("记忆摘要覆盖位置与完整历史不一致")
    if covered and state.get("covered_sha256") != fingerprint([message_to_dict(m) for m in normal[:covered]]):
        raise ValueError("记忆摘要覆盖的原始消息指纹不一致")
    rounds = split_rounds(normal)
    if rounds and estimate_tokens(rounds[-1]) > budget:
        raise MemoryBudgetError("会话最新一轮超过历史上下文预算，请提高预算或缩短输入；完整历史未被裁掉")
    older_end = sum(len(row) for row in rounds[:-keep])
    if mode == "tokens":
        summary_limit = getattr(config, "memory_summary_max_tokens", None)
        if summary_limit is None or not 0 < int(summary_limit) < budget:
            raise ValueError("Token触发实验需要正的摘要Token上限，且小于历史总预算")
        remaining = budget - int(summary_limit) - estimate_message_tokens(SystemMessage(content=config.memory_summary_tag))
        retained = 0
        for row in reversed(rounds):
            cost = estimate_tokens(row)
            if cost > remaining:
                break
            remaining -= cost
            retained += len(row)
        older_end = max(0, len(normal) - retained)
    active = ([SystemMessage(content=f"{config.memory_summary_tag}\n{state['summary']}")] if state["summary"] else []) + normal[covered:]
    should_refresh = config.memory_summary_enabled and older_end > covered and (
        (mode == "rounds" and len(rounds) >= trigger) or estimate_tokens(active) > budget)
    attempt = fingerprint([message_to_dict(m) for m in normal[:older_end]])
    if should_refresh and (retry_failed or state.get("attempt_sha256") != attempt):
        state["attempt_sha256"] = attempt
        try:
            result = summarize(normal[covered:older_end], state["summary"])
            text = result.text if isinstance(result, SummaryResult) else result
            validate_summary(text)
            state.update(summary=text.strip(), covered_messages=older_end,
                         covered_sha256=fingerprint([message_to_dict(m) for m in normal[:older_end]]),
                         status="ready", error=None,
                         model_info=result.metadata if isinstance(result, SummaryResult) else {})
            covered = older_end
        except (RuntimeError, ValueError, TypeError) as exc:
            # 不保存接口原始异常，避免其中包含认证信息；详细类型足以标明降级原因。
            state.update(status="degraded", error=f"摘要失败：{type(exc).__name__}")
    if not config.memory_summary_enabled:
        state["status"] = "disabled"
    else:
        state.setdefault("status", "not_triggered")
    tail = normal[covered:] if config.memory_summary_enabled else [m for row in rounds[-keep:] for m in row]
    tail_rounds = split_rounds(tail)
    summary = [SystemMessage(content=f"{config.memory_summary_tag}\n{state['summary']}")] if state["summary"] and config.memory_summary_enabled else []
    dropped = 0
    while estimate_tokens(summary + [m for row in tail_rounds for m in row]) > budget and len(tail_rounds) > 1:
        dropped += len(tail_rounds.pop(0))
    context = summary + [m for row in tail_rounds for m in row]
    if estimate_tokens(context) > budget and summary:
        # 先保留最新原文；旧摘要仍在状态中，不因为这一次预算不足而抹除。
        context = [m for row in tail_rounds for m in row]
        state["summary_omitted_for_budget"] = True
    else:
        state["summary_omitted_for_budget"] = False
    if estimate_tokens(context) > budget:
        raise MemoryBudgetError("会话最新一轮超过历史上下文预算，请提高预算或缩短输入；完整历史未被裁掉")
    if dropped or state["summary_omitted_for_budget"]:
        state["status"] = "degraded"
        state["error"] = state.get("error") or "上下文预算不足，部分历史仅保存在完整存档中"
    state.update(context_estimated_tokens=estimate_tokens(context), dropped_messages_for_context=dropped,
                 token_estimator="cl100k_base+角色估计" if encoding() else "UTF-8字节保守估计")
    return context, state
