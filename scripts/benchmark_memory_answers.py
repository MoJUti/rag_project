"""复用固定摘要，以同一道问题交错重复比较记忆问答质量和流式耗时。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Callable, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from core import config
from core.prompt_loader import load_prompt
from memory.history_store import _build_light_model
from memory.session_context import fingerprint
from scripts.run_node6_rerank_eval import save_cache

QUESTION = "请列出 P001、P024、P048 各自已确认的预算（元）、交付日期，以及是否禁止将资料外发。"
EXPECTED = {
    "P001": {"budget_yuan": 7017, "delivery_date": "2027-11-02", "external_sharing_forbidden": True},
    "P024": {"budget_yuan": 7408, "delivery_date": "2027-11-25", "external_sharing_forbidden": True},
    "P048": {"budget_yuan": 7816, "delivery_date": "2027-11-21", "external_sharing_forbidden": True},
}


def score_answer(text: str) -> dict:
    """严格比较九个事实，格式错误、未知和错误事实分别记录。"""
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        return {"format_valid": False, "correct_fields": 0, "complete_correct": False}
    if not isinstance(answer, dict):
        return {"format_valid": False, "correct_fields": 0, "complete_correct": False}
    details = {}
    for group, fields in EXPECTED.items():
        actual = answer.get(group)
        actual = actual if isinstance(actual, dict) else {}
        for field, expected in fields.items():
            value = actual.get(field)
            details[f"{group}.{field}"] = (
                "unknown" if value is None else "correct"
                if type(value) is type(expected) and value == expected else "wrong")
    correct = sum(value == "correct" for value in details.values())
    return {"format_valid": True, "correct_fields": correct,
            "complete_correct": correct == 9, "details": details}


def timed_answer(model: Any, messages: list, clock: Callable[[], float] = time.perf_counter) -> dict:
    """首字以首次非空正文为准；异常时保留已输出正文，不伪装成完成。"""
    start, first, parts, metadata = clock(), None, [], {}
    error = None
    try:
        for chunk in model.stream(messages, max_tokens=800):
            if chunk.content:
                if first is None:
                    first = clock() - start
                parts.append(chunk.content)
            metadata.update(chunk.response_metadata)
        if not parts:
            raise RuntimeError("记忆问答未返回正文")
    except Exception as exc:
        # 评测边界保存失败现场，调用方会累计失败并终止批次；不作自动重试。
        error = f"{type(exc).__name__}: {exc}"
    duration = clock() - start
    text = "".join(parts)
    return {"status": "failed" if error else "completed", "error": error,
            "first_text_seconds": first, "total_seconds": duration,
            "text": text, "metadata": metadata, "score": score_answer(text)}


def main(source: Path, live: bool) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["identity"]["model"] != config.zhipu_chat_model or data["identity"]["base_url"] != config.zhipu_base_url:
        raise ValueError("当前模型或平台与摘要实验不一致，请先确认配置")
    prompt = load_prompt("memory/budget_answer_probe.md")
    names = list(data["rows"])
    orders = [names, list(reversed(names)), names[3:] + names[:3]]
    identity = {"protocol": "memory-answer-latency-v1", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model": config.zhipu_chat_model, "base_url": config.zhipu_base_url,
                "reasoning_effort": "low", "max_tokens": 800, "question": QUESTION,
                "prompt": prompt, "orders": orders, "expected": EXPECTED}
    output = source.parent / f"answers_{fingerprint(identity)[:12]}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"identity": identity, "rows": {}}
    if report["identity"] != identity:
        raise ValueError("问答耗时缓存身份不一致")
    print(f"report={output}; planned_calls=21; live={live}", flush=True)
    if not live:
        return
    failures = 0
    for repeat, order in enumerate(orders, 1):
        for name in order:
            key = f"{repeat}:{name}"
            if key in report["rows"]:
                continue
            row = data["rows"][name]
            # 原摘要降为历史数据，避免其系统角色在评测中获得指令权限。
            history = [{"role": m["type"], "content": m["data"]["content"]} for m in row["context"]]
            messages = [SystemMessage(content=prompt), HumanMessage(content=json.dumps(
                {"history": history, "question": QUESTION}, ensure_ascii=False))]
            result = timed_answer(_build_light_model(), messages)
            result.update({"profile": name, "repeat": repeat, "context_status": row["state"]["status"]})
            report["rows"][key] = result
            save_cache(output, report)
            print(json.dumps({"case": key, **{k: result[k] for k in (
                "status", "first_text_seconds", "total_seconds", "score")}}, ensure_ascii=False), flush=True)
            failures = failures + 1 if result["status"] == "failed" else 0
            if failures >= 3:
                raise RuntimeError("连续三次调用失败，已保存结果并停止付费批次")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "storage/evaluations/session_memory_v1/reports/budgets_1bd74ae46e94_live.json")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    main(args.source, args.live)
