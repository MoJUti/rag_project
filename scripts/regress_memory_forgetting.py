"""复放固定多轮轨迹，验证摘要撤除旧值及旧摘要修复，不重复检索或完整历史对照。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage, message_to_dict, messages_from_dict
from core import config
from core.prompt_loader import load_prompt
from memory.history_store import _build_light_model, _summarize_messages
from memory.session_context import SummaryResult, build_context, fingerprint
from scripts.benchmark_memory_answers import timed_answer
from scripts.evaluate_memory_behaviors import exact_score
from scripts.run_node6_rerank_eval import save_cache


def audit_forgotten_values(text: str) -> dict:
    """仅识别这份固定轨迹的常见旧值写法；不是通用语义遗忘过滤器。"""
    value = unicodedata.normalize("NFKC", text).casefold()
    compact = re.sub(r"\s+", "", value)
    date = bool(re.search(r"2027(?:年|[-/.])0?2(?:月|[-/.])0?19(?:日)?", compact))
    code = bool(re.search(r"acl[\s_\-‐‑–—]*7(?!\d)", value))
    return {"old_date_present": date, "old_code_present": code, "passed": not date and not code}


def run(source: Path, live: bool) -> None:
    baseline = json.loads(source.read_text(encoding="utf-8"))
    if (config.memory_history_max_tokens, config.memory_summary_max_tokens, config.memory_compression_trigger,
        config.memory_summary_enabled) != (32768, 4096, "tokens", True):
        raise ValueError("遗忘回归必须使用已启用摘要的32K/4K策略")
    if (baseline["identity"]["model"], baseline["identity"]["base_url"]) != (
        config.zhipu_chat_model, config.zhipu_base_url):
        raise ValueError("回归模型或平台与基线不同")
    if len(baseline["summary_calls"]) != 2:
        raise ValueError("基线必须包含两个实际摘要检查点")
    identity = {"protocol": "memory-forgetting-regression-v1", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_path": str(source.resolve()), "model": config.zhipu_chat_model,
                "base_url": config.zhipu_base_url, "reasoning_effort": "low", "max_answer_tokens": 800,
                "summary_prompt": load_prompt("memory/conversation_summary.md", ("existing_summary", "transcript", "summary_limit")),
                "answer_prompt": load_prompt("memory/behavior_answer_probe.md"),
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                    "scripts/regress_memory_forgetting.py", "memory/session_context.py", "memory/history_store.py",
                    "scripts/benchmark_memory_answers.py", "scripts/evaluate_memory_behaviors.py")}}
    output = source.parent / f"forgetting_{fingerprint(identity)[:12]}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "identity": identity, "summaries": {}, "answers": {}}
    if report["identity"] != identity:
        raise ValueError("遗忘回归缓存身份不一致")
    print(f"report={output}; live={live}; planned_summary_calls=3; planned_answers=7", flush=True)
    if not live:
        return

    def summarize(key: str, messages: list, previous: str) -> SummaryResult:
        serialized = [message_to_dict(m) for m in messages]
        input_id = fingerprint({"messages": serialized, "previous": previous})
        cached = report["summaries"].get(key)
        if cached:
            if cached["input_id"] != input_id:
                raise ValueError(f"{key}摘要回放输入与已付费缓存不一致")
            if cached["status"] != "completed":
                raise RuntimeError(f"{key}此前摘要失败，保留记录，不自动重复付费")
            return SummaryResult(**cached["result"])
        row = {"input_id": input_id, "messages": serialized, "existing_summary": previous}
        started = time.perf_counter()
        try:
            result = _summarize_messages(messages, previous)
        except (RuntimeError, ValueError) as exc:
            row.update(status="failed", error=str(exc), elapsed_seconds=time.perf_counter() - started)
            report["summaries"][key] = row
            save_cache(output, report)
            raise
        audit = audit_forgotten_values(result.text)
        # 首次检查点尚未撤销审批码，不能把合法保留该码判成失败。
        audit["passed"] = not audit["old_date_present"] and (key == "clean_first" or not audit["old_code_present"])
        row.update(status="completed", result=asdict(result), elapsed_seconds=time.perf_counter() - started, audit=audit)
        report["summaries"][key] = row
        save_cache(output, report)
        print(json.dumps({"summary": key, "audit": row["audit"]}, ensure_ascii=False), flush=True)
        return result

    full = messages_from_dict(baseline["contexts"]["full_history"])
    first_count = baseline["first_checkpoint"]["message_count"]
    _, state = build_context(full[:first_count], {}, lambda m, p: summarize("clean_first", m, p))
    context, state = build_context(full, state, lambda m, p: summarize("clean_second", m, p))
    if state["status"] != "ready":
        raise RuntimeError(f"回归上下文构建降级：{state.get('error')}")
    report["state"] = state
    report["context"] = [message_to_dict(m) for m in context]
    save_cache(output, report)
    old_call = baseline["summary_calls"][1]
    summarize("legacy_second", messages_from_dict(old_call["input_messages"]), old_call["existing_summary"])
    for probe in baseline["identity"]["fixture"]["probes"]:
        if probe["id"] in report["answers"]:
            continue
        payload = {"history": [{"role": m.type, "content": m.content} for m in context],
                   "question": probe["question"], "field": probe["field"]}
        row = timed_answer(_build_light_model(), [SystemMessage(content=identity["answer_prompt"]),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
        row["score"] = exact_score(row["text"], probe)
        report["answers"][probe["id"]] = row
        save_cache(output, report)
        print(json.dumps({"case": probe["id"], "status": row["status"], "score": row["score"]}, ensure_ascii=False), flush=True)
        if row["status"] != "completed":
            raise RuntimeError("遗忘回归问答失败，已保存并停止，不自动重试")
    report["summary_audit_note"] = "first检查点尚未请求撤销审批码，只要求撤除日期；second与legacy要求同时撤除日期和审批码。匹配器只作诊断，需人工复核。"
    save_cache(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "storage/evaluations/session_memory_v1/reports/behaviors_fe4fc7ec3865.json")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run(args.source, args.live)
