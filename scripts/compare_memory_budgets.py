"""固定合成长历史比较容量与摘要预算，默认只离线预检，显式--live才调用GLM。"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, message_to_dict
from core import config
from memory.history_store import _build_light_model, _build_summary_prompt
from memory.session_context import SummaryResult, build_context, estimate_tokens, estimate_text_tokens, fingerprint
from scripts.run_node6_rerank_eval import save_cache


def make_trajectory() -> tuple[list, list[dict]]:
    """原始记录含独立预算事实和低信息量讨论，所有试验共用同一份合成数据。"""
    messages, probes = [], []
    for number in range(1, 49):
        group, amount = f"P{number:03d}", 7000 + number * 17
        text = f"确认资料组 {group}：预算 {amount} 元；交付日2027年11月{number % 28 + 1}日；禁止将资料外发。\n"
        details = []
        while estimate_tokens([HumanMessage(content=text + "\n".join(details))]) < 1450:
            index = len(details) + 1
            details.append(f"第{number}轮讨论记录{index}：评审先检查目录和命名，核对版本来源，记录待处理事项；这是流程举例，不改变本轮确认的预算、日期或禁止事项。")
        messages.extend([HumanMessage(content=text + "\n".join(details)), AIMessage(content="收到，按本轮用户明确条件继续；流程举例不构成新约束。")])
        probes.append({"group": group, "amount": amount, "round": number})
    return messages, probes


def budget_association_ids(text: str, probes: list[dict]) -> list[str]:
    """只作诊断性字面关联，不把字符串命中当作语义正确或问答准确率。"""
    value = re.sub(r"[,，*`\s]", "", text)
    hits = []
    for probe in probes:
        pattern = re.escape(probe["group"]) + r"(?:(?!P\d{3}).){0,80}?(?<!\d)" + str(probe["amount"]) + r"(?!\d)"
        if re.search(pattern, value):
            hits.append(probe["group"])
    return hits


def main(live: bool) -> None:
    messages, probes = make_trajectory()
    profiles = [*[
        (f"history{budget}_summary{limit}", budget, limit, "tokens")
        for budget in (16384, 32768, 65536) for limit in (2048, 4096)],
        ("baseline_4000_chars1500", 4000, None, "rounds")]
    identity = {"protocol": "memory-budget-single-checkpoint-v1", "synthetic": True,
                "messages": [message_to_dict(m) for m in messages], "probes": probes, "profiles": profiles,
                "model": config.zhipu_chat_model, "base_url": config.zhipu_base_url, "reasoning_effort": "low",
                "prompt_template": (ROOT / "prompts/memory/conversation_summary.md").read_text(encoding="utf-8"),
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                    "scripts/compare_memory_budgets.py", "memory/session_context.py", "memory/history_store.py")}}
    output = ROOT / f"storage/evaluations/session_memory_v1/reports/budgets_{fingerprint(identity)[:12]}{'_live' if live else '_offline'}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "identity": identity, "source_estimated_tokens": estimate_tokens(messages), "rows": {},
        "quality_note": "单个合成长历史检查点；字面关联是定位指标，不是语义或回答正确率"}
    if fingerprint(report["identity"]) != fingerprint(identity):
        raise ValueError("预算实验缓存与本轮身份不一致")
    for name, budget, limit, mode in profiles:
        if name in report["rows"]:
            print(f"cached={name}", flush=True)
            continue
        calls = []

        def summarize(older: list, existing: str) -> SummaryResult:
            prompt = _build_summary_prompt(existing, "\n".join(
                f"[{'用户' if m.type == 'human' else '助手'}] {m.content}" for m in older))
            call = {"input_message_count": len(older), "input_estimated_tokens": estimate_tokens(older),
                    "actual_prompt": prompt, "live": live}
            calls.append(call)
            if live:
                response = _build_light_model().invoke([HumanMessage(content=prompt)])
                result = SummaryResult(str(response.content).strip(), dict(response.response_metadata))
            else:
                result = SummaryResult("离线摘要替身，仅用于验证预算与输入选择，不代表记住任何事实。")
            call["result"] = asdict(result)
            return result

        settings = {"memory_history_max_tokens": budget, "memory_summary_max_tokens": limit,
                    "memory_compression_trigger": mode, "memory_summary_max_chars": 1500,
                    "memory_keep_recent_rounds": 3, "memory_summary_trigger_rounds": 5, "memory_summary_enabled": True}
        with patch.multiple(config, **settings):
            context, state = build_context(messages, {}, summarize)
        text = "\n".join(str(m.content) for m in context)
        raw = [m for m in context if m.type != "system"]
        row = {"settings": settings, "state": state, "calls": calls,
               "context": [message_to_dict(m) for m in context],
               "retained_raw_rounds": sum(m.type == "human" for m in raw),
               "retained_raw_estimated_tokens": estimate_tokens(raw),
               "summary_estimated_tokens": estimate_text_tokens(state["summary"]),
               "diagnostic_budget_associations": budget_association_ids(text, probes) if live else None}
        report["rows"][name] = row
        save_cache(output, report)
        print(json.dumps({"profile": name, "status": state["status"], "source_tokens": report["source_estimated_tokens"],
                          "context_tokens": state["context_estimated_tokens"], "raw_rounds": row["retained_raw_rounds"],
                          "summary_tokens": row["summary_estimated_tokens"],
                          "budget_associations": len(row["diagnostic_budget_associations"]) if live else None}, ensure_ascii=False), flush=True)
        if live and state.get("error", "") and state["error"].startswith("摘要失败"):
            raise RuntimeError(f"{name}降级，已保存原始输出并停止；不继续付费批次")
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    main(parser.parse_args().live)
