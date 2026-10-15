"""固定GLM原批次的10块证据，定点复测回答修复与正常/拒答对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.chat_history import InMemoryChatMessageHistory

from generation.prompts import SYSTEM_PROMPT, USER_PROMPT
from generation.evidence import build_evidence, evidence_json
from generation.rag_service import RagService
from generation.table_context import named_cell_evidence_json
from scripts.replay_node7_answer_fixes import NoLiveVectorStore, restore_documents
from scripts.run_node6_rerank_eval import save_cache
from scripts.run_node7_end_to_end import check_failure_limit, evaluation_model
from scripts.run_node7_prompt_ab import digest, run_answer


DEFINITIONS = {
    "ret-te-2010-01": 3, "ret-te-2013-01": 2, "wb-346-01-02": 2,
    "accept-v1-x02": 2, "accept-v1-x01": 1,
    **{f"accept-v1-{suffix}": 1 for suffix in (
        "a04", "a06", "a09", "a10", "a11", "a12", "x03", "x04", "r01", "r02", "r03", "r04")},
}


def build_cases(original: dict) -> list[dict]:
    """只从历史真实上下文恢复，标答仅存档，绝不传给回答链。"""
    if original["identity"].get("provider") != "zhipu" or original["identity"]["retrieval"]["hybrid_final_k"] != 10:
        raise ValueError("本轮只能回放GLM默认10块的原批次")
    cases = []
    for key, repeats in DEFINITIONS.items():
        row = original["rows"].get(key)
        if not row or row["status"] != "success" or len(row["evidence"]) != 10:
            raise ValueError(f"回放题缺少成功的10块原始证据：{key}")
        restore_documents(row["evidence"])
        cases.append({"question_id": key, "question": row["question"], "evidence": row["evidence"],
                      "historical_answer": row["raw_answer"], "repeats": repeats})
    return cases


def main(limit: int = 0, table_view: bool = False) -> None:
    reports = ROOT / "storage/evaluations/zh_public_minibench_v0.1/reports"
    original = json.loads((reports / "node7_end_to_end_3c25d305a9b5.json").read_text(encoding="utf-8"))
    cases = build_cases(original)
    if table_view:
        # 只验证表格读数及四道相关对照，不把呈现实验扩展成新一轮全量调参。
        selected = {"ret-te-2010-01", "accept-v1-a04", "accept-v1-a06", "accept-v1-a09", "accept-v1-r02"}
        cases = [case for case in cases if case["question_id"] in selected]
    renderer = named_cell_evidence_json if table_view else evidence_json
    model_identity, model = evaluation_model("zhipu")
    if model_identity != {name: original["identity"][name] for name in model_identity}:
        raise ValueError("当前GLM供应商、型号、平台或思考设置与原批次不同")
    identity = {"protocol": "node7-glm-answer-fix-v1", **model_identity, "max_tokens": 800,
                "original_report_sha256": digest(original), "cases": cases,
                "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT,
                "table_view": table_view,
                "actual_model_contexts": {case["question_id"]: renderer(build_evidence(restore_documents(case["evidence"])))
                                          for case in cases},
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                    "scripts/replay_node7_glm_fixes.py", "scripts/run_node7_prompt_ab.py",
                    "generation/rag_service.py", "generation/evidence.py", "generation/model_pool.py",
                    "generation/table_context.py")}}
    output = reports / f"node7_glm_answer_fixes_{digest(identity)[:12]}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"identity": identity, "rows": {}}
    if report["identity"] != identity:
        raise ValueError("GLM回放缓存与本次输入不一致")
    histories = {}
    rag = RagService(vector_service=NoLiveVectorStore(), chat_model=model.bind(max_tokens=800),
                     history_factory=lambda key: histories.setdefault(key, InMemoryChatMessageHistory()),
                     evidence_renderer=renderer)
    tasks = [(case, number) for case in cases for number in range(1, case["repeats"] + 1)]
    failures = 0
    for case, number in tasks[:limit] if limit else tasks:
        key = f"{case['question_id']}/{number}"
        if key in report["rows"]:
            print(f"cached={key}", flush=True)
            continue
        fixture = {"question": {"query": case["question"]["query"]},
                   "documents": restore_documents(case["evidence"])}
        report["rows"][key] = run_answer(rag, fixture, f"glm-fix-{digest(identity)[:12]}-{key}")
        save_cache(output, report)
        print(json.dumps({"case": key, "status": report["rows"][key]["status"], "completed": len(report["rows"]),
                          "expected": len(tasks)}, ensure_ascii=False), flush=True)
        failures = check_failure_limit(report["rows"][key], failures)
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0为全部22次调用；已完成结果不重复调用")
    parser.add_argument("--table-view", action="store_true", help="仅做7次表格呈现实验，不修改正式回答默认")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit不能小于0")
    main(args.limit, args.table_view)
