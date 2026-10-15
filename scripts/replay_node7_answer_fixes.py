"""固定失败题和正常题的历史证据，复测回答修复，不重新检索或生成嵌入。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document

from core import config
from generation.evidence import build_evidence
from generation.model import build_chat_model
from generation.model_pool import AnswerModelPool
from generation.prompts import SYSTEM_PROMPT, USER_PROMPT
from generation.rag_service import RagService
from scripts.run_node6_rerank_eval import save_cache
from scripts.run_node7_prompt_ab import digest, run_answer


def restore_documents(evidence: list[dict]) -> list[Document]:
    """从实际上下文重建文档，要求重新编号后的全文与来源快照完全相同。"""
    docs = [Document(page_content=row["text"], metadata={"chunk_id": row["chunk_id"],
            "document_id": row["document_id"], "source": row["source"],
            "source_locations_json": json.dumps(row["locations"], ensure_ascii=False)}) for row in evidence]
    if [asdict(row) for row in build_evidence(docs)] != evidence:
        raise ValueError("回放证据的顺序、编号或来源与原始快照不符")
    return docs


def build_replays(original: dict, comparison: dict) -> list[dict]:
    """两道失败题各三次，其余正常/缺失/恢复证据题各一次，禁止注入标准答案。"""
    definitions = [("accept-v1-r01", 3), ("accept-v1-x04", 3), ("accept-v1-a04", 1),
                   ("accept-v1-a06", 1), ("accept-v1-a11", 1), ("wb-37-02-01", 1),
                   ("accept-v1-x01", 1), ("accept-v1-r02", 1), ("accept-v1-a05", 1)]
    cases = []
    for key, repeats in definitions:
        row = original["rows"][key]
        if row["status"] != "success":
            raise ValueError(f"历史基线没有成功答案：{key}")
        cases.append({"case_id": key, "question": row["question"], "evidence": row["evidence"],
                      "repeats": repeats, "historical_raw_answer": row["raw_answer"]})
    # 使用此前已提供过的完整上下文作恢复对照；不把标准答案或新挑选的片段塞给模型。
    if comparison["identity"]["system_prompts"]["candidate"] != original["identity"]["system_prompt"]:
        raise ValueError("完整证据对照与端到端基线使用的历史提示词不同")
    fixture = next(row for row in comparison["identity"]["fixtures"] if row["question_id"] == "accept-v1-x04")
    old = comparison["rows"]["candidate/accept-v1-x04"]
    cases.append({"case_id": "accept-v1-x04-complete-context", "question": fixture["question"],
                  "evidence": fixture["evidence"], "repeats": 1, "historical_raw_answer": old["raw_answer"]})
    return cases


class NoLiveVectorStore:
    """仅供固定证据复测；误走真实召回立即报错，防止多调用一次嵌入。"""

    def get_vector_docs(self, query: str, count: int) -> list[Document]:
        raise RuntimeError("回答回放不允许重新召回")

    def get_all_documents(self) -> list[Document]:
        return []

    def count(self) -> int:
        return 0


def main(limit: int = 0) -> None:
    if config.chat_model_name != "qwen3.8-max" or config.evidence_expansion != "none":
        raise ValueError("回答回放固定千问qwen3.8-max且不开邻接扩展，不自动修改当前配置")
    reports = ROOT / "storage/evaluations/zh_public_minibench_v0.1/reports"
    original = json.loads((reports / "node7_end_to_end_40d575b537eb.json").read_text(encoding="utf-8"))
    comparison = json.loads((reports / "node7_prompt_ab_679c28ad301c.json").read_text(encoding="utf-8"))
    cases = build_replays(original, comparison)
    documents = {case["case_id"]: restore_documents(case["evidence"]) for case in cases}
    paths = [ROOT / name for name in ("generation/evidence.py", "generation/rag_service.py",
             "generation/model_pool.py", "generation/prompts.py", "scripts/run_node7_prompt_ab.py")]
    identity = {"protocol": "node7-answer-fix-replay-v1", "model": config.chat_model_name,
                "base_url": config.embedding_base_url, "enable_thinking": False, "fallback": "disabled_for_evaluation",
                "max_tokens": 800, "cases": cases, "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT,
                "historical_system_prompt": original["identity"]["system_prompt"],
                "historical_user_prompt": original["identity"]["user_prompt"],
                "historical_report_sha256": digest(original), "complete_context_report_sha256": digest(comparison),
                "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [*paths, Path(__file__)]}}
    output = reports / f"node7_answer_fix_replay_{digest(identity)[:12]}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"identity": identity, "rows": {}}
    if report["identity"] != identity:
        raise ValueError("回答回放缓存与当前输入不一致")
    model = AnswerModelPool([build_chat_model().members[0]]).bind(max_tokens=800)
    histories = {}
    rag = RagService(vector_service=NoLiveVectorStore(), chat_model=model,
                     history_factory=lambda key: histories.setdefault(key, InMemoryChatMessageHistory()))
    tasks = [(case, repeat) for case in cases for repeat in range(1, case["repeats"] + 1)]
    for case, repeat in tasks[:limit] if limit else tasks:
        key = f"{case['case_id']}/{repeat}"
        if key in report["rows"]:
            print(f"cached={key}", flush=True)
            continue
        report["rows"][key] = run_answer(rag, {"question": case["question"], "documents": documents[case["case_id"]]},
                                          f"answer-fix-{digest(identity)[:12]}-{key}")
        save_cache(output, report)
        print(json.dumps({"case": key, "status": report["rows"][key]["status"], "completed": len(report["rows"]),
                          "expected": len(tasks)}, ensure_ascii=False), flush=True)
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="先执行前N次调用，0为全部14次；已有结果不重复调用")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit不能小于0")
    main(args.limit)
