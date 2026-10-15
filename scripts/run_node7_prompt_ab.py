"""固定证据比较两版回答提示词；结果逐题保存，语义评分由独立复核记录给出。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from core import config
from core.prompt_loader import render_prompt
from generation.evidence import audit_citations, build_evidence
from generation.model import build_chat_model
from generation.model_pool import AnswerModelPool
from generation.prompts import answer_prompt_sha256, build_answer_prompt
from generation.rag_service import RagService
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from ingestion.storage import StorageLayout
from retrieval.pipeline import RetrievalPipeline
from scripts.import_cached_chroma import OfflineEmbeddings
from scripts.run_chunking_benchmark import read_jsonl
from scripts.run_node6_rerank_eval import save_cache
from scripts.run_node7_answer_smoke import UsageRecorder, scope_is_present


def digest(value: object) -> str:
    """使用排序后的稳定JSON计算内容身份，不依赖修改时间。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_cases(spec: dict, questions: dict, rankings: dict, documents: dict) -> list[dict]:
    """仅使用冻结配置提供的证据；合成题明确标记，不注入真实题标准答案。"""
    cases = []
    seen = set()
    for item in spec["cases"]:
        key = item["question_id"]
        if key in seen:
            raise ValueError(f"对比题号重复：{key}")
        seen.add(key)
        if item["group"] == "synthetic_boundary":
            docs = [Document(page_content=entry["text"], metadata={
                "source": entry["source"], "chunk_id": f"{key}-{index}",
                "document_id": f"synthetic-{key}-{index}", "source_locations_json": "[]",
            }) for index, entry in enumerate(item["documents"])]
            question = {"query": item["query"], "gold_answer": item["gold_answer"]}
        else:
            question = questions[key]
            docs = [documents[chunk_id] for chunk_id in rankings[key]]
        if not docs or len(docs) > 6:
            raise ValueError(f"固定上下文需要1至6个切块：{key}")
        cases.append({"question_id": key, "group": item["group"], "question": question,
                      "documents": docs, "evidence": [asdict(entry) for entry in build_evidence(docs)]})
    return cases


def experiment_identity(spec: dict, rubric: str, systems: dict[str, str], cases: list[dict]) -> dict:
    """记录两版正文及同一份上下文；评审标准也参与缓存版本。"""
    return {"spec": spec, "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
            "model": config.chat_model_name, "base_url": config.embedding_base_url,
            "enable_thinking": False, "fallback": "disabled_for_comparison",
            "max_tokens": spec["max_tokens"], "system_prompts": systems,
            "prompt_sha256": {name: answer_prompt_sha256(text) for name, text in systems.items()},
            "fixtures": [{"question_id": row["question_id"], "group": row["group"],
                          "question": row["question"], "evidence": row["evidence"]} for row in cases]}


class FixedRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def get_candidates(self, query: str, count: int) -> list[Document]:
        return self.documents[:count]


def run_answer(rag: RagService, case: dict, session_id: str) -> dict:
    """调用正式回答链，保存原文与可复核证据，不自动宣布答案正确。"""
    rag.retrieval_pipeline = RetrievalPipeline(FixedRetriever(case["documents"]))
    recorder = UsageRecorder()
    run_config = config.build_session_config(session_id)
    run_config["callbacks"] = [recorder]
    started = time.perf_counter()
    try:
        answer = "".join(rag.chain.stream({"input": case["question"]["query"]}, run_config))
    except (RuntimeError, ValueError) as exc:
        return {"status": "failed", "error": str(exc), "model_info": recorder.info,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
    audit = audit_citations(recorder.body, build_evidence(case["documents"]))
    return {"status": "success", "raw_answer": recorder.body, "display_answer": answer,
            "model_info": recorder.info,
            "scope_document_present": scope_is_present(case["question"], case["documents"]),
            "citation_audit": asdict(audit),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "review_status": "awaiting_source_based_review"}


def load_comparison_prompts(baseline_report: Path) -> dict[str, str]:
    """从历史评测恢复旧正文；当前提示词只维护一份，不复制版本文件。"""
    try:
        report = json.loads(baseline_report.read_text(encoding="utf-8"))
        baseline = report["identity"]["system_prompts"]["baseline"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"读取历史基线提示词失败：{baseline_report}：{exc}") from exc
    if not isinstance(baseline, str) or not baseline.strip():
        raise ValueError(f"历史基线提示词为空或类型错误：{baseline_report}")
    return {"baseline": baseline, "candidate": render_prompt("generation/knowledge_answer.md", {})}


def main(baseline_report: Path, limit: int = 0) -> None:
    systems = load_comparison_prompts(baseline_report)
    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    annotations = ROOT / "data/eval/zh_public_minibench_v0.1/annotations"
    spec = json.loads((annotations / "node7_prompt_ab_v1.json").read_text(encoding="utf-8"))
    if config.chat_model_name != spec["model"]:
        raise ValueError(f"实验固定模型为{spec['model']}，当前配置不符")
    questions = {row["question_id"]: row for row in read_jsonl(annotations / "acceptance_questions_v1.jsonl")}
    acceptance = json.loads((evaluation.reports / "node6_acceptance_v1_report.json").read_text(encoding="utf-8"))
    rankings = {row["question_id"]: row["ranked_chunk_ids"]
                for row in acceptance["methods"]["hybrid_mixed"]["questions"]}
    rankings.update({row["question_id"]: row["ranked_chunk_ids"] for row in acceptance["refusal_cases"]})
    vector = VectorStoreService(adapter=ChromaVectorStoreAdapter(OfflineEmbeddings()))
    documents = {doc.metadata["chunk_id"]: doc for doc in vector.get_all_documents()}
    cases = build_cases(spec, questions, rankings, documents)
    identity = experiment_identity(spec, (ROOT / "docs/节点7提示词评分规则.md").read_text(encoding="utf-8"), systems, cases)
    output = evaluation.reports / f"node7_prompt_ab_{digest(identity)[:12]}.json"
    cache = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"identity": identity, "rows": {}}
    if cache["identity"] != identity:
        raise ValueError("提示词对比缓存与当前实验不一致")
    # 主模型故障就记录失败；不允许通过备用模型获得本应属于千问的实验成绩。
    model = AnswerModelPool([build_chat_model().members[0]]).bind(max_tokens=spec["max_tokens"])
    histories = {}
    rag = RagService(vector_service=vector, chat_model=model,
                     history_factory=lambda key: histories.setdefault(key, InMemoryChatMessageHistory()))
    for index, case in enumerate(cases[:limit] if limit else cases):
        versions = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for version in versions:
            key = f"{version}/{case['question_id']}"
            if key in cache["rows"]:
                print(f"cached={key}", flush=True)
                continue
            rag.prompt_template = build_answer_prompt(systems[version])
            rag.chain = rag._build_chain()
            cache["rows"][key] = run_answer(rag, case, f"prompt-ab-{digest(identity)[:12]}-{key}")
            save_cache(output, cache)
            row = cache["rows"][key]
            print(json.dumps({"case": key, "status": row["status"],
                              "usage": row["model_info"].get("token_usage")}, ensure_ascii=False), flush=True)
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True, help="包含旧系统提示词正文的历史评测JSON")
    parser.add_argument("--limit", type=int, default=0, help="先跑前N题的两版回答，0为完整14题")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit不能小于0")
    main(args.baseline_report, args.limit)
