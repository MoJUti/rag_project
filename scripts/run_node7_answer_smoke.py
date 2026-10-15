"""固定已有候选，检查真实模型的通用回答、引用和资料不足处理。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.outputs import LLMResult

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config
from generation.evidence import audit_citations, build_evidence
from generation.model import build_chat_model
from generation.prompts import answer_prompt_sha256
from generation.rag_service import RagService
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from ingestion.storage import StorageLayout
from retrieval.pipeline import RetrievalPipeline
from scripts.import_cached_chroma import OfflineEmbeddings
from scripts.run_chunking_benchmark import read_jsonl
from scripts.run_node6_rerank_eval import save_cache


class UsageRecorder(BaseCallbackHandler):
    """只记录本次回答的原文、结束原因与接口用量，不记录密钥。"""

    def __init__(self) -> None:
        self.body = ""
        self.info: dict = {}

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        generation = response.generations[0][0]
        self.body = generation.text
        self.info = generation.generation_info or {}


def choose_fixture(question: dict, ranked: list[str], documents: dict, mode: str) -> list:
    """对照实验才注入指定原文；与真实检索候选明确区分，不混算成绩。"""
    if mode == "scope_controlled" and not question["answerable"]:
        filename = Path(question["scope_source_path"]).name
        selected = [doc for doc in documents.values() if Path(doc.metadata.get("source", "")).name == filename]
        if not selected or len(selected) > 6:
            raise ValueError(f"指定来源对照需要1至6个完整切块：{question['question_id']}")
        return selected
    return [documents[chunk_id] for chunk_id in ranked]


def scope_is_present(question: dict, docs: list) -> bool | None:
    """只判断拒答范围是否进入上下文，不把文档命中当成正确拒答。"""
    path = question.get("scope_source_path")
    if not path:
        return None
    return any(Path(doc.metadata.get("source", "")).name == Path(path).name for doc in docs)


def main(limit: int, evidence_mode: str = "cached", question_ids: list[str] | None = None) -> None:
    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    question_path = ROOT / "data/eval/zh_public_minibench_v0.1/annotations/acceptance_questions_v1.jsonl"
    questions = {row["question_id"]: row for row in read_jsonl(question_path)}
    acceptance = json.loads((evaluation.reports / "node6_acceptance_v1_report.json").read_text(encoding="utf-8"))
    ranked = {row["question_id"]: row["ranked_chunk_ids"]
              for row in acceptance["methods"]["hybrid_mixed"]["questions"]}
    ranked.update({row["question_id"]: row["ranked_chunk_ids"] for row in acceptance["refusal_cases"]})
    vector = VectorStoreService(adapter=ChromaVectorStoreAdapter(OfflineEmbeddings()))
    documents = {doc.metadata["chunk_id"]: doc for doc in vector.get_all_documents()}
    selected = [f"accept-v1-{suffix}" for suffix in ("a04", "a06", "x02", "r01", "r02", "r03", "r04")]
    if question_ids:
        if set(question_ids) - set(selected):
            raise ValueError("指定问题不属于本轮7题套件")
        selected = [key for key in selected if key in question_ids]
    fixtures = {key: choose_fixture(questions[key], ranked[key], documents, evidence_mode) for key in selected}
    identity = {"model": config.chat_model_name, "base_url": config.embedding_base_url,
                "fallback_provider": config.chat_fallback_provider,
                "fallback_model": config.zhipu_chat_model if config.chat_fallback_provider == "zhipu" else None,
                "fallback_base_url": config.zhipu_base_url if config.chat_fallback_provider == "zhipu" else None,
                "fallback_reasoning_effort": "low" if config.chat_fallback_provider == "zhipu" else None,
                "enable_thinking": False, "max_output_tokens": 800,
                "evidence_mode": evidence_mode,
                "prompt_sha256": answer_prompt_sha256(),
                "questions_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
                "fixture_sha256": hashlib.sha256(json.dumps({
                    key: [(doc.page_content, doc.metadata) for doc in docs] for key, docs in fixtures.items()
                }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    output = evaluation.reports / f"node7_answer_smoke_{fingerprint}.json"
    cached = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"identity": identity, "rows": {}}
    if cached["identity"] != identity:
        raise ValueError("回答验证缓存与当前配置不一致")
    model = build_chat_model().bind(max_tokens=800)
    histories = {}
    rag = RagService(vector_service=vector, chat_model=model,
                     history_factory=lambda key: histories.setdefault(key, InMemoryChatMessageHistory()))
    for key in selected[:limit] if limit else selected:
        if key in cached["rows"] and cached["rows"][key].get("status") == "success":
            print(f"cached={key}", flush=True)
            continue
        question = questions[key]
        docs = fixtures[key]

        class FixedCandidates:
            def get_candidates(self, query: str, count: int) -> list:
                return docs[:count]

        # 固定证据隔离检索波动，回答链、模型、流式显示和引用逻辑仍走正式代码。
        rag.retrieval_pipeline = RetrievalPipeline(FixedCandidates())
        recorder = UsageRecorder()
        run_config = config.build_session_config(key)
        run_config["callbacks"] = [recorder]
        started = time.perf_counter()
        try:
            answer = "".join(rag.chain.stream({"input": question["query"]}, run_config))
        except (RuntimeError, ValueError) as exc:
            cached["rows"][key] = {"status": "failed", "stage": "generation",
                                   "error": str(exc), "review_status": "not_generated"}
            save_cache(output, cached)
            raise
        audit = audit_citations(recorder.body, build_evidence(docs))
        cached["rows"][key] = {"status": "success", "query": question["query"], "gold_answer": question["gold_answer"],
                               "answerable": question["answerable"], "raw_answer": recorder.body,
                               "evidence": [asdict(item) for item in build_evidence(docs)],
                               "scope_document_present": scope_is_present(question, docs),
                               "generation_complete": recorder.info.get("finish_reason") == "stop",
                               "display_answer": answer, "citation_audit": {
                                   "cited_ids": audit.cited_ids, "unknown_ids": audit.unknown_ids,
                                   "has_citations": audit.has_citations},
                               "model_info": recorder.info,
                               "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                               "review_status": "awaiting_source_based_review"}
        save_cache(output, cached)
        print(json.dumps({"question_id": key, "usage": recorder.info.get("token_usage"),
                          "unknown_citations": audit.unknown_ids,
                          "finish_reason": recorder.info.get("finish_reason")}, ensure_ascii=False), flush=True)
    print(f"report={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="先运行前N题，0表示全部7题")
    parser.add_argument("--evidence-mode", choices=("cached", "scope_controlled"), default="cached",
                        help="cached沿用已有召回；scope_controlled为拒答题提供指定原文，仅做诊断")
    parser.add_argument("--question-id", action="append", help="只运行指定套件题，可重复传入")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit不能小于0")
    main(args.limit, args.evidence_mode, args.question_id)
