"""使用当前检索配置运行真实问答链，逐题保存证据、阶段记录和待复核答案。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.chat_history import InMemoryChatMessageHistory

from core import config
from generation.evidence import audit_citations, build_evidence
from generation.model import build_chat_model
from generation.model_pool import AnswerModelPool, CompatibleChatModel
from generation.prompts import SYSTEM_PROMPT, USER_PROMPT, NO_EVIDENCE_ANSWER
from generation.rag_service import RagService
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from ingestion.document_model import stable_document_id
from ingestion.storage import EvaluationStorageLayout, StorageLayout
from retrieval.embedding_cache import EmbeddingCache
from retrieval.pipeline import RetrievalResult
from scripts.evaluate_embedded_retrieval import evaluate_ranking, load_chunk_catalog
from scripts.evaluate_node6_acceptance import score_question
from scripts.run_chunking_benchmark import read_jsonl
from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings
from scripts.run_node6_rerank_eval import save_cache
from scripts.run_node7_answer_smoke import UsageRecorder, scope_is_present
from scripts.run_node7_prompt_ab import digest


class ObservedRagService(RagService):
    """仅观察正式链实际调用的那次检索，不替换候选，也不重复召回。"""

    last_retrieval: RetrievalResult | None = None

    def retrieve_with_trace(self, query: str) -> RetrievalResult:
        result = super().retrieve_with_trace(query)
        self.last_retrieval = result
        return result


def retrieval_metrics(question: dict, ids: list[str], catalog: dict, mapping: dict,
                      final_k: int = 6) -> dict | None:
    """沿用两套评分规则，指标名跟随截取数量；默认参数兼容旧6块记录。"""
    if not question["answerable"]:
        return None
    if question["evaluation_suite"] == "acceptance_23":
        return {key.replace("_at_6", f"_at_{final_k}"): value
                for key, value in score_question(question, ids, catalog, mapping).items()}
    return evaluate_ranking(question, ids, catalog, mapping, (final_k,))


def run_case(rag: ObservedRagService, question: dict, catalog: dict, mapping: dict, session_id: str) -> dict:
    """每题独立历史；失败也保留实际检索快照，不自动给语义准确率。"""
    rag.last_retrieval = None
    recorder = UsageRecorder()
    run_config = config.build_session_config(session_id)
    run_config["callbacks"] = [recorder]
    started = time.perf_counter()
    row = {"question": question, "review_status": "awaiting_source_based_review"}
    try:
        displayed = "".join(rag.chain.stream({"input": question["query"]}, run_config))
        row.update(status="success", display_answer=displayed,
                   raw_answer=recorder.body if recorder.body else displayed)
    except (RuntimeError, ValueError) as exc:
        row.update(status="failed", stage="generation" if rag.last_retrieval is not None else "retrieval",
                   error=str(exc), review_status="not_generated")
    row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    row["model_info"] = recorder.info
    result = rag.last_retrieval
    if result is None:
        return row
    evidence = build_evidence(result.documents)
    ids = [entry.chunk_id for entry in evidence]
    if set(ids) - set(catalog):
        raise ValueError(f"检索返回未冻结的切块：{question['question_id']}")
    row.update(evidence=[asdict(entry) for entry in evidence],
               ranked_chunk_ids=ids, trace=[asdict(stage) for stage in result.trace],
               retrieval_metrics=retrieval_metrics(question, ids, catalog, mapping, rag.retrieval_pipeline.final_k),
               scope_document_present=scope_is_present(question, result.documents))
    if row["status"] == "success":
        row["citation_audit"] = asdict(audit_citations(row["raw_answer"], evidence))
        row["model_called"] = bool(recorder.info)
        row["generation_complete"] = (recorder.info.get("finish_reason") == "stop" if evidence
                                      else row["raw_answer"] == NO_EVIDENCE_ANSWER)
    return row


def summarize_run(rows: dict[str, dict], expected_count: int) -> dict:
    """自动汇总仅含运行、召回和编号检查；不能把成功请求等同正确答案。"""
    successful = [row for row in rows.values() if row["status"] == "success"]
    summary = {"expected": expected_count, "attempted": len(rows), "successful": len(successful),
               "failed": len(rows) - len(successful), "semantic_review": "pending",
               "unknown_citation_answers": sum(bool(row["citation_audit"]["unknown_ids"]) for row in successful),
               "nonstandard_citation_answers": sum(bool(row["citation_audit"].get("nonstandard_ids")) for row in successful),
               "answers_without_citations": sum(not row["citation_audit"]["has_citations"] for row in successful),
               "generation_incomplete": sum(not row["generation_complete"] for row in successful),
               "retrieval_by_suite": {}}
    for suite in sorted({row["question"]["evaluation_suite"] for row in rows.values()}):
        measured = [row["retrieval_metrics"] for row in rows.values()
                    if row["question"]["evaluation_suite"] == suite and row.get("retrieval_metrics") is not None]
        summary["retrieval_by_suite"][suite] = {"answerable_count": len(measured), "means": {
            key: round(statistics.fmean(row[key] for row in measured), 6)
            for key in measured[0] if key != "first_relevant_rank"
        } if measured else {}}
    usage = [row["model_info"].get("token_usage", {}) for row in rows.values()]
    summary["answer_tokens"] = {field: sum(row.get(field, 0) for row in usage)
                               for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
    refusals = [row for row in rows.values() if not row["question"]["answerable"]]
    summary["refusal_scope"] = {"attempted": len(refusals), "target_present": sum(
        row.get("scope_document_present") is True for row in refusals)}
    return summary


def load_questions_and_vectors(bundle: Path, evaluation: EvaluationStorageLayout) -> tuple[list[dict], dict]:
    """验证冻结题目及向量缓存，不悄悄生成或覆盖问题向量。"""
    annotations = bundle / "annotations"
    locked = json.loads((annotations / "acceptance_v1.lock.json").read_text(encoding="utf-8"))
    if hashlib.sha256((annotations / "acceptance_questions_v1.jsonl").read_bytes()).hexdigest() != locked["questions_sha256"]:
        raise ValueError("新增验收题与冻结指纹不一致")
    suites = (("retrieval_55", "retrieval_questions.jsonl", "question_embeddings.sqlite3"),
              ("acceptance_23", "acceptance_questions_v1.jsonl", "acceptance_v1_embeddings.sqlite3"))
    questions, vectors = [], {}
    seen = set()
    for suite, filename, cache_name in suites:
        cache_path = evaluation.embeddings / "text-embedding-v4-2048" / cache_name
        if not cache_path.is_file():
            raise ValueError(f"问题向量缓存不存在：{cache_path}")
        with EmbeddingCache(cache_path) as cache:
            for original in read_jsonl(annotations / filename):
                row = {**original, "evaluation_suite": suite}
                key, query = row["question_id"], row["query"]
                if key in seen:
                    raise ValueError(f"评测题号重复：{key}")
                if not cache.has_valid(key, query, "text-embedding-v4", 2048):
                    raise ValueError(f"问题向量缓存不匹配：{key}")
                vectors[query] = cache.get(key, "text-embedding-v4", 2048).vector.tolist()
                seen.add(key)
                questions.append(row)
    return questions, vectors


def check_failure_limit(row: dict, previous: int, limit: int = 3) -> int:
    """连续失败达到上限后停止整批，避免额度或配置错误引发无意义请求。"""
    count = previous + 1 if row["status"] == "failed" else 0
    if count >= limit:
        raise RuntimeError(f"端到端批次连续失败{count}题，已保存结果并停止：{row.get('error', '未知错误')}")
    return count


def evaluation_model(provider: str) -> tuple[dict, AnswerModelPool]:
    """评测显式指定一家模型，不改变生产优先级，也不自动备用切换。"""
    if provider == "qwen":
        identity = {"provider": "qwen", "model": config.chat_model_name,
                    "base_url": config.embedding_base_url, "enable_thinking": False}
        return identity, AnswerModelPool([build_chat_model().members[0]])
    if provider == "zhipu":
        model = CompatibleChatModel(config.zhipu_chat_model, config.zhipu_base_url,
                                    os.getenv("ZHIPU_API_KEY", "").strip(), provider="zhipu")
        identity = {"provider": "zhipu", "model": config.zhipu_chat_model,
                    "base_url": config.zhipu_base_url, "enable_thinking": True, "reasoning_effort": "low"}
        return identity, AnswerModelPool([("zhipu", lambda: model)])
    raise ValueError(f"不支持的评测供应商：{provider}")


def main(limit: int = 0, retry_failed: bool = False, provider: str = "qwen") -> None:
    if (config.vector_store_backend, config.embedding_model_name, config.embedding_dimensions, config.hybrid_final_k) != (
        "chroma", "text-embedding-v4", 2048, 10,
    ):
        raise ValueError("本轮要求当前Chroma、text-embedding-v4/2048、最终10块证据；不自动修改配置")
    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    bundle = ROOT / "data/eval/zh_public_minibench_v0.1"
    questions, vectors = load_questions_and_vectors(bundle, evaluation)
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    vector = VectorStoreService(adapter=ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors)))
    actual = {doc.metadata["chunk_id"]: {"document_id": doc.metadata["document_id"], "text": doc.page_content}
              for doc in vector.get_all_documents()}
    if actual != catalog or vector.count() != len(catalog):
        raise ValueError("当前库的切块ID、正文或文档ID与冻结目录不一致")
    mapping = {row["item_id"]: stable_document_id(row["sha256"]) for row in read_jsonl(bundle / "manifest.jsonl")}
    keys = ("collection_name", "chroma_mode", "persist_directory", "chroma_host", "chroma_port",
            "chroma_ssl", "chroma_hnsw_ef_search", "retrieval_strategy", "bm25_tokenizer", "hybrid_vector_k",
            "hybrid_bm25_k", "hybrid_final_k", "hybrid_rrf_k", "rerank_strategy", "rerank_candidate_k",
            "evidence_expansion", "evidence_neighbor_radius", "evidence_extra_tokens", "evidence_chunks_directory")
    code_paths = [*sorted((ROOT / "generation").glob("*.py")), *sorted((ROOT / "retrieval").glob("*.py")),
                  ROOT / "infra/vector_stores/chroma.py", ROOT / "scripts/evaluate_node6_acceptance.py",
                  ROOT / "scripts/evaluate_embedded_retrieval.py", Path(__file__)]
    model_identity, selected_model = evaluation_model(provider)
    identity = {"protocol": "node7-e2e-v2-k10", "questions": questions, "catalog_sha256": digest(catalog),
                "query_vectors_sha256": digest(vectors), "retrieval": {key: getattr(config, key) for key in keys},
                "system_prompt": SYSTEM_PROMPT, "user_prompt": USER_PROMPT, **model_identity,
                "fallback": "disabled_for_evaluation",
                "max_tokens": 800, "embedding": "cached-text-embedding-v4-2048",
                "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in code_paths}}
    fingerprint = digest(identity)[:12]
    output = evaluation.reports / f"node7_end_to_end_{fingerprint}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"identity": identity, "rows": {}}
    if report["identity"] != identity:
        raise ValueError("端到端评测缓存与当前输入不一致")
    model = selected_model.bind(max_tokens=800)
    histories = {}
    rag = ObservedRagService(vector_service=vector, chat_model=model,
                             history_factory=lambda key: histories.setdefault(key, InMemoryChatMessageHistory()))
    consecutive_failures = 0
    for question in questions[:limit] if limit else questions:
        key = question["question_id"]
        if key in report["rows"]:
            previous = report["rows"][key]
            if previous["status"] != "failed" or not retry_failed:
                print(f"cached={key}", flush=True)
                continue
            # 显式重试失败题时保留旧记录；已成功题永不自动重复付费调用。
            report.setdefault("previous_attempts", {}).setdefault(key, []).append(previous)
        report["rows"][key] = run_case(rag, question, catalog, mapping, f"e2e-{fingerprint}-{key}")
        report["summary"] = summarize_run(report["rows"], len(questions))
        save_cache(output, report)
        print(json.dumps({"question_id": key, "status": report["rows"][key]["status"],
                          "completed": len(report["rows"]), "expected": len(questions)}, ensure_ascii=False), flush=True)
        consecutive_failures = check_failure_limit(report["rows"][key], consecutive_failures)
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="先运行前N题；0为全部，已有结果不重复调用")
    parser.add_argument("--retry-failed", action="store_true", help="显式重试失败题并保留旧记录；已成功题仍跳过")
    parser.add_argument("--provider", choices=("qwen", "zhipu"), default="qwen", help="显式选择本轮模型，不改变生产默认")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit不能小于0")
    main(args.limit, args.retry_failed, args.provider)
