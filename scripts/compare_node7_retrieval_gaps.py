"""不调用模型，比较固定语料上的分词、邻接扩展与可验证的历史重排。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from ingestion.chunk_models import KnowledgeChunk
from ingestion.chunking import TokenCounter
from ingestion.document_model import stable_document_id
from ingestion.storage import StorageLayout
from retrieval.embedding_cache import EmbeddingCache
from retrieval.evidence_expansion import NeighborhoodExpander
from retrieval.hybrid_retriever import HybridRetrieverService
from scripts.audit_node7_evidence import audit, match_requirement
from scripts.evaluate_embedded_retrieval import evaluate_ranking, load_chunk_catalog
from scripts.evaluate_evidence_expansion import original_context_score
from scripts.evaluate_node6_acceptance import score_question
from scripts.run_chunking_benchmark import read_jsonl
from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings
from scripts.run_node6_rerank_eval import save_cache
from scripts.run_node7_prompt_ab import digest


def verified_rerank(row: dict, query: str, pool: list[str]) -> list[str]:
    """只接受相同查询和有序候选上的完整重排，不能拿旧前十名拼接新结果。"""
    stages = {stage["stage"]: stage for stage in row["trace"]}
    if not {"query", "recall", "rerank"} <= stages.keys():
        raise ValueError("重排缓存缺少查询、输入候选或完整重排阶段")
    if stages["query"]["queries"] != [query] or stages["recall"]["chunk_ids"] != pool:
        raise ValueError("重排缓存的查询或有序候选不一致")
    ids = stages["rerank"]["chunk_ids"]
    if len(ids) != len(pool) or len(set(ids)) != len(ids) or set(ids) != set(pool):
        raise ValueError("重排缓存不是完整候选的无重复排列")
    return ids


def measure_context(question: dict, ids: list[str], seeds: list[str], catalog: dict,
                    mapping: dict, tokens: dict, requirements: list[dict]) -> dict:
    """原评测协议与诊断原文匹配分开记录，扩展上下文不冒充前六名。"""
    if len(ids) != len(set(ids)) or set(ids) - catalog.keys() or not set(seeds) <= set(ids):
        raise ValueError(f"上下文切块重复、未知或丢失种子：{question['question_id']}")
    if question["evaluation_suite"] == "retrieval_55":
        scores = original_context_score(question, ids, catalog, mapping)
        document_recall = evaluate_ranking(question, ids, catalog, mapping, (6,))["recall_at_6"] if ids == seeds else None
    else:
        metrics = score_question(question, ids, catalog, mapping)
        scores = {"evidence_recall": metrics["evidence_recall_at_6"],
                  "complete_evidence": metrics["complete_evidence_at_6"]}
        document_recall = metrics["document_recall_at_6"]
    documents = [{"chunk_id": key, **catalog[key]} for key in ids]
    hits = {item["id"]: bool(match_requirement(item, documents, mapping)) for item in requirements}
    return {"chunk_ids": ids, "block_count": len(ids),
            "total_estimated_tokens": sum(tokens[key] for key in ids),
            "added_estimated_tokens": sum(tokens[key] for key in ids if key not in seeds),
            "document_recall_seed6": document_recall if ids == seeds else None,
            "protocol_evidence": scores, "diagnostic_requirements": hits,
            "diagnostic_complete": all(hits.values()) if hits else None}


def main() -> None:
    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    bundle = ROOT / "data/eval/zh_public_minibench_v0.1"
    reports = evaluation.reports
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    original_path = reports / "node7_end_to_end_40d575b537eb.json"
    original = read(original_path)
    specification = read(bundle / "annotations/node7_evidence_audit_v1.json")
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    mapping = {row["item_id"]: stable_document_id(row["sha256"])
               for row in read_jsonl(bundle / "manifest.jsonl")}
    diagnostic = audit(original, specification, catalog, mapping)
    gaps = [key for key, row in diagnostic["evidence_rows"].items() if row["diagnosis"] == "retrieval_gap"]
    questions = [row["question"] for row in original["rows"].values() if row["question"]["answerable"]]
    if len(gaps) != 8 or len(questions) != 74:
        raise ValueError("冻结实验应包含8道漏证据题和74道可回答回归题")
    if (config.embedding_model_name, config.embedding_dimensions, config.collection_name, config.chroma_mode) != (
        "text-embedding-v4", 2048, "rag_text_embedding_v4_2048", "local",
    ) or config.retrieval_source_filter:
        raise ValueError("当前向量模型、集合、连接方式或来源过滤与冻结实验不一致")
    vectors = {}
    for suite, filename in (("retrieval_55", "question_embeddings.sqlite3"),
                            ("acceptance_23", "acceptance_v1_embeddings.sqlite3")):
        cache_path = evaluation.embeddings / "text-embedding-v4-2048" / filename
        if not cache_path.is_file():
            raise ValueError(f"缺少问题向量缓存，不允许在线补齐：{cache_path}")
        with EmbeddingCache(cache_path) as cache:
            for question in questions:
                if (question["evaluation_suite"] == "retrieval_55") != (suite == "retrieval_55"):
                    continue
                key, query = question["question_id"], question["query"]
                if not cache.has_valid(key, query, "text-embedding-v4", 2048):
                    raise ValueError(f"缓存向量不匹配，不允许在线补齐：{key}")
                vectors[query] = cache.get(key, "text-embedding-v4", 2048).vector.tolist()
    service = VectorStoreService(adapter=ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors)))
    documents = service.get_all_documents()
    actual = {doc.metadata["chunk_id"]: {"document_id": doc.metadata["document_id"], "text": doc.page_content}
              for doc in documents}
    if actual != catalog or len(documents) != len(catalog):
        raise ValueError("正式库的ID、文本或来源与冻结语料不同")
    by_id = {doc.metadata["chunk_id"]: doc for doc in documents}
    counter = TokenCounter()
    tokens = {key: counter.count(value["text"]) for key, value in catalog.items()}
    chunks = [KnowledgeChunk.model_validate(row)
              for path in sorted((evaluation.chunks / "structure_aware_v1").rglob("chunks.jsonl"))
              for row in read_jsonl(path)]
    expander = NeighborhoodExpander(chunks, radius=1, extra_token_budget=2000, count_tokens=counter.count)
    expander.validate_index(documents)
    acceptance_path = reports / "node6_acceptance_v1_cache.json"
    acceptance = read(acceptance_path)
    expected_catalog = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    if acceptance["identity"]["catalog_sha256"] != expected_catalog:
        raise ValueError("重排缓存语料指纹不一致")
    question_path = bundle / "annotations/acceptance_questions_v1.jsonl"
    if acceptance["identity"]["questions_sha256"] != hashlib.sha256(question_path.read_bytes()).hexdigest():
        raise ValueError("重排缓存问题指纹不一致")
    dense_cache = {}

    def shared_dense(query: str, limit: int) -> list:
        """同题只查询向量库一次，避免近似检索波动混入分词的配对对照。"""
        if limit != 20:
            raise ValueError("本实验冻结每路20个候选")
        if query not in dense_cache:
            dense_cache[query] = service.get_vector_docs(query, limit)
        return dense_cache[query]

    retrievers = {name: HybridRetrieverService(shared_dense, lambda: documents,
                  strategy_id="hybrid_rrf", tokenizer_id=name, vector_k=20, bm25_k=20, rrf_k=60)
                  for name in ("char", "mixed")}
    report = {"identity": {"original_report_sha256": digest(original), "catalog_sha256": digest(catalog),
                           "specification_sha256": digest(specification), "vectors_sha256": digest(vectors),
                           "acceptance_cache_sha256": digest(acceptance),
                           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "settings": {"vector_k": 20, "bm25_k": 20, "rrf_k": 60, "seed_k": 6,
                           "neighbor_radius": 1, "extra_token_budget": 2000},
              "model_requests_this_run": 0, "diagnostic_gap_ids": gaps,
              "token_note": "cl100k_base估算，不是千问计费用量；不比较缓存重排的耗时",
              "methods": {}, "candidate_diagnostics": {}, "cache_unavailable": {}, "historical_baseline_drift": {}}
    report["identity"]["retrieval_code_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
            "retrieval/hybrid_retriever.py", "retrieval/tokenizers.py", "retrieval/evidence_expansion.py",
            "scripts/audit_node7_evidence.py", "scripts/evaluate_node6_acceptance.py",
            "scripts/evaluate_evidence_expansion.py")}
    for question in questions:
        key, query = question["question_id"], question["query"]
        requirements = specification["evidence_cases"].get(key, {}).get("requirements", [])
        rankings = {name: retriever.get_candidates(query, 40) for name, retriever in retrievers.items()}
        char_ids = [doc.metadata["chunk_id"] for doc in rankings["char"][:6]]
        if char_ids != original["rows"][key]["ranked_chunk_ids"]:
            report["historical_baseline_drift"][key] = {
                "historical": original["rows"][key]["ranked_chunk_ids"], "current": char_ids}
        cache_key = f"hybrid_mixed_qwen37/{key}"
        if cache_key in acceptance["rows"]:
            pool = [doc.metadata["chunk_id"] for doc in rankings["mixed"]]
            try:
                reranked = verified_rerank(acceptance["rows"][cache_key], query, pool)
            except ValueError as exc:
                report["cache_unavailable"][key] = str(exc)
            else:
                rankings["mixed_cached_rerank"] = [by_id[item] for item in reranked]
        else:
            report["cache_unavailable"][key] = "旧55题仅有重排前十名，缺少可校验的完整输入候选，不纳入严格对照"
        for method, ranked in rankings.items():
            seeds = ranked[:6]
            seed_ids = [doc.metadata["chunk_id"] for doc in seeds]
            for variant, selected in (("seed6", seeds), ("neighbors", expander(query, seeds))):
                ids = [doc.metadata["chunk_id"] for doc in selected]
                result = measure_context(question, ids, seed_ids, catalog, mapping, tokens, requirements)
                if result["added_estimated_tokens"] > 2000:
                    raise ValueError(f"邻接扩展超出预算：{key}")
                report["methods"].setdefault(f"{method}/{variant}", {})[key] = result
        if key in gaps:
            # 标注只用于结果诊断，不参与召回、融合或邻接候选的选择。
            report["candidate_diagnostics"][key] = {}
            for name, ranked in rankings.items():
                selected = [{"chunk_id": doc.metadata["chunk_id"],
                             **catalog[doc.metadata["chunk_id"]]} for doc in ranked]
                positions = {doc["chunk_id"]: index for index, doc in enumerate(selected, 1)}
                report["candidate_diagnostics"][key][name] = {
                    requirement["id"]: [positions[hit["chunk_id"]]
                        for hit in match_requirement(requirement, selected, mapping)]
                    for requirement in requirements}
            print(f"完成漏证据题：{key}", flush=True)
    report["summary"] = {}
    for method, rows in report["methods"].items():
        targeted = [rows[key] for key in gaps if key in rows]
        suites = {}
        for suite in ("retrieval_55", "acceptance_23"):
            values = [rows[q["question_id"]] for q in questions if q["evaluation_suite"] == suite and q["question_id"] in rows]
            if values:
                suites[suite] = {"count": len(values),
                    "protocol_complete_count": sum(row["protocol_evidence"]["complete_evidence"] for row in values),
                    "protocol_evidence_recall": statistics.fmean(row["protocol_evidence"]["evidence_recall"] for row in values),
                    "mean_total_estimated_tokens": statistics.fmean(row["total_estimated_tokens"] for row in values),
                    "mean_added_estimated_tokens": statistics.fmean(row["added_estimated_tokens"] for row in values)}
        report["summary"][method] = {"diagnostic_covered": len(targeted),
            "diagnostic_requirements_completed": sum(row["diagnostic_complete"] for row in targeted),
            "recovered_ids": [key for key in gaps if key in rows and rows[key]["diagnostic_complete"]], "suites": suites}
        baseline = report["methods"]["char/seed6"]
        report["summary"][method]["paired_protocol_changes_vs_char_seed6"] = {
            metric: {label: [key for key, row in rows.items()
                if compare(row["protocol_evidence"][metric], baseline[key]["protocol_evidence"][metric])]
                for label, compare in (("improved", lambda new, old: new > old),
                                       ("regressed", lambda new, old: new < old))}
            for metric in ("evidence_recall", "complete_evidence")}
    report["dense_candidates"] = {query: [doc.metadata["chunk_id"] for doc in docs]
                                  for query, docs in dense_cache.items()}
    # 排名本身纳入文件名，同样代码下的近似检索波动不会覆盖前次结果。
    output = reports / f"node7_retrieval_gaps_{digest(report)[:12]}.json"
    save_cache(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
