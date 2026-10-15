"""复用已冻结的向量候选，对比6、10、20块上下文，不调用模型。"""

from __future__ import annotations

import argparse
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
from ingestion.chunking import TokenCounter
from ingestion.document_model import stable_document_id
from ingestion.storage import StorageLayout
from retrieval.hybrid_retriever import HybridRetrieverService
from scripts.audit_node7_evidence import audit, match_requirement
from scripts.evaluate_embedded_retrieval import evaluate_ranking, load_chunk_catalog
from scripts.evaluate_evidence_expansion import original_context_score
from scripts.evaluate_node6_acceptance import score_question
from scripts.run_chunking_benchmark import percentile, read_jsonl
from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings
from scripts.run_node6_rerank_eval import save_cache
from scripts.run_node7_prompt_ab import digest

K_VALUES = (6, 10, 20)


def context_metrics(question: dict, ids: list[str], catalog: dict, mapping: dict,
                    token_counts: dict, requirements: list[dict]) -> dict:
    """分别记录同来源上下文覆盖与最小原文要求，不把证据覆盖当成回答准确率。"""
    if not ids or len(ids) != len(set(ids)) or set(ids) - catalog.keys():
        raise ValueError(f"上下文为空、重复或包含未知切块：{question['question_id']}")
    if question["evaluation_suite"] == "retrieval_55":
        scores = original_context_score(question, ids, catalog, mapping)
        old = evaluate_ranking(question, ids, catalog, mapping, (len(ids),))
        document_recall = old[f"recall_at_{len(ids)}"]
        legacy = {"evidence_recall": old[f"evidence_recall_at_{len(ids)}"],
                  "complete_evidence": old[f"complete_evidence_hit_at_{len(ids)}"]}
    else:
        old = score_question(question, ids, catalog, mapping)
        scores = {"evidence_recall": old["evidence_recall_at_6"],
                  "complete_evidence": old["complete_evidence_at_6"]}
        document_recall = old["document_recall_at_6"]
        legacy = None
    documents = [{"chunk_id": key, **catalog[key]} for key in ids]
    hits = {item["id"]: bool(match_requirement(item, documents, mapping)) for item in requirements}
    relevant = {mapping[item] for item in question["relevant_document_ids"]}
    return {"chunk_ids": ids, "block_count": len(ids),
            "estimated_tokens": sum(token_counts[key] for key in ids),
            "document_recall": document_recall, "context_evidence": scores,
            "legacy_single_chunk_evidence": legacy,
            "relevant_source_blocks": sum(doc["document_id"] in relevant for doc in documents),
            "diagnostic_requirements": hits, "diagnostic_complete": all(hits.values()) if hits else None}


def prefixes(pool: list[str], catalog: dict) -> dict[int, list[str]]:
    """同一个有序候选池截取嵌套前缀，不能为不同块数重新召回。"""
    if len(pool) < max(K_VALUES) or len(pool) != len(set(pool)) or set(pool) - catalog.keys():
        raise ValueError("融合候选不足20条、重复或包含未知切块")
    return {k: pool[:k] for k in K_VALUES}


def summarize(rows: dict, keys: list[str]) -> dict:
    """仅聚合同一题组，四道拒答题不纳入召回率。"""
    values = [rows[key] for key in keys]
    if not values:
        raise ValueError("统计题组不能为空")
    tokens = [row["estimated_tokens"] for row in values]
    return {"count": len(values),
            "complete_evidence_count": sum(row["context_evidence"]["complete_evidence"] for row in values),
            "mean_evidence_recall": statistics.fmean(row["context_evidence"]["evidence_recall"] for row in values),
            "mean_document_recall": statistics.fmean(row["document_recall"] for row in values),
            "estimated_tokens": {"mean": statistics.fmean(tokens), "median": statistics.median(tokens),
                                 "p95": percentile(tokens, 0.95), "max": max(tokens)},
            "mean_relevant_source_block_ratio": statistics.fmean(
                row["relevant_source_blocks"] / row["block_count"] for row in values)}


def main(frozen_path: Path) -> None:
    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    bundle = ROOT / "data/eval/zh_public_minibench_v0.1"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    original = json.loads((evaluation.reports / "node7_end_to_end_40d575b537eb.json").read_text(encoding="utf-8"))
    specification = json.loads((bundle / "annotations/node7_evidence_audit_v1.json").read_text(encoding="utf-8"))
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    mapping = {row["item_id"]: stable_document_id(row["sha256"])
               for row in read_jsonl(bundle / "manifest.jsonl")}
    audit(original, specification, catalog, mapping)
    if frozen["identity"]["original_report_sha256"] != digest(original) or frozen["identity"]["catalog_sha256"] != digest(catalog):
        raise ValueError("冻结向量候选与原报告或语料指纹不一致")
    if frozen["settings"] != {"vector_k": 20, "bm25_k": 20, "rrf_k": 60, "seed_k": 6,
                              "neighbor_radius": 1, "extra_token_budget": 2000}:
        raise ValueError("冻结候选的检索参数与本轮基线不一致")
    if (config.embedding_model_name, config.embedding_dimensions, config.collection_name, config.chroma_mode) != (
        "text-embedding-v4", 2048, "rag_text_embedding_v4_2048", "local",
    ) or config.retrieval_source_filter:
        raise ValueError("正式库配置与本轮冻结条件不一致")
    # 空缓存嵌入器拒绝任何嵌入调用；本轮只读取库内文本，向量候选来自已完成实验。
    service = VectorStoreService(adapter=ChromaVectorStoreAdapter(CachedQuestionEmbeddings({})))
    documents = service.get_all_documents()
    actual = {doc.metadata["chunk_id"]: {"document_id": doc.metadata["document_id"], "text": doc.page_content}
              for doc in documents}
    if actual != catalog or len(documents) != len(catalog):
        raise ValueError("正式库切块ID、文本、来源或数量与冻结语料不同")
    by_id = {doc.metadata["chunk_id"]: doc for doc in documents}

    def frozen_dense(query: str, limit: int) -> list:
        if limit != 20 or query not in frozen["dense_candidates"]:
            raise ValueError("冻结向量候选缺少本题或候选数量参数不是20")
        ids = frozen["dense_candidates"][query]
        if len(ids) != 20 or len(set(ids)) != 20 or set(ids) - catalog.keys():
            raise ValueError("冻结向量候选应为20个不同的已知切块")
        return [by_id[key] for key in ids]

    retriever = HybridRetrieverService(frozen_dense, lambda: documents, tokenizer_id="char",
                                       vector_k=20, bm25_k=20, rrf_k=60)
    questions = {key: row["question"] for key, row in original["rows"].items() if row["question"]["answerable"]}
    if len(questions) != 74 or len(frozen["diagnostic_gap_ids"]) != 8:
        raise ValueError("本轮应包含74道可回答题和8道漏证据诊断题")
    counter = TokenCounter()
    tokens = {key: counter.count(row["text"]) for key, row in catalog.items()}
    report = {"identity": {"frozen_candidates_sha256": digest(frozen), "catalog_sha256": digest(catalog),
                           "original_report_sha256": digest(original), "specification_sha256": digest(specification),
                           "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                               "scripts/compare_node7_context_sizes.py", "retrieval/hybrid_retriever.py",
                               "scripts/evaluate_evidence_expansion.py", "scripts/evaluate_node6_acceptance.py",
                               "scripts/evaluate_embedded_retrieval.py", "scripts/audit_node7_evidence.py")}},
              "settings": {"tokenizer": "char", "vector_k": 20, "bm25_k": 20, "rrf_k": 60,
                           "rerank": "none", "expansion": "none", "k_values": K_VALUES},
              "model_requests_this_run": 0, "token_note": "cl100k_base估算，仅含正文，不含提示词和输出",
              "diagnostic_gap_ids": frozen["diagnostic_gap_ids"], "rankings": {},
              "methods": {str(k): {} for k in K_VALUES}}
    for key, question in questions.items():
        pool = [doc.metadata["chunk_id"] for doc in retriever.get_candidates(question["query"], 40)]
        selections = prefixes(pool, catalog)
        if selections[6] != frozen["methods"]["char/seed6"][key]["chunk_ids"]:
            raise ValueError(f"逐字分词前六名未复现冻结基线，停止比较：{key}")
        report["rankings"][key] = pool
        requirements = specification["evidence_cases"].get(key, {}).get("requirements", [])
        for k, ids in selections.items():
            report["methods"][str(k)][key] = context_metrics(question, ids, catalog, mapping, tokens, requirements)
    report["summary"] = {}
    for k, rows in report["methods"].items():
        baseline = report["methods"]["6"]
        report["summary"][k] = {
            "suites": {suite: summarize(rows, [key for key, q in questions.items() if q["evaluation_suite"] == suite])
                       for suite in ("retrieval_55", "acceptance_23")},
            "tokens_all74": summarize(rows, list(questions))["estimated_tokens"],
            "diagnostic_recovered_ids": [key for key in report["diagnostic_gap_ids"] if rows[key]["diagnostic_complete"]],
            "paired_vs_6": {label: [key for key in questions if compare(
                rows[key]["context_evidence"]["complete_evidence"], baseline[key]["context_evidence"]["complete_evidence"])]
                for label, compare in (("improved", lambda new, old: new > old), ("regressed", lambda new, old: new < old))}}
    output = evaluation.reports / f"node7_context_sizes_{digest(report)[:12]}.json"
    save_cache(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-candidates", type=Path, default=ROOT / (
        "storage/evaluations/zh_public_minibench_v0.1/reports/node7_retrieval_gaps_411d6190927b.json"))
    main(parser.parse_args().frozen_candidates)
