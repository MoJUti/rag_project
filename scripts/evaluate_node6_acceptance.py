"""固定配置评测新增验收题；重复来源按等价组计分，拒答题单列。"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from ingestion.document_model import stable_document_id
from ingestion.storage import StorageLayout
from retrieval.compatible_embeddings import CompatibleEmbeddingClient
from retrieval.embedding_cache import EmbeddingCache
from retrieval.hybrid_retriever import HybridRetrieverService
from retrieval.pipeline import RetrievalPipeline
from scripts.evaluate_embedded_retrieval import load_chunk_catalog
from scripts.run_chunking_benchmark import compact, ngram_recall, percentile, read_jsonl
from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings
from scripts.run_node6_rerank_eval import save_cache


def evidence_matches(evidence: dict, texts: list[str]) -> bool:
    """文字表格采用精确规范化匹配；图像文字允许少量排版差异。"""
    if not texts:
        return False
    joined = "\n".join(texts)
    if evidence["locator"]["type"] == "page_region":
        return ngram_recall(evidence["quote"], joined) >= 0.9
    required = [evidence["quote"], *evidence.get("context_quotes", [])]
    normalized = compact(joined)
    return all(compact(quote) in normalized for quote in required)


def score_question(question: dict, ids: list[str], catalog: dict, mapping: dict[str, str]) -> dict[str, float]:
    """证据可分布在多个已召回切块；等价副本不要求重复召回。"""
    if not question["answerable"]:
        raise ValueError("资料不足题不适用召回率，必须在回答阶段评测拒答")
    if not question.get("evidence") or not question.get("relevant_document_groups"):
        raise ValueError("可回答题缺少证据或文档等价组")
    selected = [catalog[chunk_id] for chunk_id in ids]
    found_documents = {row["document_id"] for row in selected}
    groups = [{mapping[item] for item in group} for group in question["relevant_document_groups"]]
    document_recall = sum(bool(group & found_documents) for group in groups) / len(groups)
    relevant = set.union(*groups)
    first = next((rank for rank, row in enumerate(selected, 1) if row["document_id"] in relevant), None)
    hits = []
    for atom in question["evidence"]:
        acceptable = {mapping[item] for item in atom["document_ids"]}
        texts = [row["text"] for row in selected if row["document_id"] in acceptable]
        hits.append(evidence_matches(atom, texts))
    return {"mrr": 1 / first if first else 0.0, "document_recall_at_6": document_recall,
            "evidence_recall_at_6": sum(hits) / len(hits), "complete_evidence_at_6": float(all(hits))}


def main() -> None:
    dataset_id = "zh_public_minibench_v0.1"
    bundle = ROOT / "data/eval" / dataset_id
    question_path = bundle / "annotations/acceptance_questions_v1.jsonl"
    frozen = json.loads((bundle / "annotations/acceptance_v1.lock.json").read_text(encoding="utf-8"))
    question_hash = hashlib.sha256(question_path.read_bytes()).hexdigest()
    if question_hash != frozen["questions_sha256"]:
        raise ValueError("验收题与冻结指纹不一致")
    if config.embedding_model_name != "text-embedding-v4" or config.embedding_dimensions != 2048:
        raise ValueError("本轮已冻结使用text-embedding-v4/2048，当前配置不一致")
    questions = read_jsonl(question_path)
    evaluation = StorageLayout.from_env(ROOT).evaluation(dataset_id)
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    catalog_hash = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    manifest = read_jsonl(bundle / "manifest.jsonl")
    mapping = {row["item_id"]: stable_document_id(row["sha256"]) for row in manifest}
    corpus_documents = {row["document_id"] for row in catalog.values()}
    if any(mapping[item] not in corpus_documents for question in questions
           for item in question.get("relevant_document_ids", [])):
        raise ValueError("验收题包含未入库的相关文档")
    cache_path = evaluation.embeddings / "text-embedding-v4-2048/acceptance_v1_embeddings.sqlite3"
    embedding_tokens = 0
    vectors = {}
    with EmbeddingCache(cache_path) as cache:
        pending = [row for row in questions if not cache.has_valid(
            row["question_id"], row["query"], "text-embedding-v4", 2048,
        )]
        if pending:
            client = CompatibleEmbeddingClient(config.embedding_base_url,
                                                os.getenv("DASHSCOPE_API_KEY", ""), "text-embedding-v4", 2048)
            try:
                for start in range(0, len(pending), 10):
                    batch = pending[start:start + 10]
                    values, tokens = client.embed([row["query"] for row in batch])
                    cache.put_many([(row["question_id"], row["query"], value, None)
                                    for row, value in zip(batch, values)], "text-embedding-v4", 2048)
                    embedding_tokens += tokens or 0
            finally:
                client.close()
        for row in questions:
            vectors[row["query"]] = cache.get(row["question_id"], "text-embedding-v4", 2048).vector.tolist()
    service = VectorStoreService(adapter=ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors)))
    if service.count() != len(catalog):
        raise ValueError("正式库数量与验收切块不一致")
    actual_ids = set(service.adapter.store._collection.get(include=[])["ids"])
    if actual_ids != set(catalog):
        raise ValueError("正式库切块ID与验收目录不一致")
    cache_file = evaluation.reports / "node6_acceptance_v1_cache.json"
    identity = {"questions_sha256": question_hash, "catalog_sha256": catalog_hash, "config": frozen}
    cached = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {
        "identity": identity, "rows": {},
    }
    if cached["identity"] != identity:
        raise ValueError("已缓存评测与当前题目、切块或配置不一致，需新建评测版本")
    report = {"identity": identity, "embedding_tokens_this_run": embedding_tokens,
              "embedding_usage_note": "仅统计本次执行；问题向量首轮生成后曾中断，前次用量未落盘，总嵌入Token不可由此字段推算",
              "answerable_count": sum(row["answerable"] for row in questions),
              "refusal_count": sum(not row["answerable"] for row in questions),
              "methods": {}, "refusal_cases": []}
    answerable = [row for row in questions if row["answerable"]]
    for name in frozen["locked_profiles"]:
        strategy = "dense" if name == "dense" else "bm25" if name == "bm25_mixed" else "hybrid_rrf"
        retriever = HybridRetrieverService(service.get_vector_docs, service.get_all_documents,
                                           get_document_count=service.count, strategy_id=strategy,
                                           tokenizer_id="mixed", vector_k=20, bm25_k=20, final_k=6, rrf_k=60)
        pipeline = RetrievalPipeline(retriever, final_k=6, candidate_k=40,
                                     rerank_strategy="qwen3.7-text-rerank" if name.endswith("qwen37") else "none")
        rows = []
        for question in answerable:
            key = f"{name}/{question['question_id']}"
            if key not in cached["rows"]:
                result = pipeline.run(question["query"])
                cached["rows"][key] = {"question_id": question["question_id"],
                                       "ranked_chunk_ids": [doc.metadata["chunk_id"] for doc in result.documents],
                                       "trace": [asdict(stage) for stage in result.trace]}
                save_cache(cache_file, cached)
                print(f"complete={key}", flush=True)
            row = dict(cached["rows"][key])
            row["metrics"] = score_question(question, row["ranked_chunk_ids"], catalog, mapping)
            rows.append(row)
        summary = {metric: round(statistics.fmean(row["metrics"][metric] for row in rows), 6)
                   for metric in rows[0]["metrics"]}
        latencies = [sum(stage["elapsed_ms"] for stage in row["trace"]) for row in rows]
        report["methods"][name] = {"summary": summary, "questions": rows,
                                   "total_tokens": sum(stage["total_tokens"] for row in rows for stage in row["trace"]),
                                   "latency_ms": {"p50": round(percentile(latencies, 0.5), 3),
                                                  "p95": round(percentile(latencies, 0.95), 3)}}
        print(json.dumps({"method": name, **summary}, ensure_ascii=False), flush=True)

    # 拒答正确与否取决于回答模型；此处只保留相似干扰材料，不设置相似度拒答阈值。
    for question in questions:
        if not question["answerable"]:
            docs = service.get_vector_docs(question["query"], 6)
            report["refusal_cases"].append({"question_id": question["question_id"],
                                             "ranked_chunk_ids": [doc.metadata["chunk_id"] for doc in docs],
                                             "status": "awaiting_generation_evaluation"})
    report["corpus_evidence_availability"] = {
        row["question_id"]: score_question(row, list(catalog), catalog, mapping)["evidence_recall_at_6"]
        for row in answerable
    }
    output = evaluation.reports / "node6_acceptance_v1_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={output}")


if __name__ == "__main__":
    main()
