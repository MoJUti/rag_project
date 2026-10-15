"""使用缓存问题向量评测当前应用的三种召回方式。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from ingestion.document_model import stable_document_id
from ingestion.storage import StorageLayout
from retrieval.embedding_cache import EmbeddingCache
from retrieval.hybrid_retriever import HybridRetrieverService
from scripts.evaluate_embedded_retrieval import aggregate, evaluate_ranking, load_chunk_catalog
from scripts.run_chunking_benchmark import percentile, read_jsonl


class CachedQuestionEmbeddings(Embeddings):
    """评测只读取问题向量缓存，不发起模型请求。"""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.vectors[text]
        except KeyError as exc:
            raise ValueError(f"问题没有缓存向量：{text[:40]}") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("基线评测不允许生成新向量")


def main() -> None:
    dataset_id = "zh_public_minibench_v0.1"
    evaluation = StorageLayout.from_env(ROOT).evaluation(dataset_id)
    dataset = ROOT / "data" / "eval" / dataset_id
    questions = read_jsonl(dataset / "annotations" / "retrieval_questions.jsonl")
    manifest = read_jsonl(dataset / "manifest.jsonl")
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    item_to_document = {row["item_id"]: stable_document_id(row["sha256"]) for row in manifest}
    model, dimensions = config.embedding_model_name, config.embedding_dimensions
    cache_path = evaluation.embeddings / f"{model}-{dimensions}" / "question_embeddings.sqlite3"
    vectors: dict[str, list[float]] = {}
    with EmbeddingCache(cache_path) as cache:
        for question in questions:
            key, text = question["question_id"], question["query"]
            if not cache.has_valid(key, text, model, dimensions):
                raise ValueError(f"问题向量缓存不匹配：{key}")
            vectors[text] = cache.get(key, model, dimensions).vector.tolist()

    adapter = ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors))
    service = VectorStoreService(adapter=adapter)
    if service.count() != len(catalog):
        raise ValueError(f"正式库与评测切块数量不符：{service.count()}/{len(catalog)}")
    actual_ids = set(adapter.store._collection.get(include=[])["ids"])
    if actual_ids != set(catalog):
        raise ValueError("正式库与评测集的切块 ID 不一致")

    retriever = HybridRetrieverService(
        get_vector_docs=service.get_vector_docs,
        get_all_docs=service.get_all_documents,
        get_document_count=service.count,
        vector_k=config.hybrid_vector_k,
        bm25_k=config.hybrid_bm25_k,
        final_k=10,
        rrf_k=config.hybrid_rrf_k,
    )
    retriever._ensure_bm25()
    k_values = (1, 3, 5, 6, 10)
    methods = {
        "dense": lambda query: service.get_vector_docs(query, 10),
        "bm25": lambda query: retriever._bm25_retriever.invoke(query)[:10],
        "hybrid_rrf": retriever.retrieve,
    }
    report = {
        "dataset_id": dataset_id,
        "question_count": len(questions),
        "chunk_count": len(catalog),
        "collection": config.collection_name,
        "embedding": {"model": model, "dimensions": dimensions},
        "config": {
            "vector_k": config.hybrid_vector_k,
            "bm25_k": config.hybrid_bm25_k,
            "rrf_k": config.hybrid_rrf_k,
            "evaluated_final_k": 10,
            "application_final_k": config.hybrid_final_k,
        },
        "methods": {},
    }
    for name, search in methods.items():
        rows, latencies = [], []
        for question in questions:
            started = time.perf_counter()
            documents = search(question["query"])
            latencies.append((time.perf_counter() - started) * 1000)
            ids = [document.metadata.get("chunk_id") for document in documents]
            if len(ids) != 10 or any(key not in catalog for key in ids) or len(set(ids)) != 10:
                raise ValueError(f"{name} 的候选数量、ID 或去重异常：{question['question_id']}")
            rows.append({
                "question_id": question["question_id"],
                "question_type": question.get("question_type"),
                "modality": question.get("modality"),
                "ranked_chunk_ids": ids,
                "metrics": evaluate_ranking(question, ids, catalog, item_to_document, k_values),
            })
        report["methods"][name] = {
            "summary": aggregate(rows, k_values),
            "latency_ms": {
                "p50": round(percentile(latencies, 0.5), 3),
                "p95": round(percentile(latencies, 0.95), 3),
            },
            "questions": rows,
        }
        print(json.dumps({"method": name, "summary": report["methods"][name]["summary"],
                          "latency_ms": report["methods"][name]["latency_ms"]}, ensure_ascii=False), flush=True)
    evaluation.reports.mkdir(parents=True, exist_ok=True)
    output = evaluation.reports / "node6_retrieval_baseline_after_cache_fix.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={output}")


if __name__ == "__main__":
    main()
