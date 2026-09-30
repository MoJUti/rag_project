"""使用真实评测问题比较 Chroma 与 Qdrant 嵌入式检索效果。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import chromadb
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.document_model import stable_document_id
from ingestion.storage import StorageLayout
from retrieval.embedding_cache import EmbeddingCache
from scripts.run_chunking_benchmark import ngram_recall, percentile, read_jsonl
from scripts.run_embedded_vector_preflight import COLLECTION_NAME


DEFAULT_DATASET_ID = "zh_public_minibench_v0.1"
DEFAULT_K_VALUES = (1, 3, 5, 10)


def load_chunk_catalog(chunks_root: Path) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(chunks_root.rglob("chunks.jsonl")):
        for row in read_jsonl(path):
            chunk_id = row["chunk_id"]
            if chunk_id in catalog:
                raise ValueError(f"重复 chunk_id: {chunk_id}")
            catalog[chunk_id] = {
                "document_id": row["document_id"],
                "text": row["text"],
            }
    if not catalog:
        raise RuntimeError(f"未找到切块目录: {chunks_root}")
    return catalog


def dcg(grades: list[int]) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def evaluate_ranking(
    question: dict[str, Any],
    ranked_chunk_ids: list[str],
    catalog: dict[str, dict[str, Any]],
    item_to_document: dict[str, str],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    relevant_items = list(question.get("relevant_document_ids") or [])
    relevant_documents = {item_to_document[item_id] for item_id in relevant_items}
    judgments = question.get("relevance_judgments") or {item_id: 1 for item_id in relevant_items}
    document_grades = {
        item_to_document[item_id]: int(grade)
        for item_id, grade in judgments.items()
        if item_id in item_to_document
    }
    ranked_documents = [catalog[chunk_id]["document_id"] for chunk_id in ranked_chunk_ids]
    first_relevant_rank = next(
        (rank for rank, document_id in enumerate(ranked_documents, 1) if document_id in relevant_documents),
        None,
    )
    metrics: dict[str, Any] = {
        "mrr": round(1 / first_relevant_rank, 6) if first_relevant_rank else 0.0,
        "first_relevant_rank": first_relevant_rank,
    }

    evidence = question.get("evidence") or []
    for k in k_values:
        top_chunk_ids = ranked_chunk_ids[:k]
        top_documents = set(ranked_documents[:k])
        relevant_found = relevant_documents & top_documents
        metrics[f"hit_at_{k}"] = float(bool(relevant_found))
        metrics[f"recall_at_{k}"] = (
            round(len(relevant_found) / len(relevant_documents), 6) if relevant_documents else 0.0
        )

        seen_documents: set[str] = set()
        grades: list[int] = []
        for document_id in ranked_documents[:k]:
            if document_id in seen_documents:
                grades.append(0)
            else:
                grades.append(document_grades.get(document_id, 0))
                seen_documents.add(document_id)
        ideal_grades = sorted(document_grades.values(), reverse=True)[:k]
        ideal = dcg(ideal_grades)
        metrics[f"ndcg_at_{k}"] = round(dcg(grades) / ideal, 6) if ideal else 0.0

        evidence_hits = 0
        for item in evidence:
            evidence_document = item_to_document.get(item.get("document_id", ""))
            quote = item.get("quote", "")
            hit = any(
                catalog[chunk_id]["document_id"] == evidence_document
                and ngram_recall(quote, catalog[chunk_id]["text"]) >= 0.75
                for chunk_id in top_chunk_ids
            )
            evidence_hits += int(hit)
        metrics[f"evidence_recall_at_{k}"] = (
            round(evidence_hits / len(evidence), 6) if evidence else 0.0
        )
        metrics[f"complete_evidence_hit_at_{k}"] = float(bool(evidence) and evidence_hits == len(evidence))
    return metrics


def aggregate(rows: list[dict[str, Any]], k_values: tuple[int, ...]) -> dict[str, float]:
    keys = ["mrr"]
    for k in k_values:
        keys.extend([
            f"hit_at_{k}",
            f"recall_at_{k}",
            f"ndcg_at_{k}",
            f"evidence_recall_at_{k}",
            f"complete_evidence_hit_at_{k}",
        ])
    return {
        key: round(statistics.fmean(row["metrics"][key] for row in rows), 6)
        for key in keys
    }


def run_database(
    name: str,
    questions: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
    search: Callable[[list[float], int], list[str]],
    catalog: dict[str, dict[str, Any]],
    item_to_document: dict[str, str],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    max_k = max(k_values)
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for question in questions:
        started = time.perf_counter()
        ranked_chunk_ids = search(query_vectors[question["question_id"]], max_k)
        latencies.append((time.perf_counter() - started) * 1000)
        if len(ranked_chunk_ids) != max_k:
            raise RuntimeError(
                f"{name} 返回数量异常: {question['question_id']} {len(ranked_chunk_ids)}/{max_k}"
            )
        rows.append({
            "question_id": question["question_id"],
            "query": question["query"],
            "question_type": question.get("question_type"),
            "modality": question.get("modality"),
            "ranked_chunk_ids": ranked_chunk_ids,
            "metrics": evaluate_ranking(
                question, ranked_chunk_ids, catalog, item_to_document, k_values,
            ),
        })
    return {
        "summary": aggregate(rows, k_values),
        "query_latency": {
            "mean_ms": round(statistics.fmean(latencies), 3),
            "p50_ms": round(percentile(latencies, 0.50), 3),
            "p95_ms": round(percentile(latencies, 0.95), 3),
        },
        "questions": rows,
    }


def run(args: argparse.Namespace) -> None:
    k_values = tuple(sorted(set(args.k_values)))
    if not k_values or min(k_values) < 1:
        raise ValueError("k 必须是正整数")
    evaluation = StorageLayout.from_env(PROJECT_ROOT).evaluation(args.dataset_id)
    dataset_root = PROJECT_ROOT / "data" / "eval" / args.dataset_id
    questions = read_jsonl(dataset_root / "annotations" / "retrieval_questions.jsonl")
    manifest = read_jsonl(dataset_root / "manifest.jsonl")
    item_to_document = {
        item["item_id"]: stable_document_id(item["sha256"])
        for item in manifest
    }
    catalog = load_chunk_catalog(evaluation.chunks / args.strategy)

    unknown_items = sorted({
        item_id
        for question in questions
        for item_id in question.get("relevant_document_ids", [])
        if item_id not in item_to_document
    })
    if unknown_items:
        raise RuntimeError(f"问题标注引用了清单中不存在的文档: {unknown_items}")

    query_cache_path = (
        evaluation.embeddings / f"{args.model}-{args.dimensions}"
        / "question_embeddings.sqlite3"
    )
    query_vectors: dict[str, list[float]] = {}
    with EmbeddingCache(query_cache_path) as cache:
        for question in questions:
            item = cache.get(question["question_id"], args.model, args.dimensions)
            if item is None or not cache.has_valid(
                question["question_id"], question["query"], args.model, args.dimensions,
            ):
                raise RuntimeError(f"缺少问题向量: {question['question_id']}")
            query_vectors[question["question_id"]] = item.vector.tolist()

    index_root = evaluation.root / "indexes" / "embedded_preflight"
    chroma = chromadb.PersistentClient(path=str(index_root / "chroma")).get_collection(COLLECTION_NAME)
    qdrant = QdrantClient(path=str(index_root / "qdrant"))
    try:
        chroma_result = run_database(
            "chroma",
            questions,
            query_vectors,
            lambda vector, k: chroma.query(
                query_embeddings=[vector], n_results=k, include=["distances"],
            )["ids"][0],
            catalog,
            item_to_document,
            k_values,
        )
        qdrant_result = run_database(
            "qdrant",
            questions,
            query_vectors,
            lambda vector, k: [
                str(point.payload["chunk_id"])
                for point in qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    limit=k,
                    with_payload=["chunk_id"],
                ).points
            ],
            catalog,
            item_to_document,
            k_values,
        )
    finally:
        qdrant.close()

    chroma_by_id = {row["question_id"]: row for row in chroma_result["questions"]}
    qdrant_by_id = {row["question_id"]: row for row in qdrant_result["questions"]}
    overlaps = []
    max_k = max(k_values)
    for question in questions:
        question_id = question["question_id"]
        left = set(chroma_by_id[question_id]["ranked_chunk_ids"])
        right = set(qdrant_by_id[question_id]["ranked_chunk_ids"])
        overlaps.append(len(left & right) / max_k)

    report = {
        "scope": "embedded_real_query_retrieval_baseline",
        "warning": "本报告用于验证检索质量与评测链路；嵌入式延迟不代表独立服务性能。",
        "dataset_id": args.dataset_id,
        "strategy": args.strategy,
        "embedding": {"model": args.model, "dimensions": args.dimensions},
        "question_count": len(questions),
        "chunk_count": len(catalog),
        "k_values": list(k_values),
        "chroma": chroma_result,
        "qdrant": qdrant_result,
        "cross_database": {
            f"mean_top_{max_k}_chunk_overlap": round(statistics.fmean(overlaps), 6),
            "identical_ranked_chunk_lists": sum(
                chroma_by_id[q["question_id"]]["ranked_chunk_ids"]
                == qdrant_by_id[q["question_id"]]["ranked_chunk_ids"]
                for q in questions
            ),
        },
    }
    report_path = evaluation.reports / "node5_embedded_real_query_baseline.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "question_count": len(questions),
        "chunk_count": len(catalog),
        "chroma": {"summary": chroma_result["summary"], "query_latency": chroma_result["query_latency"]},
        "qdrant": {"summary": qdrant_result["summary"], "query_latency": qdrant_result["query_latency"]},
        "cross_database": report["cross_database"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--strategy", default="structure_aware_v1")
    parser.add_argument("--model", default="text-embedding-v4")
    parser.add_argument("--dimensions", type=int, default=2048)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
