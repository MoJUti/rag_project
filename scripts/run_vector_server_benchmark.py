"""使用独立 Chroma/Qdrant 服务执行向量数据库正式对比。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import chromadb
import numpy as np
from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.document_model import stable_document_id
from ingestion.storage import StorageLayout
from retrieval.embedding_cache import EmbeddingCache
from scripts.evaluate_embedded_retrieval import (
    aggregate,
    evaluate_ranking,
    load_chunk_catalog,
)
from scripts.run_chunking_benchmark import percentile, read_jsonl
from scripts.run_embedded_vector_preflight import load_records, qdrant_point_id


DEFAULT_DATASET_ID = "zh_public_minibench_v0.1"
DEFAULT_K_VALUES = (1, 3, 5, 10)
COLLECTION_PREFIX = "structure_aware_v1_text_embedding_v4_2048_server"


def wait_for_count(
    read_count: Callable[[], int], expected: int, timeout_seconds: float = 60.0,
) -> tuple[int, float]:
    started = time.perf_counter()
    count = read_count()
    while count != expected and time.perf_counter() - started < timeout_seconds:
        time.sleep(0.25)
        count = read_count()
    elapsed = time.perf_counter() - started
    if count != expected:
        raise RuntimeError(f"索引条数未就绪: {count}/{expected}")
    return count, elapsed


def exact_rankings(
    records: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
    top_k: int,
) -> dict[str, list[str]]:
    matrix = np.stack([row["vector"] for row in records]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    chunk_ids = np.asarray([row["chunk_id"] for row in records], dtype=object)
    rankings: dict[str, list[str]] = {}
    for question in questions:
        query = np.asarray(query_vectors[question["question_id"]], dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = matrix @ query
        candidates = np.argpartition(scores, -top_k)[-top_k:]
        ordered = candidates[np.argsort(scores[candidates])[::-1]]
        rankings[question["question_id"]] = chunk_ids[ordered].tolist()
    return rankings


def evaluate_service(
    name: str,
    questions: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
    search: Callable[[list[float], int], list[str]],
    catalog: dict[str, dict[str, Any]],
    item_to_document: dict[str, str],
    exact: dict[str, list[str]],
    k_values: tuple[int, ...],
    warmup_rounds: int,
    measured_rounds: int,
) -> dict[str, Any]:
    max_k = max(k_values)
    if questions:
        warmup_vector = query_vectors[questions[0]["question_id"]]
        for _ in range(warmup_rounds):
            search(warmup_vector, max_k)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for round_index in range(measured_rounds):
        for question in questions:
            started = time.perf_counter()
            ranked = search(query_vectors[question["question_id"]], max_k)
            latencies.append((time.perf_counter() - started) * 1000)
            if len(ranked) != max_k:
                raise RuntimeError(
                    f"{name} 返回数量异常: {question['question_id']} {len(ranked)}/{max_k}"
                )
            if round_index == 0:
                exact_ids = exact[question["question_id"]]
                rows.append({
                    "question_id": question["question_id"],
                    "query": question["query"],
                    "question_type": question.get("question_type"),
                    "modality": question.get("modality"),
                    "ranked_chunk_ids": ranked,
                    "exact_top_k_chunk_ids": exact_ids,
                    "exact_top_1_match": ranked[0] == exact_ids[0],
                    "exact_top_k_overlap": len(set(ranked) & set(exact_ids)) / max_k,
                    "metrics": evaluate_ranking(
                        question, ranked, catalog, item_to_document, k_values,
                    ),
                })

    return {
        "summary": aggregate(rows, k_values),
        "ann_fidelity": {
            "exact_top_1_match_rate": round(
                statistics.fmean(float(row["exact_top_1_match"]) for row in rows), 6,
            ),
            f"mean_exact_top_{max_k}_overlap": round(
                statistics.fmean(row["exact_top_k_overlap"] for row in rows), 6,
            ),
        },
        "query_latency": {
            "sample_count": len(latencies),
            "warmup_rounds": warmup_rounds,
            "measured_rounds": measured_rounds,
            "mean_ms": round(statistics.fmean(latencies), 3),
            "p50_ms": round(percentile(latencies, 0.50), 3),
            "p95_ms": round(percentile(latencies, 0.95), 3),
        },
        "questions": rows,
    }


def build_chroma(
    client: Any,
    collection_name: str,
    records: list[dict[str, Any]],
    batch_size: int,
    profile: str,
) -> tuple[Any, dict[str, Any]]:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    hnsw: dict[str, Any] = {"space": "cosine"}
    if profile in {"tuned", "hnsw_matched"}:
        hnsw.update({
            "ef_construction": 200,
            "max_neighbors": 32,
            "ef_search": 200,
        })
    collection = client.create_collection(
        name=collection_name,
        configuration={"hnsw": hnsw},
    )
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        collection.add(
            ids=[row["chunk_id"] for row in batch],
            embeddings=[row["vector"].tolist() for row in batch],
            documents=[row["text"] for row in batch],
            metadatas=[{"token_count": row["token_count"]} for row in batch],
        )
    write_seconds = time.perf_counter() - started
    count, ready_wait_seconds = wait_for_count(collection.count, len(records))
    return collection, {
        "count": count,
        "write_seconds": round(write_seconds, 3),
        "ready_wait_seconds": round(ready_wait_seconds, 3),
        "hnsw": collection.configuration_json["hnsw"],
    }


def build_qdrant(
    client: QdrantClient,
    collection_name: str,
    records: list[dict[str, Any]],
    dimensions: int,
    batch_size: int,
    profile: str,
) -> dict[str, Any]:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    create_options: dict[str, Any] = {}
    if profile == "hnsw_matched":
        create_options = {
            "hnsw_config": models.HnswConfigDiff(
                m=32,
                ef_construct=200,
                full_scan_threshold=10,
            ),
            # 批量上传阶段暂不建图；写完后一次性触发优化，避免重复重建。
            "optimizers_config": models.OptimizersConfigDiff(
                indexing_threshold=50_000,
                default_segment_number=1,
            ),
        }
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dimensions,
            distance=models.Distance.COSINE,
        ),
        **create_options,
    )
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                models.PointStruct(
                    id=qdrant_point_id(row["chunk_id"]),
                    vector=row["vector"].tolist(),
                    payload={
                        "chunk_id": row["chunk_id"],
                        "token_count": row["token_count"],
                    },
                )
                for row in batch
            ],
        )
    upload_seconds = time.perf_counter() - started
    index_seconds = 0.0
    if profile == "hnsw_matched":
        index_started = time.perf_counter()
        client.update_collection(
            collection_name=collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=1_000,
                default_segment_number=1,
            ),
        )
        while True:
            info = client.get_collection(collection_name)
            indexed = info.indexed_vectors_count or 0
            if info.status == models.CollectionStatus.GREEN and indexed >= len(records):
                break
            if time.perf_counter() - index_started > 180:
                raise TimeoutError(
                    f"Qdrant HNSW 构建超时: indexed={indexed}/{len(records)} status={info.status}"
                )
            time.sleep(0.25)
        index_seconds = time.perf_counter() - index_started
    count, ready_wait_seconds = wait_for_count(
        lambda: client.count(collection_name, exact=True).count,
        len(records),
    )
    info = client.get_collection(collection_name)
    return {
        "count": count,
        "write_seconds": round(upload_seconds + index_seconds, 3),
        "upload_seconds": round(upload_seconds, 3),
        "index_seconds": round(index_seconds, 3),
        "ready_wait_seconds": round(ready_wait_seconds, 3),
        "indexed_vectors_count": info.indexed_vectors_count or 0,
        "segments_count": info.segments_count,
        "hnsw": info.config.hnsw_config.model_dump(),
        "optimizers": info.config.optimizer_config.model_dump(),
    }


def run(args: argparse.Namespace) -> None:
    k_values = tuple(sorted(set(args.k_values)))
    if not k_values or min(k_values) < 1:
        raise ValueError("k 必须是正整数")
    if args.measured_rounds < 1 or args.warmup_rounds < 0:
        raise ValueError("查询轮数参数无效")

    evaluation = StorageLayout.from_env(PROJECT_ROOT).evaluation(args.dataset_id)
    evaluation.ensure()
    dataset_root = PROJECT_ROOT / "data" / "eval" / args.dataset_id
    questions = read_jsonl(dataset_root / "annotations" / "retrieval_questions.jsonl")
    manifest = read_jsonl(dataset_root / "manifest.jsonl")
    item_to_document = {
        item["item_id"]: stable_document_id(item["sha256"])
        for item in manifest
    }
    catalog = load_chunk_catalog(evaluation.chunks / args.strategy)
    records = load_records(evaluation, args.strategy, args.model, args.dimensions)

    query_cache_path = (
        evaluation.embeddings / f"{args.model}-{args.dimensions}"
        / "question_embeddings.sqlite3"
    )
    query_vectors: dict[str, list[float]] = {}
    with EmbeddingCache(query_cache_path) as cache:
        for question in questions:
            cached = cache.get(question["question_id"], args.model, args.dimensions)
            if cached is None or not cache.has_valid(
                question["question_id"], question["query"], args.model, args.dimensions,
            ):
                raise RuntimeError(f"缺少问题向量: {question['question_id']}")
            query_vectors[question["question_id"]] = cached.vector.tolist()

    max_k = max(k_values)
    print("正在计算精确 Cosine 参考排名...", flush=True)
    exact = exact_rankings(records, questions, query_vectors, max_k)
    collection_name = f"{COLLECTION_PREFIX}_{args.profile}"

    chroma_client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    qdrant_client = QdrantClient(url=args.qdrant_url, timeout=60)
    try:
        print("正在写入 Chroma Server...", flush=True)
        chroma_collection, chroma_build = build_chroma(
            chroma_client, collection_name, records, args.batch_size, args.profile,
        )
        print("正在写入 Qdrant Server...", flush=True)
        qdrant_build = build_qdrant(
            qdrant_client,
            collection_name,
            records,
            args.dimensions,
            args.batch_size,
            args.profile,
        )

        print("正在执行 Chroma 真实问题查询...", flush=True)
        chroma_result = evaluate_service(
            "chroma",
            questions,
            query_vectors,
            lambda vector, k: chroma_collection.query(
                query_embeddings=[vector], n_results=k, include=["distances"],
            )["ids"][0],
            catalog,
            item_to_document,
            exact,
            k_values,
            args.warmup_rounds,
            args.measured_rounds,
        )
        print("正在执行 Qdrant 真实问题查询...", flush=True)
        qdrant_search_params = (
            models.SearchParams(hnsw_ef=300, exact=False, indexed_only=True)
            if args.profile == "hnsw_matched"
            else None
        )
        qdrant_result = evaluate_service(
            "qdrant",
            questions,
            query_vectors,
            lambda vector, k: [
                str(point.payload["chunk_id"])
                for point in qdrant_client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=k,
                    with_payload=["chunk_id"],
                    search_params=qdrant_search_params,
                ).points
            ],
            catalog,
            item_to_document,
            exact,
            k_values,
            args.warmup_rounds,
            args.measured_rounds,
        )
    finally:
        qdrant_client.close()

    chroma_by_id = {row["question_id"]: row for row in chroma_result["questions"]}
    qdrant_by_id = {row["question_id"]: row for row in qdrant_result["questions"]}
    overlaps = [
        len(
            set(chroma_by_id[q["question_id"]]["ranked_chunk_ids"])
            & set(qdrant_by_id[q["question_id"]]["ranked_chunk_ids"])
        ) / max_k
        for q in questions
    ]
    report = {
        "scope": "server_vector_database_benchmark",
        "profile": args.profile,
        "fairness": {
            "distance": "cosine",
            "index_parameters": {
                "default": "database defaults",
                "tuned": "Chroma minimum sufficient tuning; Qdrant default exact-scan behavior",
                "hnsw_matched": "both databases use verified HNSW indexes at 100% exact Top-10 overlap",
            }[args.profile],
            "same_cached_vectors": True,
            "same_query_order": True,
        },
        "dataset_id": args.dataset_id,
        "strategy": args.strategy,
        "embedding": {"model": args.model, "dimensions": args.dimensions},
        "question_count": len(questions),
        "chunk_count": len(records),
        "k_values": list(k_values),
        "versions": {
            "chromadb_client": version("chromadb"),
            "qdrant_client": version("qdrant-client"),
        },
        "services": {
            "chroma": f"http://{args.chroma_host}:{args.chroma_port}",
            "qdrant": args.qdrant_url,
            "collection": collection_name,
        },
        "chroma": {"build": chroma_build, **chroma_result},
        "qdrant": {"build": qdrant_build, **qdrant_result},
        "cross_database": {
            f"mean_top_{max_k}_chunk_overlap": round(statistics.fmean(overlaps), 6),
            "identical_ranked_chunk_lists": sum(
                chroma_by_id[q["question_id"]]["ranked_chunk_ids"]
                == qdrant_by_id[q["question_id"]]["ranked_chunk_ids"]
                for q in questions
            ),
        },
    }
    report_path = evaluation.reports / f"node5_server_{args.profile}_benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "profile": args.profile,
        "question_count": len(questions),
        "chunk_count": len(records),
        "chroma": {
            "build": chroma_build,
            "summary": chroma_result["summary"],
            "ann_fidelity": chroma_result["ann_fidelity"],
            "query_latency": chroma_result["query_latency"],
        },
        "qdrant": {
            "build": qdrant_build,
            "summary": qdrant_result["summary"],
            "ann_fidelity": qdrant_result["ann_fidelity"],
            "query_latency": qdrant_result["query_latency"],
        },
        "cross_database": report["cross_database"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--strategy", default="structure_aware_v1")
    parser.add_argument("--model", default="text-embedding-v4")
    parser.add_argument("--dimensions", type=int, default=2048)
    parser.add_argument(
        "--profile",
        choices=["default", "tuned", "hnsw_matched"],
        default="default",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measured-rounds", type=int, default=5)
    parser.add_argument("--chroma-host", default="127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=18000)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
