"""用同一批缓存向量预检 Chroma 与 Qdrant 的嵌入式本地模式。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.storage import StorageLayout
from retrieval.embedding_cache import EmbeddingCache
from scripts.embed_eval_chunks import load_chunks


DEFAULT_DATASET_ID = "zh_public_minibench_v0.1"
COLLECTION_NAME = "structure_aware_v1_text_embedding_v4_2048"
QDRANT_ID_NAMESPACE = uuid.UUID("0f3d6b2e-fbe0-4fbd-a111-a219d99fc01d")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
    }


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(QDRANT_ID_NAMESPACE, chunk_id))


def load_records(
    evaluation: Any,
    strategy: str,
    model: str,
    dimensions: int,
) -> list[dict[str, Any]]:
    chunks = load_chunks(evaluation.chunks / strategy)
    cache_path = evaluation.embeddings / f"{model}-{dimensions}" / "embeddings.sqlite3"
    records: list[dict[str, Any]] = []
    with EmbeddingCache(cache_path) as cache:
        for chunk_id, text, token_count in chunks:
            cached = cache.get(chunk_id, model, dimensions)
            if cached is None or not cache.has_valid(chunk_id, text, model, dimensions):
                raise RuntimeError(f"缺少或失效的缓存向量: {chunk_id}")
            records.append({
                "chunk_id": chunk_id,
                "text": text,
                "token_count": token_count,
                "vector": cached.vector,
            })
    return records


def build_chroma(path: Path, records: list[dict[str, Any]], batch_size: int) -> tuple[Any, float]:
    client = chromadb.PersistentClient(path=str(path))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
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
    return collection, time.perf_counter() - started


def build_qdrant(
    path: Path,
    records: list[dict[str, Any]],
    dimensions: int,
    batch_size: int,
) -> tuple[QdrantClient, float]:
    client = QdrantClient(path=str(path))
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
    )
    started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[
                models.PointStruct(
                    id=qdrant_point_id(row["chunk_id"]),
                    vector=row["vector"].tolist(),
                    payload={
                        "chunk_id": row["chunk_id"],
                        "text": row["text"],
                        "token_count": row["token_count"],
                    },
                )
                for row in batch
            ],
        )
    return client, time.perf_counter() - started


def select_samples(records: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size >= len(records):
        return records
    indexes = np.linspace(0, len(records) - 1, num=sample_size, dtype=int)
    return [records[int(index)] for index in indexes]


def run(args: argparse.Namespace) -> None:
    evaluation = StorageLayout.from_env(PROJECT_ROOT).evaluation(args.dataset_id)
    evaluation.ensure()
    records = load_records(evaluation, args.strategy, args.model, args.dimensions)
    samples = select_samples(records, args.sample_size)
    index_root = evaluation.root / "indexes" / "embedded_preflight"
    chroma_path = index_root / "chroma"
    qdrant_path = index_root / "qdrant"
    chroma_path.mkdir(parents=True, exist_ok=True)
    qdrant_path.mkdir(parents=True, exist_ok=True)

    print(f"records={len(records)} samples={len(samples)} dimensions={args.dimensions}", flush=True)
    print("正在写入 Chroma 嵌入式索引...", flush=True)
    chroma_collection, chroma_insert_seconds = build_chroma(chroma_path, records, args.batch_size)
    print("正在写入 Qdrant Local Mode 索引...", flush=True)
    qdrant_client, qdrant_insert_seconds = build_qdrant(
        qdrant_path, records, args.dimensions, args.batch_size,
    )

    # 重新获取集合句柄，确保验证阶段读取的是完整提交后的持久化集合状态。
    chroma_collection = chromadb.PersistentClient(path=str(chroma_path)).get_collection(COLLECTION_NAME)

    chroma_count = chroma_collection.count()
    qdrant_count = qdrant_client.count(COLLECTION_NAME, exact=True).count
    chroma_latencies: list[float] = []
    qdrant_latencies: list[float] = []
    chroma_top1_hits = 0
    qdrant_top1_hits = 0
    chroma_top_k_hits = 0
    qdrant_top_k_hits = 0
    overlaps: list[float] = []
    top1_mismatches: list[dict[str, Any]] = []

    print("正在执行相同向量的抽样查询...", flush=True)
    for row in samples:
        vector = row["vector"].tolist()
        started = time.perf_counter()
        chroma_result = chroma_collection.query(
            query_embeddings=[vector], n_results=args.top_k, include=["distances"],
        )
        chroma_latencies.append((time.perf_counter() - started) * 1000)
        chroma_ids = chroma_result["ids"][0]

        started = time.perf_counter()
        qdrant_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=args.top_k,
            with_payload=["chunk_id"],
        )
        qdrant_latencies.append((time.perf_counter() - started) * 1000)
        qdrant_ids = [str(point.payload["chunk_id"]) for point in qdrant_result.points]

        chroma_top1_hits += int(bool(chroma_ids) and chroma_ids[0] == row["chunk_id"])
        qdrant_top1_hits += int(bool(qdrant_ids) and qdrant_ids[0] == row["chunk_id"])
        chroma_top_k_hits += int(row["chunk_id"] in chroma_ids)
        qdrant_top_k_hits += int(row["chunk_id"] in qdrant_ids)
        overlaps.append(len(set(chroma_ids) & set(qdrant_ids)) / args.top_k)
        if (
            not chroma_ids or chroma_ids[0] != row["chunk_id"]
            or not qdrant_ids or qdrant_ids[0] != row["chunk_id"]
        ):
            top1_mismatches.append({
                "query_chunk_id": row["chunk_id"],
                "chroma_top_ids": chroma_ids,
                "qdrant_top_ids": qdrant_ids,
            })

    qdrant_client.close()

    # 持久化复开检查：重新创建客户端并读取集合计数。
    reopened_chroma_count = chromadb.PersistentClient(path=str(chroma_path)).get_collection(COLLECTION_NAME).count()
    reopened_qdrant = QdrantClient(path=str(qdrant_path))
    reopened_qdrant_count = reopened_qdrant.count(COLLECTION_NAME, exact=True).count
    reopened_qdrant.close()

    report = {
        "scope": "embedded_functional_preflight",
        "warning": "本地嵌入模式耗时不代表独立服务性能，不能据此选择最终数据库。",
        "dataset_id": args.dataset_id,
        "strategy": args.strategy,
        "embedding": {"model": args.model, "dimensions": args.dimensions},
        "record_count": len(records),
        "query": {"sample_count": len(samples), "top_k": args.top_k, "type": "stored_vector_self_query"},
        "versions": {
            "chromadb": version("chromadb"),
            "qdrant-client": version("qdrant-client"),
        },
        "chroma": {
            "count": chroma_count,
            "reopened_count": reopened_chroma_count,
            "insert_seconds": round(chroma_insert_seconds, 3),
            "top1_self_hit_rate": round(chroma_top1_hits / len(samples), 4),
            "top_k_self_hit_rate": round(chroma_top_k_hits / len(samples), 4),
            "query_latency": latency_summary(chroma_latencies),
            "path": str(chroma_path),
        },
        "qdrant": {
            "count": qdrant_count,
            "reopened_count": reopened_qdrant_count,
            "insert_seconds": round(qdrant_insert_seconds, 3),
            "top1_self_hit_rate": round(qdrant_top1_hits / len(samples), 4),
            "top_k_self_hit_rate": round(qdrant_top_k_hits / len(samples), 4),
            "query_latency": latency_summary(qdrant_latencies),
            "path": str(qdrant_path),
        },
        "cross_database": {
            "mean_top_k_overlap": round(statistics.fmean(overlaps), 4),
            "all_counts_match": chroma_count == qdrant_count == len(records),
            "all_reopen_counts_match": reopened_chroma_count == reopened_qdrant_count == len(records),
            "top1_mismatches": top1_mismatches,
        },
    }
    report_path = evaluation.reports / "node5_embedded_vector_preflight.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"report={report_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--strategy", default="structure_aware_v1")
    parser.add_argument("--model", default="text-embedding-v4")
    parser.add_argument("--dimensions", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
