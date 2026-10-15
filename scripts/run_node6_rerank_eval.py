"""在固定候选池上对照无重排与千问文本重排，并保存用量以便续跑。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from ingestion.document_model import stable_document_id
from ingestion.storage import StorageLayout
from retrieval.embedding_cache import EmbeddingCache
from retrieval.qwen_reranker import QwenTextReranker
from scripts.evaluate_embedded_retrieval import aggregate, evaluate_ranking, load_chunk_catalog
from scripts.run_chunking_benchmark import percentile, read_jsonl
from scripts.run_node6_fusion_experiment import fuse_ids, mixed_tokenize
from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings


def save_cache(path: Path, rows: dict[str, dict]) -> None:
    """每题完成后原子保存，避免中断时丢失已付费结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def main(limit: int) -> None:
    load_dotenv(ROOT / ".env")
    dataset_id = "zh_public_minibench_v0.1"
    evaluation = StorageLayout.from_env(ROOT).evaluation(dataset_id)
    dataset = ROOT / "data" / "eval" / dataset_id
    questions = read_jsonl(dataset / "annotations" / "retrieval_questions.jsonl")
    manifest = read_jsonl(dataset / "manifest.jsonl")
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    item_to_document = {row["item_id"]: stable_document_id(row["sha256"]) for row in manifest}
    cache_path = evaluation.embeddings / (
        f"{config.embedding_model_name}-{config.embedding_dimensions}/question_embeddings.sqlite3"
    )
    vectors: dict[str, list[float]] = {}
    with EmbeddingCache(cache_path) as cache:
        for question in questions:
            key, query = question["question_id"], question["query"]
            if not cache.has_valid(key, query, config.embedding_model_name, config.embedding_dimensions):
                raise ValueError(f"问题向量缓存不匹配：{key}")
            vectors[query] = cache.get(
                key, config.embedding_model_name, config.embedding_dimensions,
            ).vector.tolist()

    service = VectorStoreService(adapter=ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors)))
    if service.count() != len(catalog):
        raise ValueError("正式向量库的切块数量与评测集不一致")
    lexical = BM25Retriever.from_documents(service.get_all_documents(), preprocess_func=mixed_tokenize)
    lexical.k = 20
    result_path = evaluation.reports / "node6_qwen37_rerank_cache.json"
    rows: dict[str, dict] = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    selected = questions[:limit] if limit else questions
    with QwenTextReranker() as reranker:
        for question in selected:
            key, query = question["question_id"], question["query"]
            if key in rows:
                if rows[key].get("query") != query:
                    raise ValueError(f"已缓存问题文本与当前评测集不一致：{key}")
                if set(rows[key].get("reranked_top10", [])) - catalog.keys():
                    raise ValueError(f"已缓存重排切块与当前评测集不一致：{key}")
                print(f"cached={key}", flush=True)
                continue
            dense = [doc.metadata["chunk_id"] for doc in service.get_vector_docs(query, 20)]
            lexical_ids = [doc.metadata["chunk_id"] for doc in lexical.invoke(query)]
            pool = fuse_ids(dense, lexical_ids, 1.0, 20, top_k=40)
            if len(dense) != 20 or len(lexical_ids) != 20 or len(pool) < 20:
                raise ValueError(f"重排候选数量异常：{key}")
            if len(set(pool)) != len(pool) or set(pool) - catalog.keys():
                raise ValueError(f"重排候选ID异常：{key}")
            started = time.perf_counter()
            result = reranker.rerank(query, [catalog[chunk_id]["text"] for chunk_id in pool])
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            rows[key] = {
                "query": query,
                "split": question["split"],
                "dense_top10": dense[:10],
                "fusion_top10": pool[:10],
                "reranked_top10": [pool[index] for index in result.indices[:10]],
                "candidate_ids": pool,
                "reranked_ids": [pool[index] for index in result.indices],
                "candidate_count": len(pool),
                "total_tokens": result.total_tokens,
                "latency_ms": latency_ms,
                "request_id": result.request_id,
            }
            save_cache(result_path, rows)
            print(json.dumps({"question_id": key, "candidate_count": len(pool),
                              "total_tokens": result.total_tokens, "latency_ms": latency_ms},
                             ensure_ascii=False), flush=True)

    report = {"question_count": len(rows), "total_tokens": sum(row["total_tokens"] for row in rows.values()),
              "methods": {}, "splits": {}, "paired_changes": {}}
    metrics_by_method: dict[str, dict[str, dict]] = {}
    for name, field in (("dense", "dense_top10"), ("fusion", "fusion_top10"),
                        ("qwen3.7_text_rerank", "reranked_top10")):
        evaluated = []
        metrics_by_method[name] = {}
        for question in questions:
            if question["question_id"] not in rows:
                continue
            ranking = rows[question["question_id"]][field]
            metrics = evaluate_ranking(
                question, ranking, catalog, item_to_document, (1, 3, 5, 6, 10),
            )
            metrics_by_method[name][question["question_id"]] = metrics
            evaluated.append({"metrics": metrics})
        report["methods"][name] = aggregate(evaluated, (1, 3, 5, 6, 10))
        report["splits"][name] = {}
        for split in ("dev", "smoke", "holdout-public"):
            split_rows = [{"metrics": metrics_by_method[name][question["question_id"]]}
                          for question in questions
                          if question["split"] == split and question["question_id"] in rows]
            if split_rows:
                report["splits"][name][split] = aggregate(split_rows, (6,))
    for metric in ("mrr", "recall_at_6", "evidence_recall_at_6", "complete_evidence_hit_at_6"):
        improved, unchanged, regressed = [], [], []
        for question in questions:
            key = question["question_id"]
            if key not in rows:
                continue
            difference = (metrics_by_method["qwen3.7_text_rerank"][key][metric]
                          - metrics_by_method["fusion"][key][metric])
            (improved if difference > 0 else regressed if difference < 0 else unchanged).append(key)
        report["paired_changes"][metric] = {
            "improved": improved, "unchanged_count": len(unchanged), "regressed": regressed,
        }
    latencies = [row["latency_ms"] for row in rows.values()]
    report["rerank_latency_ms"] = {
        "p50": round(percentile(latencies, 0.5), 3),
        "p95": round(percentile(latencies, 0.95), 3),
    }
    report["estimated_yuan_at_0.5_per_million_tokens"] = round(report["total_tokens"] * 0.5 / 1_000_000, 4)
    report_path = evaluation.reports / "node6_qwen37_rerank_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"question_count": report["question_count"],
                      "total_tokens": report["total_tokens"],
                      "estimated_yuan": report["estimated_yuan_at_0.5_per_million_tokens"],
                      "paired_changes": {key: {"improved": len(value["improved"]),
                                               "regressed": len(value["regressed"])}
                                         for key, value in report["paired_changes"].items()}},
                     ensure_ascii=False), flush=True)
    print(f"cache={result_path}\nreport={report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="只运行前N题；0表示全部")
    arguments = parser.parse_args()
    if arguments.limit < 0:
        parser.error("--limit 不能小于0")
    main(arguments.limit)
