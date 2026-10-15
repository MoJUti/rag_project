"""在开发题上比较中文词法召回与融合参数，并单独报告保留题。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
from retrieval.hybrid_retriever import _char_tokenize
from retrieval.tokenizers import mixed_tokenize
from scripts.evaluate_embedded_retrieval import evaluate_ranking, load_chunk_catalog
from scripts.run_chunking_benchmark import read_jsonl
from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings


def fuse_ids(
    dense: list[str], lexical: list[str], weight: float, candidate_k: int, top_k: int = 10,
) -> list[str]:
    """对相同候选数的两路结果执行带权RRF，分数相同时使用稳定ID排序。"""
    scores: dict[str, float] = {}
    for ids, factor in ((dense[:candidate_k], 1.0), (lexical[:candidate_k], weight)):
        for rank, chunk_id in enumerate(ids, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + factor / (60 + rank)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]


def summarize(
    questions: list[dict], rankings: dict[str, list[str]], catalog: dict, item_to_document: dict,
) -> dict[str, float]:
    """聚合少数决策指标，完整逐题结果留给现有基线报告。"""
    names = ("mrr", "recall_at_6", "evidence_recall_at_6", "complete_evidence_hit_at_6")
    totals = dict.fromkeys(names, 0.0)
    for question in questions:
        metrics = evaluate_ranking(
            question, rankings[question["question_id"]], catalog, item_to_document, (6,),
        )
        for name in names:
            totals[name] += metrics[name]
    return {name: round(value / len(questions), 6) for name, value in totals.items()}


def main() -> None:
    dataset_id = "zh_public_minibench_v0.1"
    evaluation = StorageLayout.from_env(ROOT).evaluation(dataset_id)
    dataset = ROOT / "data" / "eval" / dataset_id
    questions = read_jsonl(dataset / "annotations" / "retrieval_questions.jsonl")
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    manifest = read_jsonl(dataset / "manifest.jsonl")
    item_to_document = {row["item_id"]: stable_document_id(row["sha256"]) for row in manifest}
    model, dimensions = config.embedding_model_name, config.embedding_dimensions
    vectors: dict[str, list[float]] = {}
    cache_path = evaluation.embeddings / f"{model}-{dimensions}" / "question_embeddings.sqlite3"
    with EmbeddingCache(cache_path) as cache:
        for question in questions:
            key, query = question["question_id"], question["query"]
            if not cache.has_valid(key, query, model, dimensions):
                raise ValueError(f"问题向量缓存不匹配：{key}")
            vectors[query] = cache.get(key, model, dimensions).vector.tolist()

    service = VectorStoreService(adapter=ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors)))
    if service.count() != len(catalog):
        raise ValueError("正式向量库的切块数量与评测集不一致")
    documents = service.get_all_documents()
    lexical = {
        "char": BM25Retriever.from_documents(documents, preprocess_func=_char_tokenize),
        "mixed": BM25Retriever.from_documents(documents, preprocess_func=mixed_tokenize),
    }
    for retriever in lexical.values():
        retriever.k = 40

    dense: dict[str, list[str]] = {}
    lexical_rankings: dict[str, dict[str, list[str]]] = {name: {} for name in lexical}
    for question in questions:
        key, query = question["question_id"], question["query"]
        dense[key] = [doc.metadata["chunk_id"] for doc in service.get_vector_docs(query, 40)]
        for name, retriever in lexical.items():
            lexical_rankings[name][key] = [doc.metadata["chunk_id"] for doc in retriever.invoke(query)]
        result_groups = {"dense": dense[key], **{
            name: ranked[key] for name, ranked in lexical_rankings.items()
        }}
        for name, ids in result_groups.items():
            if len(ids) != 40 or len(set(ids)) != 40 or set(ids) - catalog.keys():
                raise ValueError(f"{name} 的检索结果数量、去重或切块ID异常：{key}")

    splits = {
        "development": [row for row in questions if row["split"] in {"dev", "smoke"}],
        "holdout": [row for row in questions if row["split"] == "holdout-public"],
    }
    if sum(map(len, splits.values())) != len(questions) or not all(splits.values()):
        raise ValueError("评测题的开发集或保留集划分异常")
    variants: dict[str, dict[str, list[str]]] = {"dense": {k: ids[:10] for k, ids in dense.items()}}
    for name, ranked in lexical_rankings.items():
        variants[f"bm25_{name}"] = {key: ids[:10] for key, ids in ranked.items()}
        for candidate_k in (10, 20, 40):
            for weight in (0.25, 0.5, 1.0, 2.0):
                variant = f"rrf_{name}_k{candidate_k}_w{weight:g}"
                variants[variant] = {
                    key: fuse_ids(dense[key], ranked[key], weight, candidate_k)
                    for key in dense
                }
    report = {
        "dataset_id": dataset_id,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "candidate_count": 40,
        "rrf_constant": 60,
        "results": {
            name: {
                split: summarize(rows, rankings, catalog, item_to_document)
                for split, rows in splits.items()
            }
            for name, rankings in variants.items()
        },
    }
    evaluation.reports.mkdir(parents=True, exist_ok=True)
    output = evaluation.reports / "node6_fusion_experiment.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ranking = sorted(
        report["results"],
        key=lambda name: (
            -report["results"][name]["development"]["evidence_recall_at_6"],
            -report["results"][name]["development"]["mrr"],
        ),
    )
    for name in ranking[:12]:
        print(json.dumps({"variant": name, **report["results"][name]}, ensure_ascii=False))
    print(f"report={output}")


if __name__ == "__main__":
    main()
