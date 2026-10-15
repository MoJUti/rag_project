"""复用固定检索排名，比较邻接扩展和继续取排名候选的上下文收益。"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.chunk_models import KnowledgeChunk
from ingestion.chunking import TokenCounter
from ingestion.document_model import stable_document_id
from ingestion.storage import StorageLayout
from retrieval.evidence_expansion import NeighborhoodExpander
from scripts.evaluate_node6_acceptance import score_question
from scripts.run_chunking_benchmark import ngram_recall, percentile, read_jsonl


def original_context_score(question: dict, ids: list[str], catalog: dict, mapping: dict) -> dict[str, float]:
    """旧套件在同来源的完整上下文上核对引文，允许证据跨块分布。"""
    hits = []
    for evidence in question["evidence"]:
        document_id = mapping[evidence["document_id"]]
        text = "\n".join(catalog[key]["text"] for key in ids
                         if catalog[key]["document_id"] == document_id)
        hits.append(ngram_recall(evidence["quote"], text) >= 0.75)
    return {"evidence_recall": sum(hits) / len(hits), "complete_evidence": float(all(hits))}


def main() -> None:
    bundle = ROOT / "data/eval/zh_public_minibench_v0.1"
    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    directory = evaluation.chunks / "structure_aware_v1"
    chunks = [KnowledgeChunk.model_validate(row) for path in sorted(directory.rglob("chunks.jsonl"))
              for row in read_jsonl(path)]
    counter = TokenCounter()
    expander = NeighborhoodExpander(chunks, radius=1, extra_token_budget=2000,
                                   count_tokens=counter.count)
    catalog = {chunk.chunk_id: {"document_id": chunk.document_id, "text": chunk.text} for chunk in chunks}
    documents = {chunk.chunk_id: Document(page_content=chunk.text, metadata={
        "chunk_id": chunk.chunk_id, "document_id": chunk.document_id,
        "token_count": counter.count(chunk.text),
    }) for chunk in chunks}
    token_counts = {key: doc.metadata["token_count"] for key, doc in documents.items()}
    mapping = {row["item_id"]: stable_document_id(row["sha256"]) for row in read_jsonl(bundle / "manifest.jsonl")}
    original = json.loads((evaluation.reports / "node6_retrieval_baseline_after_cache_fix.json").read_text(encoding="utf-8"))
    acceptance = json.loads((evaluation.reports / "node6_acceptance_v1_report.json").read_text(encoding="utf-8"))
    suites = [("original55", read_jsonl(bundle / "annotations/retrieval_questions.jsonl"), original),
              ("acceptance19", [row for row in read_jsonl(bundle / "annotations/acceptance_questions_v1.jsonl")
                                if row["answerable"]], acceptance)]
    report = {"radius": 1, "extra_token_budget": 2000,
              "token_estimator": "cl100k_base（可用时），不等于千问实际计费用量", "suites": {}}
    for suite, questions, rankings in suites:
        indexed_questions = {row["question_id"]: row for row in questions}
        methods = {}
        for method, results in rankings["methods"].items():
            rows = []
            for result in results["questions"]:
                question = indexed_questions[result["question_id"]]
                ranked_ids = result["ranked_chunk_ids"]
                seeds = ranked_ids[:6]
                expanded = expander(question["query"], [documents[key] for key in seeds])
                expanded_ids = [doc.metadata["chunk_id"] for doc in expanded]
                # 旧55题缓存有前10名；新增套件从阶段追踪取完整排序池。
                pool = ranked_ids
                if suite == "acceptance19":
                    stages = [stage for stage in result["trace"] if stage["stage"] in {"rerank", "fusion"}]
                    pool = stages[-1]["chunk_ids"] if stages else ranked_ids
                more_ranked = list(seeds)
                remaining = 2000
                for key in pool[6:]:
                    if key not in more_ranked and token_counts[key] <= remaining:
                        more_ranked.append(key)
                        remaining -= token_counts[key]
                contexts = {"seed6": seeds, "neighbors": expanded_ids, "more_ranked": more_ranked}
                row = {"question_id": question["question_id"],
                       "cached_candidate_count": len(pool), "contexts": {}}
                for variant, ids in contexts.items():
                    if suite == "original55":
                        metrics = original_context_score(question, ids, catalog, mapping)
                    else:
                        score = score_question(question, ids, catalog, mapping)
                        metrics = {"evidence_recall": score["evidence_recall_at_6"],
                                   "complete_evidence": score["complete_evidence_at_6"]}
                    added = [key for key in ids if key not in seeds]
                    tokens = sum(token_counts[key] for key in added)
                    if tokens > 2000:
                        raise RuntimeError(f"扩展超出Token预算：{question['question_id']}")
                    row["contexts"][variant] = {"chunk_ids": ids, "added_tokens": tokens,
                                                 "total_tokens": sum(token_counts[key] for key in ids), **metrics}
                rows.append(row)
            summary = {}
            for variant in ("seed6", "neighbors", "more_ranked"):
                values = [row["contexts"][variant] for row in rows]
                summary[variant] = {metric: round(statistics.fmean(value[metric] for value in values), 6)
                                    for metric in ("evidence_recall", "complete_evidence", "added_tokens", "total_tokens")}
                summary[variant]["added_tokens_p95"] = percentile([value["added_tokens"] for value in values], 0.95)
            methods[method] = {"summary": summary, "questions": rows,
                               "cases_with_more_cached_candidates": sum(
                                   row["cached_candidate_count"] > 6 for row in rows)}
            print(json.dumps({"suite": suite, "method": method, "summary": summary}, ensure_ascii=False), flush=True)
        report["suites"][suite] = methods
    output = evaluation.reports / "node6_evidence_expansion_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={output}")


if __name__ == "__main__":
    main()
