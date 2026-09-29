"""在中文多格式评测集上生成切块，并比较切块阶段的证据可用性。"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunk_models import KnowledgeChunk
from ingestion.chunk_registry import build_chunker
from ingestion.chunking import ChunkingConfig
from ingestion.document_model import DocumentFormat, ParsedDocument, stable_document_id
from ingestion.parsers.base import ParseRequest
from ingestion.parsers.local_text import LocalTextParser
from ingestion.storage import StorageLayout


DEFAULT_DATASET_ID = "zh_public_minibench_v0.1"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "eval" / DEFAULT_DATASET_ID
DEFAULT_STRATEGIES = ("fixed_token_v1", "structure_aware_v1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(content + ("\n" if rows else ""), encoding="utf-8")


async def ensure_markdown_parsed(
    manifest: list[dict[str, Any]], dataset_root: Path, parsed_root: Path,
) -> int:
    parser = LocalTextParser()
    created = 0
    for item in manifest:
        if item["format"].lower() not in {"md", "markdown", "txt"}:
            continue
        source_path = dataset_root / item["path"]
        document_id = stable_document_id(item["sha256"])
        destination = parsed_root / document_id / "parsed_document.json"
        if destination.exists():
            continue
        document_format = DocumentFormat.MARKDOWN if item["format"].lower() in {"md", "markdown"} else DocumentFormat.TXT
        document = await parser.parse(ParseRequest(
            source_path=source_path,
            document_format=document_format,
            document_id=document_id,
        ))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        created += 1
    return created


def compact(text: str) -> str:
    """去掉排版差异，但保留汉字、字母、数字与数学符号的语义顺序。"""
    text = text.casefold()
    text = re.sub(r"\\(?:left|right|mathrm|text|mathbf|operatorname)\b", "", text)
    text = text.replace("\\leq", "≤").replace("\\geq", "≥").replace("\\pi", "π")
    return re.sub(r"[\s`*_#$:{}\[\]（）()，,。.!！?？；;：:'\"“”‘’<>|]", "", text)


def ngram_recall(reference: str, candidate: str, n: int = 3) -> float:
    reference_value = compact(reference)
    candidate_value = compact(candidate)
    if not reference_value:
        return 1.0
    if reference_value in candidate_value:
        return 1.0
    size = min(n, len(reference_value))
    reference_grams = {reference_value[index:index + size] for index in range(len(reference_value) - size + 1)}
    if not reference_grams:
        return 0.0
    candidate_grams = {candidate_value[index:index + size] for index in range(max(0, len(candidate_value) - size + 1))}
    return len(reference_grams & candidate_grams) / len(reference_grams)


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def evaluate_strategy(
    strategy: str,
    config: ChunkingConfig,
    documents: dict[str, ParsedDocument],
    manifest: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    chunks_root: Path,
) -> dict[str, Any]:
    chunker = build_chunker(strategy, config)
    item_to_document_id = {item["item_id"]: stable_document_id(item["sha256"]) for item in manifest}
    chunks_by_item: dict[str, list[KnowledgeChunk]] = {}
    chunks_by_document: dict[str, list[KnowledgeChunk]] = {}
    all_chunks: list[KnowledgeChunk] = []

    for item in manifest:
        document_id = item_to_document_id[item["item_id"]]
        chunks = chunks_by_document.get(document_id)
        if chunks is None:
            document = documents[document_id]
            chunks = chunker.chunk(document)
            chunks_by_document[document_id] = chunks
            all_chunks.extend(chunks)
            write_jsonl(
                chunks_root / strategy / document_id / "chunks.jsonl",
                [chunk.model_dump(mode="json") for chunk in chunks],
            )
        chunks_by_item[item["item_id"]] = chunks

    question_results: list[dict[str, Any]] = []
    for question in questions:
        relevant_chunks = [
            chunk
            for item_id in question.get("relevant_document_ids", [])
            for chunk in chunks_by_item.get(item_id, [])
        ]
        evidence_scores = []
        preservation_scores = []
        for evidence in question.get("evidence", []):
            candidates = chunks_by_item.get(evidence.get("document_id"), relevant_chunks)
            evidence_scores.append(max((ngram_recall(evidence.get("quote", ""), chunk.text) for chunk in candidates), default=0.0))
            preservation_scores.append(ngram_recall(
                evidence.get("quote", ""), "\n".join(chunk.text for chunk in candidates),
            ))
        accepted_answers = question.get("accepted_answers") or [question.get("gold_answer", "")]
        answer_score = max(
            (ngram_recall(answer, chunk.text, n=2) for answer in accepted_answers for chunk in relevant_chunks),
            default=0.0,
        )
        evidence_threshold = 0.75
        preservation_threshold = 0.90
        answer_threshold = 0.80
        all_evidence_hit = bool(evidence_scores) and all(score >= evidence_threshold for score in evidence_scores)
        any_evidence_hit = any(score >= evidence_threshold for score in evidence_scores)
        question_results.append({
            "question_id": question["question_id"],
            "evidence_scores": [round(value, 4) for value in evidence_scores],
            "preservation_scores": [round(value, 4) for value in preservation_scores],
            "all_evidence_hit": all_evidence_hit,
            "any_evidence_hit": any_evidence_hit,
            "all_evidence_preserved": bool(preservation_scores) and all(
                score >= preservation_threshold for score in preservation_scores
            ),
            "answer_score": round(answer_score, 4),
            "answer_hit": answer_score >= answer_threshold,
        })

    token_counts = [chunk.token_count for chunk in all_chunks]
    unique_texts = len({compact(chunk.text) for chunk in all_chunks})
    traceable = [chunk for chunk in all_chunks if chunk.element_ids and chunk.source_locations]
    metrics = {
        "document_count": len(documents),
        "question_count": len(question_results),
        "chunk_count": len(all_chunks),
        "mean_tokens": round(statistics.fmean(token_counts), 2) if token_counts else 0.0,
        "p50_tokens": percentile(token_counts, 0.50),
        "p95_tokens": percentile(token_counts, 0.95),
        "max_tokens_observed": max(token_counts, default=0),
        "oversize_chunk_count": sum(value > config.max_tokens for value in token_counts),
        "short_chunk_rate": round(sum(value < 64 for value in token_counts) / len(token_counts), 4) if token_counts else 0.0,
        "exact_duplicate_rate": round(1 - unique_texts / len(all_chunks), 4) if all_chunks else 0.0,
        "traceable_chunk_rate": round(len(traceable) / len(all_chunks), 4) if all_chunks else 0.0,
        "evidence_preservation_rate": round(sum(row["all_evidence_preserved"] for row in question_results) / len(question_results), 4),
        "single_chunk_evidence_coverage": round(sum(row["all_evidence_hit"] for row in question_results) / len(question_results), 4),
        "any_evidence_coverage": round(sum(row["any_evidence_hit"] for row in question_results) / len(question_results), 4),
        "answer_coverage": round(sum(row["answer_hit"] for row in question_results) / len(question_results), 4),
    }
    metrics["selection_score"] = round(
        metrics["evidence_preservation_rate"] * 0.50
        + metrics["answer_coverage"] * 0.20
        + metrics["single_chunk_evidence_coverage"] * 0.15
        + metrics["traceable_chunk_rate"] * 0.15
        - (metrics["oversize_chunk_count"] / max(1, metrics["chunk_count"])) * 0.20,
        4,
    )
    return {
        "strategy": strategy,
        "config": config.model_dump(mode="json"),
        "config_hash": config.fingerprint(),
        "metrics": metrics,
        "failed_evidence_questions": [row["question_id"] for row in question_results if not row["all_evidence_hit"]],
        "failed_preservation_questions": [
            row["question_id"] for row in question_results if not row["all_evidence_preserved"]
        ],
        "failed_answer_questions": [row["question_id"] for row in question_results if not row["answer_hit"]],
        "question_results": question_results,
    }


async def run(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    layout = StorageLayout.from_env(PROJECT_ROOT).evaluation(args.dataset_id)
    layout.ensure()
    manifest = read_jsonl(dataset_root / "manifest.jsonl")
    questions = read_jsonl(dataset_root / "annotations" / "questions.jsonl")
    created = await ensure_markdown_parsed(manifest, dataset_root, layout.parsed)

    documents: dict[str, ParsedDocument] = {}
    missing: list[str] = []
    for item in manifest:
        document_id = stable_document_id(item["sha256"])
        parsed_path = layout.parsed / document_id / "parsed_document.json"
        if not parsed_path.exists():
            missing.append(item["item_id"])
            continue
        documents[document_id] = ParsedDocument.model_validate_json(parsed_path.read_text(encoding="utf-8"))
    if missing:
        raise RuntimeError(f"缺少 {len(missing)} 份解析结果: {missing}")

    config = ChunkingConfig(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        table_rows_per_chunk=args.table_rows_per_chunk,
    )
    results = [
        evaluate_strategy(strategy, config, documents, manifest, questions, layout.chunks)
        for strategy in args.strategies
    ]
    winner = max(results, key=lambda row: row["metrics"]["selection_score"])
    report = {
        "dataset_id": args.dataset_id,
        "dataset_root": str(dataset_root),
        "parsed_document_count": len(documents),
        "dataset_item_count": len(manifest),
        "markdown_documents_created": created,
        "question_count": len(questions),
        "selection_rule": {
            "formula": "0.50*evidence_preservation_rate + 0.20*answer_coverage + 0.15*single_chunk_evidence_coverage + 0.15*traceable_chunk_rate - 0.20*oversize_rate",
            "note": "这里只选择进入向量数据库对比的切块输入，不代表最终召回效果。",
        },
        "recommended_strategy": winner["strategy"],
        "results": results,
    }
    destination = layout.reports / args.report_name
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=80)
    parser.add_argument("--table-rows-per-chunk", type=int, default=20)
    parser.add_argument("--report-name", default="node4_chunking_benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    report_path = asyncio.run(run(parse_args()))
    print(report_path)
