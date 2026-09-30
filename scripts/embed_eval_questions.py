"""批量生成评测问题向量，写入独立 SQLite 缓存并支持断点续传。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.storage import StorageLayout
from retrieval.embedding_cache import EmbeddingCache
from retrieval.compatible_embeddings import CompatibleEmbeddingClient


DEFAULT_DATASET_ID = "zh_public_minibench_v0.1"


def load_questions(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row["question_id"])
        query = str(row["query"]).strip()
        if question_id in seen:
            raise ValueError(f"问题ID重复: {question_id}")
        if not query:
            raise ValueError(f"问题内容为空: {question_id}")
        seen.add(question_id)
        rows.append((question_id, query))
    if not rows:
        raise RuntimeError(f"没有找到评测问题: {path}")
    return rows


def run(args: argparse.Namespace) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_BASE_URL")
    if args.batch_size < 1 or args.batch_size > 10:
        raise ValueError("text-embedding-v4 同步接口 batch_size 必须在 1..10")

    evaluation = StorageLayout.from_env(PROJECT_ROOT).evaluation(args.dataset_id)
    evaluation.ensure()
    question_path = (
        PROJECT_ROOT / "data" / "eval" / args.dataset_id
        / "annotations" / "retrieval_questions.jsonl"
    )
    questions = load_questions(question_path)
    cache_path = (
        evaluation.embeddings / f"{args.model}-{args.dimensions}"
        / "question_embeddings.sqlite3"
    )
    client = CompatibleEmbeddingClient(base_url, api_key, args.model, args.dimensions)
    total_api_tokens = 0
    try:
        with EmbeddingCache(cache_path) as cache:
            before = cache.valid_count(questions, args.model, args.dimensions)
            pending = [
                (item_id, text) for item_id, text in questions
                if not cache.has_valid(item_id, text, args.model, args.dimensions)
            ]
            print(
                f"questions={len(questions)} cached={before} pending={len(pending)} "
                f"model={args.model} dimensions={args.dimensions}",
                flush=True,
            )
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                vectors, token_count = client.embed([text for _, text in batch])
                cache.put_many(
                    [
                        (item_id, text, vector, None)
                        for (item_id, text), vector in zip(batch, vectors)
                    ],
                    args.model,
                    args.dimensions,
                )
                total_api_tokens += token_count or 0
                print(
                    f"progress={before + min(start + len(batch), len(pending))}/{len(questions)} "
                    f"batch_tokens={token_count or 'unknown'} total_run_tokens={total_api_tokens}",
                    flush=True,
                )
            final_count = cache.valid_count(questions, args.model, args.dimensions)
            if final_count != len(questions):
                raise RuntimeError(f"问题向量缓存不完整: {final_count}/{len(questions)}")
            print(
                f"complete={final_count}/{len(questions)} total_run_tokens={total_api_tokens} "
                f"cache={cache_path}",
                flush=True,
            )
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"))
    parser.add_argument("--dimensions", type=int, default=int(os.getenv("EMBEDDING_DIMENSIONS", "2048")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
