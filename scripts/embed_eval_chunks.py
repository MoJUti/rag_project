"""批量生成评测 Chunk 向量，逐批写入 SQLite 并支持断点续传。"""

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
from retrieval.compatible_embeddings import CompatibleEmbeddingClient
from retrieval.embedding_cache import EmbeddingCache


DEFAULT_DATASET_ID = "zh_public_minibench_v0.1"


def load_chunks(chunks_root: Path) -> list[tuple[str, str, int]]:
    chunks: dict[str, tuple[str, int]] = {}
    for path in sorted(chunks_root.rglob("chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = row["chunk_id"]
            text = row["text"]
            token_count = int(row.get("token_count") or 0)
            previous = chunks.get(chunk_id)
            if previous and previous[0] != text:
                raise ValueError(f"相同 chunk_id 对应不同文本: {chunk_id}")
            chunks[chunk_id] = (text, token_count)
    if not chunks:
        raise RuntimeError(f"没有找到切块结果: {chunks_root}")
    return [(chunk_id, text, token_count) for chunk_id, (text, token_count) in sorted(chunks.items())]


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
    chunks_root = evaluation.chunks / args.strategy
    chunks = load_chunks(chunks_root)
    cache_path = evaluation.embeddings / f"{args.model}-{args.dimensions}" / "embeddings.sqlite3"
    items = [(chunk_id, text) for chunk_id, text, _ in chunks]
    estimated_tokens = sum(token_count for _, _, token_count in chunks)

    print(f"chunks={len(chunks)} model={args.model} dimensions={args.dimensions} batch={args.batch_size}")
    print(f"estimated_chunk_tokens={estimated_tokens} cache={cache_path}")

    client = CompatibleEmbeddingClient(base_url, api_key, args.model, args.dimensions)
    total_api_tokens = 0
    try:
        with EmbeddingCache(cache_path) as cache:
            before = cache.valid_count(items, args.model, args.dimensions)
            pending = [(item_id, text) for item_id, text in items if not cache.has_valid(item_id, text, args.model, args.dimensions)]
            print(f"cached={before} pending={len(pending)}")
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                texts = [text for _, text in batch]
                vectors, token_count = client.embed(texts)
                cache.put_many(
                    [(item_id, text, vector, None) for (item_id, text), vector in zip(batch, vectors)],
                    args.model,
                    args.dimensions,
                )
                total_api_tokens += token_count or 0
                completed = before + min(start + len(batch), len(pending))
                print(
                    f"progress={completed}/{len(items)} batch_tokens={token_count or 'unknown'} total_run_tokens={total_api_tokens}",
                    flush=True,
                )
            final_count = cache.valid_count(items, args.model, args.dimensions)
            if final_count != len(items):
                raise RuntimeError(f"缓存不完整: {final_count}/{len(items)}")
            print(f"complete={final_count}/{len(items)} total_run_tokens={total_api_tokens}")
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--strategy", default="structure_aware_v1")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"))
    parser.add_argument("--dimensions", type=int, default=int(os.getenv("EMBEDDING_DIMENSIONS", "2048")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
