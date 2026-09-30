"""将结构感知切块和已校验缓存导入应用配置的 Chroma，绝不调用嵌入 API。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import config
from ingestion.storage import StorageLayout
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from retrieval.embedding_cache import EmbeddingCache


class OfflineEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise RuntimeError("导入只允许使用缓存向量")

    def embed_query(self, text):
        raise RuntimeError("验证时必须显式提供缓存查询向量")


def main():
    if config.vector_store_backend != "chroma":
        raise RuntimeError("当前应用后端不是 Chroma")
    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    cache_path = evaluation.embeddings / f"{config.embedding_model_name}-{config.embedding_dimensions}" / "embeddings.sqlite3"
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    records = {}
    sources = {}
    with EmbeddingCache(cache_path) as cache:
        for path in sorted((evaluation.chunks / "structure_aware_v1").rglob("chunks.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key, text = row["chunk_id"], row["text"]
                if key in records:
                    raise ValueError(f"重复切块 ID: {key}")
                if not cache.has_valid(key, text, config.embedding_model_name, config.embedding_dimensions):
                    raise ValueError(f"缓存缺失或文本不匹配: {key}")
                vector = cache.get(key, config.embedding_model_name, config.embedding_dimensions).vector
                if not np.isfinite(vector).all():
                    raise ValueError(f"向量包含非有限值: {key}")
                if row["document_id"] not in sources:
                    parsed = evaluation.parsed / row["document_id"] / "parsed_document.json"
                    sources[row["document_id"]] = json.loads(parsed.read_text(encoding="utf-8"))["source"]
                source = sources[row["document_id"]]
                metadata = {
                    "source": source["filename"], "source_uri": source["source_uri"],
                    "document_id": row["document_id"], "chunk_id": key,
                    "strategy": row["strategy"], "chunk_type": row["chunk_type"],
                    "token_count": row["token_count"],
                    "embedding_model": config.embedding_model_name,
                    "embedding_dimensions": config.embedding_dimensions,
                    "vector_schema_version": config.vector_schema_version,
                    "dataset_id": "zh_public_minibench_v0.1",
                }
                for field in ("source_locations", "element_ids", "section_path", "asset_ids", "metadata"):
                    metadata[field + "_json"] = json.dumps(row.get(field), ensure_ascii=False)
                records[key] = (text, metadata, vector)
    if len(records) != 2881:
        raise ValueError(f"预期2881条，实际{len(records)}条；拒绝写入")
    print(f"validated_records={len(records)}", flush=True)
    adapter = ChromaVectorStoreAdapter(OfflineEmbeddings())
    collection = adapter.store._collection
    extra = set(collection.get(include=[])["ids"]) - records.keys()
    if extra:
        raise RuntimeError("目标含其他数据，拒绝混入；请检查集合配置")
    keys = sorted(records)
    for start in range(0, len(keys), 100):
        batch = keys[start:start + 100]
        collection.upsert(ids=batch, documents=[records[k][0] for k in batch],
                          metadatas=[records[k][1] for k in batch],
                          embeddings=[records[k][2].tolist() for k in batch])
    # 全量核对文本、来源、向量，而不只核对条数。
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    assert set(data["ids"]) == set(records)
    for i, key in enumerate(data["ids"]):
        text, metadata, vector = records[key]
        assert data["documents"][i] == text
        assert data["metadatas"][i] == metadata
        np.testing.assert_allclose(data["embeddings"][i], vector, rtol=1e-5, atol=1e-7)
    # 经应用服务入口验证读取、查询及来源过滤；使用缓存向量，不消耗 token。
    sample = keys[0]
    adapter.embedding.embed_query = lambda text: records[sample][2].tolist()
    service = VectorStoreService(adapter=adapter)
    assert len(service.get_all_documents()) == len(records)
    assert service.get_vector_docs("缓存查询验证", 5)
    source = records[sample][1]["source"]
    filtered = adapter.similarity_search("来源过滤验证", 5, source)
    assert filtered and all(d.metadata["source"] == source for d in filtered)
    print(json.dumps({"health": service.health_check(), "verified_records": len(records),
                      "documents": len({r[1]['document_id'] for r in records.values()}),
                      "api_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
