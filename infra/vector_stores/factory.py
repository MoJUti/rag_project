"""向量数据库适配器工厂。"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from core import config
from infra.vector_stores.base import VectorStoreAdapter


def create_vector_store(
    embedding: Embeddings,
    backend: str | None = None,
) -> VectorStoreAdapter:
    selected = (backend or config.vector_store_backend).strip().lower()
    if selected == "chroma":
        from infra.vector_stores.chroma import ChromaVectorStoreAdapter

        return ChromaVectorStoreAdapter(embedding)
    if selected == "qdrant":
        from infra.vector_stores.qdrant import QdrantVectorStoreAdapter

        return QdrantVectorStoreAdapter(embedding)
    raise ValueError("VECTOR_STORE 仅支持 chroma 或 qdrant")
