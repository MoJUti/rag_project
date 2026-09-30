"""Chroma 适配器：默认本地持久化，也可连接独立 HTTP 服务。"""

from __future__ import annotations

from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core import config
from infra.vector_stores.base import VectorStoreAdapter, VectorStoreHealth


class ChromaVectorStoreAdapter(VectorStoreAdapter):
    backend_name = "chroma"

    def __init__(self, embedding: Embeddings) -> None:
        self.embedding = embedding
        self.collection_name = config.collection_name
        self.mode = config.chroma_mode
        if self.mode == "local":
            self.client = chromadb.PersistentClient(path=config.persist_directory)
        elif self.mode == "http":
            self.client = chromadb.HttpClient(
                host=config.chroma_host,
                port=config.chroma_port,
                ssl=config.chroma_ssl,
            )
        else:
            raise ValueError("CHROMA_MODE 仅支持 local 或 http")

        existing = {collection.name for collection in self.client.list_collections()}
        kwargs: dict[str, Any] = {}
        if self.collection_name not in existing:
            kwargs["collection_configuration"] = {
                "hnsw": {
                    "space": "cosine",
                    "max_neighbors": config.chroma_hnsw_m,
                    "ef_construction": config.chroma_hnsw_ef_construction,
                    "ef_search": config.chroma_hnsw_ef_search,
                }
            }
        self.store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            **kwargs,
        )
        self._validate_existing_dimension()

    def _validate_existing_dimension(self) -> None:
        if self.store._collection.count() == 0:  # noqa: SLF001
            return
        data = self.store._collection.get(limit=1, include=["embeddings"])  # noqa: SLF001
        embeddings = data.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return
        actual = len(embeddings[0])
        if actual != config.embedding_dimensions:
            raise ValueError(
                f"Chroma 集合 {self.collection_name!r} 是 {actual} 维，"
                f"当前配置是 {config.embedding_dimensions} 维。请更换 VECTOR_COLLECTION，"
                "不要把不同维度的向量写入同一集合。"
            )

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        return self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int,
        source_filter: str = "",
    ) -> list[Document]:
        where = {"source": source_filter} if source_filter else None
        return self.store.similarity_search(query=query, k=k, filter=where)

    def get_all_documents(self) -> list[Document]:
        data = self.store._collection.get(include=["documents", "metadatas"])  # noqa: SLF001
        return [
            Document(page_content=text or "", metadata=metadata or {})
            for text, metadata in zip(
                data.get("documents") or [],
                data.get("metadatas") or [],
            )
        ]

    def count(self) -> int:
        return self.store._collection.count()  # noqa: SLF001

    def health_check(self) -> VectorStoreHealth:
        try:
            count = self.count()
            self.client.heartbeat()
            return VectorStoreHealth(
                backend=self.backend_name,
                status="ok",
                collection=self.collection_name,
                vector_count=count,
                vector_dimensions=config.embedding_dimensions,
                embedding_model=config.embedding_model_name,
                schema_version=config.vector_schema_version,
                mode=self.mode,
            )
        except Exception as exc:  # 健康检查必须返回状态，不把页面直接打崩
            return VectorStoreHealth(
                backend=self.backend_name,
                status="error",
                collection=self.collection_name,
                vector_count=0,
                vector_dimensions=config.embedding_dimensions,
                embedding_model=config.embedding_model_name,
                schema_version=config.vector_schema_version,
                mode=self.mode,
                detail=str(exc),
            )
