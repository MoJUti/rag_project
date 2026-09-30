"""业务层统一向量库服务。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core import config
from infra.vector_stores import VectorStoreAdapter, create_vector_store
from retrieval.compatible_embeddings import CompatibleEmbeddings


class _Retriever:
    def __init__(self, service: "VectorStoreService") -> None:
        self.service = service

    def invoke(self, query: str) -> list[Document]:
        return self.service.get_vector_docs(query, config.retrieval_top_k)


class VectorStoreService:
    def __init__(
        self,
        embedding: Embeddings | None = None,
        adapter: VectorStoreAdapter | None = None,
        backend: str | None = None,
    ) -> None:
        if adapter is not None:
            self.embedding = embedding
            self.adapter = adapter
            return
        self.embedding = embedding or CompatibleEmbeddings.from_env(
            base_url=config.embedding_base_url,
            model=config.embedding_model_name,
            dimensions=config.embedding_dimensions,
            batch_size=config.embedding_batch_size,
        )
        self.adapter = create_vector_store(self.embedding, backend)

    @property
    def backend_name(self) -> str:
        return self.adapter.backend_name

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        source_rows = metadatas or [{} for _ in texts]
        if len(source_rows) != len(texts):
            raise ValueError("texts 和 metadatas 数量必须一致")
        enriched = []
        for metadata in source_rows:
            row = dict(metadata)
            row.setdefault("vector_schema_version", config.vector_schema_version)
            row.setdefault("embedding_model", config.embedding_model_name)
            row.setdefault("embedding_dimensions", config.embedding_dimensions)
            enriched.append(row)
        return self.adapter.add_texts(texts, enriched, ids)

    def get_retriever(self) -> _Retriever:
        return _Retriever(self)

    def get_vector_docs(self, query: str, top_k: int) -> list[Document]:
        return self.adapter.similarity_search(
            query=query,
            k=top_k,
            source_filter=config.retrieval_source_filter,
        )

    def get_all_documents(self) -> list[Document]:
        return self.adapter.get_all_documents()

    def count(self) -> int:
        return self.adapter.count()

    def health_check(self) -> dict[str, str | int]:
        return self.adapter.health_check().to_dict()
