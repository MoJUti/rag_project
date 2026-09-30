"""Qdrant 独立服务适配器。"""

from __future__ import annotations

import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from core import config
from infra.vector_stores.base import VectorStoreAdapter, VectorStoreHealth


_ID_NAMESPACE = uuid.UUID("95a56916-e359-4b2b-9b84-cdfb95082c8e")


def normalize_point_id(value: str) -> str:
    """Qdrant 只接受整数或 UUID；字符串 ID 稳定映射为 UUID。"""
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(_ID_NAMESPACE, value))


class QdrantVectorStoreAdapter(VectorStoreAdapter):
    backend_name = "qdrant"

    def __init__(self, embedding: Embeddings) -> None:
        self.embedding = embedding
        self.collection_name = config.collection_name
        self.client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=config.embedding_dimensions,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=config.qdrant_hnsw_m,
                    ef_construct=config.qdrant_hnsw_ef_construction,
                    full_scan_threshold=config.qdrant_full_scan_threshold_kb,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=config.qdrant_indexing_threshold_kb,
                    default_segment_number=1,
                ),
            )
        info = self.client.get_collection(self.collection_name)
        vector_config = info.config.params.vectors
        actual = getattr(vector_config, "size", None)
        if actual != config.embedding_dimensions:
            raise ValueError(
                f"Qdrant 集合 {self.collection_name!r} 是 {actual} 维，"
                f"当前配置是 {config.embedding_dimensions} 维。请更换 VECTOR_COLLECTION。"
            )

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if metadatas is not None and len(texts) != len(metadatas):
            raise ValueError("texts 和 metadatas 数量必须一致")
        if ids is not None and len(texts) != len(ids):
            raise ValueError("texts 和 ids 数量必须一致")
        raw_ids = ids or [str(uuid.uuid4()) for _ in texts]
        point_ids = [normalize_point_id(value) for value in raw_ids]
        metadata_rows = metadatas or [{} for _ in texts]
        vectors = self.embedding.embed_documents(texts)
        points = [
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "page_content": text,
                    "metadata": metadata,
                    "document_id": raw_id,
                },
            )
            for point_id, raw_id, text, metadata, vector in zip(
                point_ids, raw_ids, texts, metadata_rows, vectors
            )
        ]
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        return raw_ids

    def similarity_search(
        self,
        query: str,
        k: int,
        source_filter: str = "",
    ) -> list[Document]:
        query_filter = None
        if source_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source_filter),
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedding.embed_query(query),
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            search_params=models.SearchParams(
                hnsw_ef=config.qdrant_hnsw_ef_search,
                exact=False,
            ),
        )
        return [_point_to_document(point) for point in response.points]

    def get_all_documents(self) -> list[Document]:
        documents: list[Document] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(_point_to_document(point) for point in points)
            if offset is None:
                break
        return documents

    def count(self) -> int:
        return self.client.count(self.collection_name, exact=True).count

    def health_check(self) -> VectorStoreHealth:
        try:
            info = self.client.get_collection(self.collection_name)
            raw_status = getattr(info.status, "value", str(info.status))
            status = "ok" if str(raw_status).lower() == "green" else str(raw_status)
            return VectorStoreHealth(
                backend=self.backend_name,
                status=status,
                collection=self.collection_name,
                vector_count=self.count(),
                vector_dimensions=config.embedding_dimensions,
                embedding_model=config.embedding_model_name,
                schema_version=config.vector_schema_version,
                mode="http",
            )
        except Exception as exc:
            return VectorStoreHealth(
                backend=self.backend_name,
                status="error",
                collection=self.collection_name,
                vector_count=0,
                vector_dimensions=config.embedding_dimensions,
                embedding_model=config.embedding_model_name,
                schema_version=config.vector_schema_version,
                mode="http",
                detail=str(exc),
            )


def _point_to_document(point) -> Document:
    payload = point.payload or {}
    return Document(
        page_content=payload.get("page_content", ""),
        metadata=payload.get("metadata") or {},
    )
