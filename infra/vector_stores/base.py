"""向量数据库统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class VectorStoreHealth:
    backend: str
    status: str
    collection: str
    vector_count: int
    vector_dimensions: int
    embedding_model: str
    schema_version: str
    mode: str
    detail: str = ""

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


class VectorStoreAdapter(ABC):
    """业务层只依赖本接口，不感知 Chroma/Qdrant 的 SDK 差异。"""

    backend_name: str

    @abstractmethod
    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int,
        source_filter: str = "",
    ) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def get_all_documents(self) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> VectorStoreHealth:
        raise NotImplementedError
