"""可切换的向量数据库适配器。"""

from infra.vector_stores.base import VectorStoreAdapter, VectorStoreHealth
from infra.vector_stores.factory import create_vector_store

__all__ = ["VectorStoreAdapter", "VectorStoreHealth", "create_vector_store"]
