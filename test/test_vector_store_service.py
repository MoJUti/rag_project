import gc
import tempfile
import unittest
from unittest.mock import patch

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core import config
from infra.vector_store import VectorStoreService
from infra.vector_stores.base import VectorStoreHealth
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from infra.vector_stores.factory import create_vector_store
from infra.vector_stores.qdrant import QdrantVectorStoreAdapter, normalize_point_id
from qdrant_client import QdrantClient


class FakeEmbeddings(Embeddings):
    def __init__(self, dimensions=3):
        self.dimensions = dimensions

    def embed_documents(self, texts):
        return [[float(index + 1)] * self.dimensions for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0] * self.dimensions


class FakeAdapter:
    backend_name = "fake"

    def __init__(self):
        self.added = []

    def add_texts(self, texts, metadatas=None, ids=None):
        self.added.append((texts, metadatas, ids))
        return ids or ["generated"]

    def similarity_search(self, query, k, source_filter=""):
        return [Document(page_content=query, metadata={"filter": source_filter})]

    def get_all_documents(self):
        return [Document(page_content="全部")]

    def count(self):
        return 1

    def health_check(self):
        return VectorStoreHealth("fake", "ok", "test", 1, 3, "model", "1", "memory")


class TestVectorStoreService(unittest.TestCase):
    def test_service_delegates_without_requiring_api_key(self):
        adapter = FakeAdapter()
        service = VectorStoreService(adapter=adapter)

        self.assertEqual("fake", service.backend_name)
        self.assertEqual(["id-1"], service.add_texts(["内容"], [{"source": "a"}], ["id-1"]))
        stored_metadata = adapter.added[0][1][0]
        self.assertEqual(config.embedding_model_name, stored_metadata["embedding_model"])
        self.assertEqual(config.embedding_dimensions, stored_metadata["embedding_dimensions"])
        self.assertEqual("问题", service.get_vector_docs("问题", 5)[0].page_content)
        self.assertEqual(1, service.count())
        self.assertEqual("ok", service.health_check()["status"])

    def test_factory_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "chroma 或 qdrant"):
            create_vector_store(FakeEmbeddings(), "unknown")

    def test_chroma_local_round_trip_and_dimension_guard(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            collection = "adapter_test"
            with (
                patch.object(config, "persist_directory", directory),
                patch.object(config, "collection_name", collection),
                patch.object(config, "chroma_mode", "local"),
                patch.object(config, "embedding_dimensions", 3),
            ):
                adapter = ChromaVectorStoreAdapter(FakeEmbeddings(3))
                adapter.add_texts(["甲", "乙"], [{"source": "A"}, {"source": "B"}], ["1", "2"])
                self.assertEqual(2, adapter.count())
                self.assertEqual(1, len(adapter.similarity_search("甲", 2, "A")))
                self.assertEqual("ok", adapter.health_check().status)

            with (
                patch.object(config, "persist_directory", directory),
                patch.object(config, "collection_name", collection),
                patch.object(config, "chroma_mode", "local"),
                patch.object(config, "embedding_dimensions", 4),
            ):
                with self.assertRaisesRegex(ValueError, "不同维度"):
                    mismatched = ChromaVectorStoreAdapter(FakeEmbeddings(4))
            del adapter
            if "mismatched" in locals():
                del mismatched
            gc.collect()
            SharedSystemClient.clear_system_cache()

    def test_qdrant_adapter_round_trip_with_local_client(self):
        local_client = QdrantClient(":memory:")
        with (
            patch("infra.vector_stores.qdrant.QdrantClient", return_value=local_client),
            patch.object(config, "collection_name", "adapter_test"),
            patch.object(config, "embedding_dimensions", 3),
        ):
            adapter = QdrantVectorStoreAdapter(FakeEmbeddings(3))
            adapter.add_texts(["甲", "乙"], [{"source": "A"}, {"source": "B"}], ["doc-a", "doc-b"])
            self.assertEqual(2, adapter.count())
            self.assertEqual(1, len(adapter.similarity_search("甲", 2, "A")))
            self.assertEqual(2, len(adapter.get_all_documents()))

    def test_qdrant_id_mapping_is_stable(self):
        self.assertEqual(normalize_point_id("doc-1"), normalize_point_id("doc-1"))
        self.assertNotEqual(normalize_point_id("doc-1"), normalize_point_id("doc-2"))


if __name__ == "__main__":
    unittest.main()
