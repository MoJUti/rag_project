import tempfile
import unittest
from pathlib import Path

import numpy as np

from retrieval.embedding_cache import EmbeddingCache


class TestEmbeddingCache(unittest.TestCase):
    def test_round_trip_and_text_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            with EmbeddingCache(path) as cache:
                cache.put_many(
                    [("chunk-1", "原始文本", [0.1, 0.2, 0.3], 4)],
                    model="test-model",
                    dimensions=3,
                )
                self.assertTrue(cache.has_valid("chunk-1", "原始文本", "test-model", 3))
                self.assertFalse(cache.has_valid("chunk-1", "修改文本", "test-model", 3))
                value = cache.get("chunk-1", "test-model", 3)
                self.assertIsNotNone(value)
                np.testing.assert_allclose(value.vector, np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
                self.assertEqual(4, value.token_count)

    def test_rejects_wrong_vector_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            with EmbeddingCache(Path(directory) / "cache.sqlite3") as cache:
                with self.assertRaises(ValueError):
                    cache.put_many(
                        [("chunk-1", "文本", [0.1, 0.2], 2)],
                        model="test-model",
                        dimensions=3,
                    )


if __name__ == "__main__":
    unittest.main()
