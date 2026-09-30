import unittest

from retrieval.compatible_embeddings import (
    CompatibleEmbeddingClient,
    CompatibleEmbeddings,
)


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, vectors):
        self.vectors = vectors

    def json(self):
        return {
            "data": [
                {"index": index, "embedding": vector}
                for index, vector in enumerate(self.vectors)
            ],
            "usage": {"total_tokens": 12},
        }


class FakeHttpClient:
    def __init__(self):
        self.calls = []

    def post(self, endpoint, headers, json):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": json})
        return FakeResponse([[float(index)] * json["dimensions"] for index, _ in enumerate(json["input"])])


class TestCompatibleEmbeddings(unittest.TestCase):
    def test_request_explicitly_sends_model_and_dimensions(self):
        http = FakeHttpClient()
        client = CompatibleEmbeddingClient(
            base_url="https://example.test/v1",
            api_key="secret",
            model="text-embedding-v4",
            dimensions=2048,
            http_client=http,
        )
        vectors, tokens = client.embed(["甲", "乙"])

        self.assertEqual(2, len(vectors))
        self.assertEqual(2048, len(vectors[0]))
        self.assertEqual(12, tokens)
        self.assertEqual("text-embedding-v4", http.calls[0]["json"]["model"])
        self.assertEqual(2048, http.calls[0]["json"]["dimensions"])

    def test_langchain_wrapper_splits_batches_at_configured_size(self):
        http = FakeHttpClient()
        client = CompatibleEmbeddingClient(
            base_url="https://example.test/v1",
            api_key="secret",
            model="text-embedding-v4",
            dimensions=4,
            http_client=http,
        )
        embeddings = CompatibleEmbeddings(
            base_url="https://unused.test/v1",
            api_key="unused",
            model="text-embedding-v4",
            dimensions=4,
            batch_size=2,
            client=client,
        )

        vectors = embeddings.embed_documents(["1", "2", "3", "4", "5"])
        self.assertEqual([2, 2, 1], [len(call["json"]["input"]) for call in http.calls])
        self.assertEqual(5, len(vectors))


if __name__ == "__main__":
    unittest.main()
