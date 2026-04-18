import unittest

from ingestion.ingest_service import add_texts_in_batches


class FakeChroma:
    def __init__(self):
        self.calls = []

    def add_texts(self, texts, metadatas=None):
        self.calls.append((list(texts), list(metadatas or [])))


class TestBatching(unittest.TestCase):
    def test_add_texts_in_batches_should_split_to_ten(self):
        chroma = FakeChroma()
        texts = [f"chunk-{i}" for i in range(23)]
        metadatas = [{"idx": i} for i in range(23)]

        add_texts_in_batches(chroma, texts, metadatas, batch_size=10)

        print("batch_calls=", [len(texts) for texts, _ in chroma.calls])
        self.assertEqual([len(texts) for texts, _ in chroma.calls], [10, 10, 3])
        self.assertEqual(chroma.calls[0][0][0], "chunk-0")
        self.assertEqual(chroma.calls[-1][0][-1], "chunk-22")
        self.assertEqual(chroma.calls[1][1][0]["idx"], 10)

    def test_add_texts_in_batches_should_reject_mismatched_lengths(self):
        chroma = FakeChroma()
        with self.assertRaises(ValueError):
            add_texts_in_batches(chroma, ["a"], [])


if __name__ == "__main__":
    unittest.main()