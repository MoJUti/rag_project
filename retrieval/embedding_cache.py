"""可恢复、可校验的 Embedding SQLite 缓存。"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedEmbedding:
    item_id: str
    text_sha256: str
    model: str
    dimensions: int
    vector: np.ndarray
    token_count: int | None


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                item_id TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                token_count INTEGER,
                created_at TEXT NOT NULL,
                PRIMARY KEY (item_id, model, dimensions)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def has_valid(self, item_id: str, text: str, model: str, dimensions: int) -> bool:
        row = self.connection.execute(
            "SELECT text_sha256, length(vector) FROM embeddings WHERE item_id=? AND model=? AND dimensions=?",
            (item_id, model, dimensions),
        ).fetchone()
        return bool(
            row
            and row[0] == text_sha256(text)
            and row[1] == dimensions * np.dtype(np.float32).itemsize
        )

    def put_many(
        self,
        rows: list[tuple[str, str, list[float], int | None]],
        model: str,
        dimensions: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        values = []
        for item_id, text, vector, token_count in rows:
            array = np.asarray(vector, dtype=np.float32)
            if array.ndim != 1 or len(array) != dimensions:
                raise ValueError(f"{item_id} 向量维度异常: {array.shape}，预期 {dimensions}")
            values.append((
                item_id,
                text_sha256(text),
                model,
                dimensions,
                array.tobytes(order="C"),
                token_count,
                now,
            ))
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO embeddings
                    (item_id, text_sha256, model, dimensions, vector, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id, model, dimensions) DO UPDATE SET
                    text_sha256=excluded.text_sha256,
                    vector=excluded.vector,
                    token_count=excluded.token_count,
                    created_at=excluded.created_at
                """,
                values,
            )

    def get(self, item_id: str, model: str, dimensions: int) -> CachedEmbedding | None:
        row = self.connection.execute(
            """
            SELECT text_sha256, vector, token_count
            FROM embeddings WHERE item_id=? AND model=? AND dimensions=?
            """,
            (item_id, model, dimensions),
        ).fetchone()
        if not row:
            return None
        vector = np.frombuffer(row[1], dtype=np.float32).copy()
        if len(vector) != dimensions:
            raise ValueError(f"缓存向量损坏: {item_id}")
        return CachedEmbedding(item_id, row[0], model, dimensions, vector, row[2])

    def valid_count(self, items: list[tuple[str, str]], model: str, dimensions: int) -> int:
        return sum(self.has_valid(item_id, text, model, dimensions) for item_id, text in items)


__all__ = ["CachedEmbedding", "EmbeddingCache", "text_sha256"]
