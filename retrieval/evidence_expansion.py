"""按原始切块顺序补充邻接证据，保留来源并限制新增上下文。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from ingestion.chunk_models import KnowledgeChunk
from ingestion.chunking import TokenCounter


class NeighborhoodExpander:
    """先保留所有命中块，再按排名与距离追加同文档、同章节的邻居。"""

    def __init__(
        self, chunks: list[KnowledgeChunk], radius: int = 1, extra_token_budget: int = 2000,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        if radius < 1 or extra_token_budget < 0:
            raise ValueError("证据扩展半径必须大于0，新增Token预算不能小于0")
        self.radius = radius
        self.extra_token_budget = extra_token_budget
        self.count_tokens = count_tokens or TokenCounter().count
        self.documents: dict[str, list[KnowledgeChunk]] = {}
        self.positions: dict[str, tuple[str, int]] = {}
        for chunk in chunks:
            if chunk.chunk_id in self.positions:
                raise ValueError(f"证据邻接表出现重复切块ID：{chunk.chunk_id}")
            siblings = self.documents.setdefault(chunk.document_id, [])
            self.positions[chunk.chunk_id] = (chunk.document_id, len(siblings))
            siblings.append(chunk)

    @classmethod
    def from_directory(cls, directory: Path, **kwargs: int) -> NeighborhoodExpander:
        paths = sorted(directory.rglob("chunks.jsonl")) if directory.is_dir() else []
        if not paths:
            raise ValueError(f"证据扩展找不到切块文件：{directory}")
        chunks: list[KnowledgeChunk] = []
        document_files: dict[str, Path] = {}
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunk = KnowledgeChunk.model_validate_json(line)
                previous = document_files.setdefault(chunk.document_id, path)
                if previous != path:
                    raise ValueError(f"同一文档出现多个切块文件，顺序不明确：{chunk.document_id}")
                chunks.append(chunk)
        if not chunks:
            raise ValueError(f"证据扩展切块文件为空：{directory}")
        return cls(chunks, **kwargs)

    def __call__(self, query: str, seeds: list[Document]) -> list[Document]:
        if not seeds:
            return []
        selected: dict[str, Document] = {}
        positions: list[tuple[Document, KnowledgeChunk, int]] = []
        for seed in seeds:
            key = str(seed.metadata.get("chunk_id", ""))
            if key not in self.positions:
                raise ValueError(f"命中切块不在证据邻接表中：{key or '缺少chunk_id'}")
            document_id, ordinal = self.positions[key]
            chunk = self.documents[document_id][ordinal]
            if seed.metadata.get("document_id") != document_id or seed.page_content != chunk.text:
                raise ValueError(f"证据邻接表与当前索引内容不一致：{key}")
            selected[key] = seed
            positions.append((seed, chunk, ordinal))

        remaining = self.extra_token_budget
        # 优先补近邻；原命中块始终保留，预算只用于追加内容。
        for distance in range(1, self.radius + 1):
            for seed, chunk, ordinal in positions:
                siblings = self.documents[chunk.document_id]
                for index in (ordinal - distance, ordinal + distance):
                    if not 0 <= index < len(siblings):
                        continue
                    neighbor = siblings[index]
                    if neighbor.chunk_id in selected or neighbor.section_path != chunk.section_path:
                        continue
                    segment = siblings[min(index, ordinal):max(index, ordinal) + 1]
                    if any(item.section_path != chunk.section_path for item in segment):
                        continue
                    if chunk.chunk_type.value == neighbor.chunk_type.value == "table":
                        if not set(chunk.element_ids) & set(neighbor.element_ids):
                            continue
                    tokens = self.count_tokens(neighbor.text)
                    if tokens > remaining:
                        continue
                    source_fields = ("source", "source_uri", "dataset_id", "embedding_model",
                                     "embedding_dimensions", "vector_schema_version")
                    metadata = {key: seed.metadata[key] for key in source_fields if key in seed.metadata}
                    metadata.update({"chunk_id": neighbor.chunk_id, "document_id": neighbor.document_id,
                                     "chunk_type": neighbor.chunk_type.value, "token_count": tokens,
                                     "strategy": neighbor.strategy,
                                     "section_path_json": json.dumps(neighbor.section_path, ensure_ascii=False),
                                     "source_locations_json": json.dumps(
                                         [location.model_dump(mode="json") for location in neighbor.source_locations],
                                         ensure_ascii=False),
                                     "element_ids_json": json.dumps(neighbor.element_ids, ensure_ascii=False),
                                     "asset_ids_json": json.dumps(neighbor.asset_ids, ensure_ascii=False),
                                     "metadata_json": json.dumps(neighbor.metadata, ensure_ascii=False),
                                     "evidence_role": "neighbor", "expanded_from_chunk_id": chunk.chunk_id,
                                     "chunk_order": index})
                    selected[neighbor.chunk_id] = Document(page_content=neighbor.text, metadata=metadata)
                    remaining -= tokens
        return list(selected.values())

    def validate_index(self, documents: list[Document]) -> None:
        """启动时校验完整索引，避免把已删除的旧邻接内容重新加入答案。"""
        indexed = {str(doc.metadata.get("chunk_id", "")): doc for doc in documents}
        if len(indexed) != len(documents) or set(indexed) != set(self.positions):
            raise ValueError("证据邻接表与当前索引的切块集合不一致，请同步切块目录并重启")
        for key, (document_id, ordinal) in self.positions.items():
            chunk = self.documents[document_id][ordinal]
            doc = indexed[key]
            if doc.metadata.get("document_id") != document_id or doc.page_content != chunk.text:
                raise ValueError(f"证据邻接表与当前索引内容不一致：{key}")
