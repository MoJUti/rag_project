"""解析文档到检索索引之间的稳定切块协议。"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ingestion.document_model import SourceLocation


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    strategy: str
    chunk_type: ChunkType
    text: str
    token_count: int = Field(ge=0)
    section_path: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    source_locations: list[SourceLocation] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def stable_chunk_id(document_id: str, strategy: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"chunk_{hashlib.sha256(f'{document_id}|{strategy}|{ordinal}|{digest}'.encode()).hexdigest()[:24]}"


__all__ = ["ChunkType", "KnowledgeChunk", "stable_chunk_id"]
