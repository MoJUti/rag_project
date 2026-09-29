"""解析器协议。具体实现可以是本地解析器，也可以是远程 API。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ingestion.document_model import DocumentFormat, ParsedDocument


class ParseOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "ch"
    enable_ocr: bool = False
    enable_table: bool = True
    enable_formula: bool = True
    page_ranges: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ParseRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    source_path: Path
    document_format: DocumentFormat
    document_id: str
    options: ParseOptions = Field(default_factory=ParseOptions)


class ParserCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser_name: str
    supported_formats: frozenset[DocumentFormat]
    remote: bool
    structured_output: bool
    supports_ocr: bool = False
    supports_tables: bool = False
    supports_formulas: bool = False


class ProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    detected_format: DocumentFormat | None = None
    reason: str = ""


@runtime_checkable
class DocumentParser(Protocol):
    @property
    def capabilities(self) -> ParserCapabilities: ...

    def probe(self, source_path: Path) -> ProbeResult: ...

    async def parse(self, request: ParseRequest) -> ParsedDocument: ...


__all__ = [
    "DocumentParser",
    "ParseOptions",
    "ParseRequest",
    "ParserCapabilities",
    "ProbeResult",
]
