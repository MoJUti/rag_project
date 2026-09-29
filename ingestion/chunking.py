"""固定长度基线与结构感知切块器。"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ingestion.chunk_models import ChunkType, KnowledgeChunk, stable_chunk_id
from ingestion.document_model import DocumentElement, ElementType, ParsedDocument, SourceLocation, config_fingerprint


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_tokens: int = Field(default=600, ge=64, le=4096)
    max_tokens: int = Field(default=800, ge=96, le=8192)
    overlap_tokens: int = Field(default=80, ge=0, le=1024)
    table_rows_per_chunk: int = Field(default=20, ge=1, le=500)

    @model_validator(mode="after")
    def validate_ranges(self) -> "ChunkingConfig":
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens 不能大于 max_tokens")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens 必须小于 max_tokens")
        return self

    def fingerprint(self) -> str:
        return config_fingerprint(self.model_dump(mode="json"))


class TokenCounter:
    def __init__(self) -> None:
        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, len(text) // 2)

    def split(self, text: str, max_tokens: int) -> list[str]:
        """按真实 tokenizer 切开超长单元；无 tokenizer 时采用保守字符上限。"""
        if self._encoding is not None:
            token_ids = self._encoding.encode(text)
            return [self._encoding.decode(token_ids[index:index + max_tokens]) for index in range(0, len(token_ids), max_tokens)]
        max_chars = max(1, max_tokens * 2)
        return [text[index:index + max_chars] for index in range(0, len(text), max_chars)]


class FixedTokenChunker:
    strategy = "fixed_token_v1"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.counter = TokenCounter()

    def chunk(self, document: ParsedDocument) -> list[KnowledgeChunk]:
        pieces = [_element_text(element) for element in document.elements]
        text = "\n".join(value for value in pieces if value.strip())
        segments = _split_text(text, self.config.max_tokens, self.config.overlap_tokens, self.counter)
        return [
            _chunk(
                document, self.strategy, index, ChunkType.TEXT, segment, self.counter,
                metadata={"chunking_config": self.config.model_dump(), "config_hash": self.config.fingerprint()},
            )
            for index, segment in enumerate(segments)
        ]


class StructureAwareChunker:
    strategy = "structure_aware_v1"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.counter = TokenCounter()

    def chunk(self, document: ParsedDocument) -> list[KnowledgeChunk]:
        elements = _prefer_pdf_repair_pages(document.elements)
        chunks: list[KnowledgeChunk] = []
        headings: list[str] = []
        buffer: list[DocumentElement] = []

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            chunks.extend(self._text_chunks(document, buffer, headings, len(chunks)))
            buffer = []

        for element in sorted(elements, key=lambda value: value.order):
            if element.element_type in {ElementType.TITLE, ElementType.HEADING}:
                flush()
                level = element.level or 1
                headings[:] = headings[: max(0, level - 1)]
                headings.append(element.text.strip())
                continue
            if element.element_type == ElementType.TABLE and element.table:
                flush()
                chunks.extend(self._table_chunks(document, element, headings, len(chunks)))
                continue
            if element.element_type in {ElementType.FIGURE, ElementType.FORMULA}:
                flush()
                body = _element_text(element)
                prefix_tokens = self.counter.count(" > ".join(headings)) if headings else 0
                available = max(1, self.config.max_tokens - prefix_tokens - 8)
                segments = _split_text(body, available, self.config.overlap_tokens, self.counter)
                kind = ChunkType.FIGURE if element.element_type == ElementType.FIGURE else ChunkType.FORMULA
                for segment in segments:
                    chunks.append(_chunk(
                        document, self.strategy, len(chunks), kind,
                        _with_heading(headings, segment), self.counter,
                        headings, [element], {"atomic": len(segments) == 1},
                    ))
                continue
            if _element_text(element).strip():
                buffer.append(element)
                if self.counter.count(_with_heading(headings, "\n".join(_element_text(v) for v in buffer))) >= self.config.target_tokens:
                    flush()
        flush()
        version = {"chunking_config": self.config.model_dump(), "config_hash": self.config.fingerprint()}
        return [chunk.model_copy(update={"metadata": {**chunk.metadata, **version}}) for chunk in chunks]

    def _text_chunks(
        self, document: ParsedDocument, elements: list[DocumentElement], headings: list[str], start: int,
    ) -> list[KnowledgeChunk]:
        body = "\n".join(_element_text(value) for value in elements if _element_text(value).strip())
        prefix = " > ".join(headings)
        available = max(1, self.config.max_tokens - self.counter.count(prefix) - 8)
        segments = _split_text(body, available, self.config.overlap_tokens, self.counter)
        return [
            _chunk(
                document, self.strategy, start + index, ChunkType.TEXT,
                _with_heading(headings, segment), self.counter, headings, elements,
                {"is_native_repair": any(v.metadata.get("source") == "native_structure_repair" for v in elements)},
            )
            for index, segment in enumerate(segments)
        ]

    def _table_chunks(
        self, document: ParsedDocument, element: DocumentElement, headings: list[str], start: int,
    ) -> list[KnowledgeChunk]:
        table = element.table
        assert table is not None
        by_row: dict[int, list] = defaultdict(list)
        for cell in table.cells:
            by_row[cell.row].append(cell)
        header_rows = sorted({cell.row for cell in table.cells if cell.is_header}) or ([0] if by_row else [])
        headers = _serialize_rows(by_row, header_rows)
        descriptor = table.caption or (element.text if not table.cells else "")
        data_rows = [row for row in sorted(by_row) if row not in header_rows]
        if not data_rows:
            data_rows = sorted(by_row)
        groups: list[list[int]] = []
        current: list[int] = []
        for row in data_rows:
            candidate = current + [row]
            text = _with_heading(headings, "\n".join(filter(None, [descriptor, headers, _serialize_rows(by_row, candidate)])))
            if current and (len(current) >= self.config.table_rows_per_chunk or self.counter.count(text) > self.config.max_tokens):
                groups.append(current)
                current = [row]
            else:
                current = candidate
        if current:
            groups.append(current)
        if not groups:
            groups = [[]]

        result: list[KnowledgeChunk] = []
        for index, rows in enumerate(groups):
            body = "\n".join(filter(None, [descriptor, headers, _serialize_rows(by_row, rows)]))
            prefix_tokens = self.counter.count(" > ".join(headings)) if headings else 0
            available = max(1, self.config.max_tokens - prefix_tokens - 8)
            segments = _split_text(body, available, self.config.overlap_tokens, self.counter)
            for segment in segments:
                result.append(_chunk(
                    document, self.strategy, start + len(result), ChunkType.TABLE,
                    _with_heading(headings, segment), self.counter,
                    headings, [element], {
                        "row_start": min(rows) if rows else None,
                        "row_end": max(rows) if rows else None,
                        "table_fragment": len(segments) > 1,
                    },
                ))
        return result


def _serialize_rows(by_row: dict[int, list], rows: Iterable[int]) -> str:
    lines: list[str] = []
    for row in rows:
        cells = sorted(by_row.get(row, []), key=lambda value: value.column)
        values = []
        for cell in cells:
            value = cell.text
            if cell.formula:
                value = f"{value}（公式：{cell.formula}）" if value else f"公式：{cell.formula}"
            values.append(value)
        if values:
            lines.append(f"第{row + 1}行 | " + " | ".join(values))
    return "\n".join(lines)


def _prefer_pdf_repair_pages(elements: list[DocumentElement]) -> list[DocumentElement]:
    repaired_pages = {
        element.source_location.page_number
        for element in elements
        if element.metadata.get("layout_mode") == "native_full_text_fallback"
        and element.source_location and element.source_location.page_number
    }
    if not repaired_pages:
        return elements
    result: list[DocumentElement] = []
    for element in elements:
        location = element.source_location
        page = location.page_number if location else None
        if page not in repaired_pages:
            result.append(element)
            continue
        if element.metadata.get("layout_mode") == "native_full_text_fallback" or element.element_type == ElementType.FIGURE:
            result.append(element)
    return result


def _element_text(element: DocumentElement) -> str:
    if element.table:
        rows: dict[int, list] = defaultdict(list)
        for cell in element.table.cells:
            rows[cell.row].append(cell)
        return "\n".join(filter(None, [element.text, _serialize_rows(rows, sorted(rows))]))
    return element.text.strip()


def _with_heading(headings: list[str], body: str) -> str:
    prefix = " > ".join(value for value in headings if value)
    return "\n".join(value for value in (prefix, body.strip()) if value)


def _split_text(text: str, max_tokens: int, overlap_tokens: int, counter: TokenCounter) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if counter.count(text) <= max_tokens:
        return [text]
    units = [value.strip() for value in re.split(r"(?<=[。！？；\n])", text) if value.strip()]
    expanded: list[str] = []
    for unit in units:
        expanded.extend(counter.split(unit, max_tokens) if counter.count(unit) > max_tokens else [unit])
    units = expanded
    result: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and counter.count("".join(current) + unit) > max_tokens:
            result.append("".join(current).strip())
            overlap: list[str] = []
            for previous in reversed(current):
                if counter.count("".join(reversed(overlap)) + previous) > overlap_tokens:
                    break
                overlap.append(previous)
            current = list(reversed(overlap))
            if current and counter.count("".join(current) + unit) > max_tokens:
                current = []
        current.append(unit)
    if current:
        result.append("".join(current).strip())
    return [value for value in result if value]


def _chunk(
    document: ParsedDocument, strategy: str, ordinal: int, kind: ChunkType, text: str,
    counter: TokenCounter, headings: list[str] | None = None,
    elements: list[DocumentElement] | None = None, metadata: dict | None = None,
) -> KnowledgeChunk:
    elements = elements or []
    locations: list[SourceLocation] = []
    for element in elements:
        if element.source_location and element.source_location not in locations:
            locations.append(element.source_location)
    chunk_metadata = {key: value for key, value in (metadata or {}).items() if value is not None}
    return KnowledgeChunk(
        chunk_id=stable_chunk_id(document.source.document_id, strategy, ordinal, text),
        document_id=document.source.document_id,
        strategy=strategy,
        chunk_type=kind,
        text=text,
        token_count=counter.count(text),
        section_path=list(headings or []),
        element_ids=[value.element_id for value in elements],
        source_locations=locations,
        asset_ids=list(dict.fromkeys(asset for value in elements for asset in value.asset_ids)),
        metadata=chunk_metadata,
    )


__all__ = ["ChunkingConfig", "FixedTokenChunker", "StructureAwareChunker", "TokenCounter"]
