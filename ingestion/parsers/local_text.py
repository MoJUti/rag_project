"""TXT 与 Markdown 的无损本地解析器。"""

from __future__ import annotations

import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path

from ingestion.document_model import (
    DocumentElement,
    DocumentFormat,
    ElementType,
    ParseQuality,
    ParseStatus,
    ParsedDocument,
    ParserProvenance,
    QualityMetric,
    QualityStatus,
    SourceDocument,
    SourceLocation,
    config_fingerprint,
    sha256_file,
    stable_document_id,
    stable_element_id,
)
from ingestion.parsers.base import ParseRequest, ParserCapabilities, ProbeResult
from ingestion.parsers.routing import detect_document_format


class LocalTextParser:
    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            parser_name="local_text",
            supported_formats=frozenset({DocumentFormat.TXT, DocumentFormat.MARKDOWN}),
            remote=False,
            structured_output=True,
        )

    def probe(self, source_path: Path) -> ProbeResult:
        try:
            detected = detect_document_format(source_path)
        except ValueError as exc:
            return ProbeResult(supported=False, confidence=0.0, reason=str(exc))
        supported = detected in self.capabilities.supported_formats
        return ProbeResult(
            supported=supported,
            confidence=1.0 if supported else 0.0,
            detected_format=detected,
            reason="扩展名与本地文本解析器匹配" if supported else "格式不受支持",
        )

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        if request.document_format not in self.capabilities.supported_formats:
            raise ValueError(f"LocalTextParser 不支持 {request.document_format.value}")
        path = request.source_path
        started_at = datetime.now(timezone.utc)
        text = _read_text(path)
        source_hash = sha256_file(path)
        document_id = request.document_id or stable_document_id(source_hash)
        elements = (
            _parse_markdown(text, document_id)
            if request.document_format == DocumentFormat.MARKDOWN
            else _parse_plain_text(text, document_id)
        )
        nonempty = bool(text.strip())
        completed_at = datetime.now(timezone.utc)
        metric = QualityMetric(
            name="nonempty_content",
            score=1.0 if nonempty else 0.0,
            threshold=1.0,
            passed=nonempty,
        )
        return ParsedDocument(
            source=SourceDocument(
                document_id=document_id,
                filename=path.name,
                document_format=request.document_format,
                media_type=mimetypes.guess_type(path.name)[0] or "text/plain",
                size_bytes=path.stat().st_size,
                sha256=source_hash,
                source_uri=str(path.resolve()),
            ),
            status=ParseStatus.SUCCESS if elements else ParseStatus.FAILED,
            parser=ParserProvenance(
                parser_name="local_text",
                parser_version="1.0",
                config_fingerprint=config_fingerprint({"encoding": "utf-8-sig-fallback"}),
                started_at=started_at,
                completed_at=completed_at,
            ),
            elements=elements,
            root_element_ids=[element.element_id for element in elements],
            quality=ParseQuality(
                status=QualityStatus.PASSED if nonempty else QualityStatus.FAILED,
                overall_score=metric.score,
                metrics=[metric],
            ),
        )


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_plain_text(text: str, document_id: str) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for order, match in enumerate(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S)):
        block = match.group(0).strip()
        start_line = text.count("\n", 0, match.start()) + 1
        end_line = start_line + block.count("\n")
        elements.append(_element(document_id, order, ElementType.PARAGRAPH, block, start_line, end_line))
    return elements


def _parse_markdown(text: str, document_id: str) -> list[DocumentElement]:
    lines = text.splitlines()
    elements: list[DocumentElement] = []
    index = 0
    order = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        start = index + 1
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            element = _element(document_id, order, ElementType.HEADING, heading.group(2).strip(), start, start)
            element.level = level
            elements.append(element)
            index += 1
        elif line.lstrip().startswith("```"):
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].lstrip().startswith("```"):
                    index += 1
                    break
                index += 1
            elements.append(_element(document_id, order, ElementType.CODE, "\n".join(block), start, index))
        else:
            block = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                if re.match(r"^(#{1,6})\s+", lines[index]) or lines[index].lstrip().startswith("```"):
                    break
                block.append(lines[index])
                index += 1
            stripped = line.lstrip()
            kind = ElementType.QUOTE if stripped.startswith(">") else (
                ElementType.LIST if re.match(r"^(?:[-*+] |\d+[.)] )", stripped) else ElementType.PARAGRAPH
            )
            elements.append(_element(document_id, order, kind, "\n".join(block).strip(), start, start + len(block) - 1))
        order += 1
    return elements


def _element(
    document_id: str,
    order: int,
    kind: ElementType,
    text: str,
    line_start: int,
    line_end: int,
) -> DocumentElement:
    locator = f"lines:{line_start}-{line_end}"
    return DocumentElement(
        element_id=stable_element_id(document_id, kind, order, locator),
        element_type=kind,
        order=order,
        text=text,
        source_location=SourceLocation(
            line_start=line_start,
            line_end=line_end,
            parser_locator=locator,
        ),
    )


__all__ = ["LocalTextParser"]
