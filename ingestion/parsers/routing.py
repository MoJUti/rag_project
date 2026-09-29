"""根据真实文件类型生成可执行的解析计划。"""

from __future__ import annotations

import mimetypes
from collections import defaultdict
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ingestion.document_model import DocumentFormat, sha256_file, stable_document_id


class ParserRoute(str, Enum):
    LOCAL_TEXT = "local_text"
    MINERU_DOCUMENT = "mineru_document"
    MINERU_HTML = "mineru_html"


EXTENSION_FORMATS: dict[str, DocumentFormat] = {
    ".txt": DocumentFormat.TXT,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".html": DocumentFormat.HTML,
    ".htm": DocumentFormat.HTML,
    ".pdf": DocumentFormat.PDF,
    ".png": DocumentFormat.IMAGE,
    ".jpg": DocumentFormat.IMAGE,
    ".jpeg": DocumentFormat.IMAGE,
    ".jp2": DocumentFormat.IMAGE,
    ".webp": DocumentFormat.IMAGE,
    ".gif": DocumentFormat.IMAGE,
    ".bmp": DocumentFormat.IMAGE,
    ".doc": DocumentFormat.DOC,
    ".docx": DocumentFormat.DOCX,
    ".ppt": DocumentFormat.PPT,
    ".pptx": DocumentFormat.PPTX,
    ".xls": DocumentFormat.XLS,
    ".xlsx": DocumentFormat.XLSX,
}

FORMAT_ROUTES: dict[DocumentFormat, ParserRoute] = {
    DocumentFormat.TXT: ParserRoute.LOCAL_TEXT,
    DocumentFormat.MARKDOWN: ParserRoute.LOCAL_TEXT,
    DocumentFormat.HTML: ParserRoute.MINERU_HTML,
    DocumentFormat.PDF: ParserRoute.MINERU_DOCUMENT,
    DocumentFormat.IMAGE: ParserRoute.MINERU_DOCUMENT,
    DocumentFormat.DOC: ParserRoute.MINERU_DOCUMENT,
    DocumentFormat.DOCX: ParserRoute.MINERU_DOCUMENT,
    DocumentFormat.PPT: ParserRoute.MINERU_DOCUMENT,
    DocumentFormat.PPTX: ParserRoute.MINERU_DOCUMENT,
    DocumentFormat.XLS: ParserRoute.MINERU_DOCUMENT,
    DocumentFormat.XLSX: ParserRoute.MINERU_DOCUMENT,
}


class PlannedDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    source_path: Path
    document_id: str
    document_format: DocumentFormat
    media_type: str
    route: ParserRoute
    model_version: str | None = None


class ParseBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: ParserRoute
    model_version: str | None = None
    documents: list[PlannedDocument] = Field(min_length=1, max_length=50)

    @property
    def is_remote(self) -> bool:
        return self.route != ParserRoute.LOCAL_TEXT


def detect_document_format(path: str | Path) -> DocumentFormat:
    extension = Path(path).suffix.lower()
    try:
        return EXTENSION_FORMATS[extension]
    except KeyError as exc:
        supported = ", ".join(sorted(EXTENSION_FORMATS))
        raise ValueError(f"不支持的文件格式 {extension!r}；当前支持: {supported}") from exc


def route_for_format(document_format: DocumentFormat) -> ParserRoute:
    return FORMAT_ROUTES[document_format]


def model_for_route(
    route: ParserRoute,
    default_model: str = "vlm",
    html_model: str = "MinerU-HTML",
) -> str | None:
    if route == ParserRoute.MINERU_DOCUMENT:
        return default_model
    if route == ParserRoute.MINERU_HTML:
        return html_model
    return None


def plan_document(
    path: str | Path,
    default_model: str = "vlm",
    html_model: str = "MinerU-HTML",
) -> PlannedDocument:
    source_path = Path(path)
    document_format = detect_document_format(source_path)
    route = route_for_format(document_format)
    file_hash = sha256_file(source_path)
    media_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    return PlannedDocument(
        source_path=source_path,
        document_id=stable_document_id(file_hash),
        document_format=document_format,
        media_type=media_type,
        route=route,
        model_version=model_for_route(route, default_model, html_model),
    )


def build_parse_batches(
    documents: list[PlannedDocument],
    max_batch_size: int = 50,
) -> list[ParseBatch]:
    if not 1 <= max_batch_size <= 50:
        raise ValueError("MinerU 单批大小必须在 1 到 50 之间")

    groups: dict[tuple[ParserRoute, str | None], list[PlannedDocument]] = defaultdict(list)
    for document in documents:
        groups[(document.route, document.model_version)].append(document)

    batches: list[ParseBatch] = []
    for (route, model_version), group in groups.items():
        for start in range(0, len(group), max_batch_size):
            batches.append(
                ParseBatch(
                    route=route,
                    model_version=model_version,
                    documents=group[start : start + max_batch_size],
                )
            )
    return batches


__all__ = [
    "EXTENSION_FORMATS",
    "FORMAT_ROUTES",
    "ParseBatch",
    "ParserRoute",
    "PlannedDocument",
    "build_parse_batches",
    "detect_document_format",
    "model_for_route",
    "plan_document",
    "route_for_format",
]
