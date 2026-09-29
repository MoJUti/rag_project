"""解析层与后续切块层之间的稳定文档协议。"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0"
_ELEMENT_NAMESPACE = uuid.UUID("9d7e6e6c-56a9-4c79-9518-8b4bc89e8390")


class DocumentFormat(str, Enum):
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    IMAGE = "image"
    DOC = "doc"
    DOCX = "docx"
    PPT = "ppt"
    PPTX = "pptx"
    XLS = "xls"
    XLSX = "xlsx"


class ParseStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class QualityStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ElementType(str, Enum):
    DOCUMENT = "document"
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CODE = "code"
    QUOTE = "quote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_BREAK = "page_break"
    UNKNOWN = "unknown"


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class BoundingBox(BaseModel):
    """页面坐标；origin 说明坐标原点，避免解析器之间产生歧义。"""

    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float | None = None
    page_height: float | None = None
    origin: str = "top_left"

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox 的结束坐标不能小于起始坐标")
        return self


class SourceLocation(BaseModel):
    """元素在原始文件中的可追溯位置，编号统一从 1 开始。"""

    model_config = ConfigDict(extra="forbid")

    page_number: int | None = Field(default=None, ge=1)
    slide_number: int | None = Field(default=None, ge=1)
    sheet_name: str | None = None
    cell_range: str | None = None
    paragraph_index: int | None = Field(default=None, ge=0)
    table_index: int | None = Field(default=None, ge=0)
    row_index: int | None = Field(default=None, ge=0)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    bbox: BoundingBox | None = None
    parser_locator: str | None = None


class TableCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: int = Field(ge=0)
    column: int = Field(ge=0)
    text: str = ""
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    is_header: bool = False
    raw_value: str | int | float | bool | None = None
    formula: str | None = None


class TableData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)
    caption: str | None = None

    @model_validator(mode="after")
    def validate_cell_bounds(self) -> "TableData":
        for cell in self.cells:
            if cell.row + cell.row_span > self.row_count:
                raise ValueError("表格单元格超出 row_count")
            if cell.column + cell.column_span > self.column_count:
                raise ValueError("表格单元格超出 column_count")
        return self


class AssetReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    media_type: str
    relative_path: str
    sha256: str | None = None
    source_location: SourceLocation | None = None


class DocumentElement(BaseModel):
    """扁平存储的语义元素，通过 parent/children 表达层级。"""

    model_config = ConfigDict(extra="forbid")

    element_id: str
    element_type: ElementType
    order: int = Field(ge=0)
    text: str = ""
    level: int | None = Field(default=None, ge=1)
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    source_location: SourceLocation | None = None
    table: TableData | None = None
    asset_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "DocumentElement":
        if self.element_type == ElementType.TABLE and self.table is None:
            raise ValueError("table 元素必须提供 table 数据")
        if self.table is not None and self.element_type != ElementType.TABLE:
            raise ValueError("只有 table 元素可以携带 table 数据")
        return self


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    document_format: DocumentFormat
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str
    source_uri: str | None = None


class ParserProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser_name: str
    parser_version: str | None = None
    model_version: str | None = None
    config_fingerprint: str
    remote_task_id: str | None = None
    remote_batch_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ParseIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: IssueSeverity
    retryable: bool = False
    element_id: str | None = None
    source_location: SourceLocation | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class QualityMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ParseQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: QualityStatus = QualityStatus.NOT_EVALUATED
    overall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metrics: list[QualityMetric] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """所有解析器必须产生的统一输出。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    source: SourceDocument
    status: ParseStatus
    parser: ParserProvenance
    elements: list[DocumentElement] = Field(default_factory=list)
    root_element_ids: list[str] = Field(default_factory=list)
    assets: list[AssetReference] = Field(default_factory=list)
    issues: list[ParseIssue] = Field(default_factory=list)
    quality: ParseQuality = Field(default_factory=ParseQuality)
    raw_artifact_paths: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_references(self) -> "ParsedDocument":
        element_ids = [element.element_id for element in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("element_id 必须唯一")

        known_elements = set(element_ids)
        for root_id in self.root_element_ids:
            if root_id not in known_elements:
                raise ValueError(f"root_element_id 不存在: {root_id}")

        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset_id 必须唯一")
        known_assets = set(asset_ids)

        for element in self.elements:
            if element.parent_id and element.parent_id not in known_elements:
                raise ValueError(f"parent_id 不存在: {element.parent_id}")
            missing_children = set(element.child_ids) - known_elements
            if missing_children:
                raise ValueError(f"child_ids 不存在: {sorted(missing_children)}")
            missing_assets = set(element.asset_ids) - known_assets
            if missing_assets:
                raise ValueError(f"asset_ids 不存在: {sorted(missing_assets)}")

        if self.status == ParseStatus.SUCCESS and not self.elements:
            raise ValueError("成功解析的文档不能为空")
        return self


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def stable_document_id(file_sha256: str) -> str:
    return f"doc_{file_sha256.lower()}"


def stable_element_id(
    document_id: str,
    element_type: ElementType | str,
    order: int,
    locator: str = "",
) -> str:
    element_value = element_type.value if isinstance(element_type, ElementType) else element_type
    identity = f"{document_id}|{element_value}|{order}|{locator}"
    return f"el_{uuid.uuid5(_ELEMENT_NAMESPACE, identity).hex}"


def config_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "AssetReference",
    "BoundingBox",
    "DocumentElement",
    "DocumentFormat",
    "ElementType",
    "IssueSeverity",
    "ParseIssue",
    "ParseQuality",
    "ParseStatus",
    "ParsedDocument",
    "ParserProvenance",
    "QualityMetric",
    "QualityStatus",
    "SCHEMA_VERSION",
    "SourceDocument",
    "SourceLocation",
    "TableCell",
    "TableData",
    "config_fingerprint",
    "sha256_file",
    "stable_document_id",
    "stable_element_id",
]
