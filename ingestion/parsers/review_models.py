"""解析异常定位与多模态复核的稳定协议。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ingestion.document_model import BoundingBox, DocumentFormat


class ReviewUnitType(str, Enum):
    PAGE = "page"
    REGION = "region"
    SLIDE = "slide"
    SHEET = "sheet"
    SECTION = "section"
    IMAGE = "image"
    DOCUMENT = "document"


class AnomalyType(str, Enum):
    EMPTY_UNIT = "empty_unit"
    LOW_TEXT_COVERAGE = "low_text_coverage"
    MISSING_TEXT = "missing_text"
    MISSING_NUMERIC_FACT = "missing_numeric_fact"
    TABLE_STRUCTURE = "table_structure"
    FORMULA_STRUCTURE = "formula_structure"
    VISUAL_SEMANTICS = "visual_semantics"
    READING_ORDER = "reading_order"


class ReviewTarget(BaseModel):
    """一个可以精确定位、可以独立复核的最小单元。"""

    model_config = ConfigDict(extra="forbid")

    target_id: str
    document_id: str
    document_format: DocumentFormat
    unit_type: ReviewUnitType
    locator: str
    page_number: int | None = Field(default=None, ge=1)
    slide_number: int | None = Field(default=None, ge=1)
    sheet_name: str | None = None
    cell_range: str | None = None
    bbox: BoundingBox | None = None
    element_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    anomalies: list[AnomalyType]
    severity: str = "warning"
    requires_vision: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    status: str
    resolved_anomalies: list[AnomalyType] = Field(default_factory=list)
    unresolved_anomalies: list[AnomalyType] = Field(default_factory=list)
    elements: list[dict[str, Any]] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)
    model: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    render_details: dict[str, Any] = Field(default_factory=dict)


__all__ = ["AnomalyType", "ReviewFinding", "ReviewTarget", "ReviewUnitType"]
