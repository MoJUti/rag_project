"""把文档级质量差异收敛为可定位的最小复核目标。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

from ingestion.document_model import DocumentElement, DocumentFormat, ElementType, ParsedDocument
from ingestion.parsers.native_extractors import (
    NativeDocumentSnapshot, NativeUnit, missing_numeric_tokens, numeric_tokens,
)
from ingestion.parsers.review_models import AnomalyType, ReviewTarget, ReviewUnitType


_COMPACT_RE = re.compile(r"[^0-9A-Za-z\u3400-\u9fff]+")


class DocumentAnomalyDetector:
    """保守检测：没有明确证据的单元不会进入多模态复核。"""

    def __init__(self, text_coverage_threshold: float = 0.92, min_reference_chars: int = 40) -> None:
        self.text_coverage_threshold = text_coverage_threshold
        self.min_reference_chars = min_reference_chars

    def detect(self, document: ParsedDocument, native: NativeDocumentSnapshot) -> list[ReviewTarget]:
        targets: list[ReviewTarget] = []
        parsed_by_locator = _parsed_units(document)
        global_missing_numbers = missing_numeric_tokens(native.numeric_tokens, _element_text(document.elements))

        for unit_index, unit in enumerate(native.units):
            parsed_elements = parsed_by_locator.get(unit.locator, [])
            if (
                not parsed_elements
                and document.source.document_format == DocumentFormat.XLSX
                and unit.locator.startswith("sheet:")
            ):
                parsed_elements = parsed_by_locator.get(f"page:{unit_index + 1}", [])
            parsed_text = _element_text(parsed_elements)
            expected = _compact(unit.text)
            actual = _compact(parsed_text)
            anomalies: list[AnomalyType] = []
            evidence: dict[str, object] = {}

            if expected and not actual:
                anomalies.append(AnomalyType.EMPTY_UNIT)
            elif len(expected) >= self.min_reference_chars:
                coverage, fragments = _shingle_coverage(expected, actual)
                low_coverage = coverage < self.text_coverage_threshold
                if document.source.document_format == DocumentFormat.PDF:
                    low_coverage = coverage < 0.55 and len(actual) < len(expected) * 0.45
                if low_coverage:
                    anomalies.extend([AnomalyType.LOW_TEXT_COVERAGE, AnomalyType.MISSING_TEXT])
                    evidence.update({"text_coverage": coverage, "missing_fragments": fragments[:12]})

            missing_numbers = sorted(
                missing_numeric_tokens(unit.numeric_tokens, parsed_text) & global_missing_numbers
            )
            missing_numbers = [
                token for token in missing_numbers if not _covered_by_longer_local_number(token, parsed_text)
            ]
            numeric_contexts = _missing_numeric_contexts(unit.text, parsed_text, missing_numbers)
            if missing_numbers:
                evidence["missing_numeric_tokens"] = missing_numbers[:100]
                anomalies.append(AnomalyType.MISSING_NUMERIC_FACT)
                if numeric_contexts:
                    evidence["missing_numeric_contexts"] = numeric_contexts[:12]
                else:
                    evidence["deterministic_numeric_repair"] = True

            parsed_table_count = sum(element.element_type == ElementType.TABLE for element in parsed_elements)
            parsed_formula_count = sum(
                (1 if element.element_type == ElementType.FORMULA else 0)
                + (sum(cell.formula is not None for cell in element.table.cells) if element.table else 0)
                for element in parsed_elements
            )
            parsed_chart_count = sum(element.metadata.get("mineru_type") == "chart" for element in parsed_elements)
            if unit.table_count > parsed_table_count:
                anomalies.append(AnomalyType.TABLE_STRUCTURE)
                evidence.update({"expected_tables": unit.table_count, "parsed_tables": parsed_table_count})
            if unit.formula_count > parsed_formula_count:
                anomalies.append(AnomalyType.FORMULA_STRUCTURE)
                evidence.update({"expected_formulas": unit.formula_count, "parsed_formulas": parsed_formula_count})
            if unit.chart_count > parsed_chart_count:
                anomalies.append(AnomalyType.VISUAL_SEMANTICS)
                evidence.update({"expected_charts": unit.chart_count, "parsed_charts": parsed_chart_count})

            if anomalies:
                targets.append(_unit_target(document, unit, parsed_elements, anomalies, evidence))

        # MinerU 已识别为 chart，但没有给出任何可检索语义时，精确复核图表区域。
        for element in document.elements:
            if element.metadata.get("mineru_type") != "chart" or element.text.strip():
                continue
            targets.append(
                _element_target(
                    document,
                    element,
                    [AnomalyType.VISUAL_SEMANTICS],
                    {"reason": "chart element has no searchable text"},
                )
            )

        # 独立图片只有在解析结果为空或极短时才复核，避免装饰图片被无条件送审。
        if document.source.document_format == DocumentFormat.IMAGE:
            visible = _compact(_element_text(document.elements))
            if len(visible) < 8:
                targets.append(
                    _document_image_target(document, len(visible))
                )
        return _deduplicate(targets)


def _unit_target(
    document: ParsedDocument,
    unit: NativeUnit,
    elements: list[DocumentElement],
    anomalies: list[AnomalyType],
    evidence: dict[str, object],
) -> ReviewTarget:
    prefix, _, value = unit.locator.partition(":")
    page = int(value) if prefix == "page" and value.isdigit() else None
    slide = int(value) if prefix == "slide" and value.isdigit() else None
    sheet = value if prefix == "sheet" else None
    unit_type = {
        "page": ReviewUnitType.PAGE,
        "slide": ReviewUnitType.SLIDE,
        "sheet": ReviewUnitType.SHEET,
        "section": ReviewUnitType.SECTION,
    }.get(prefix, ReviewUnitType.DOCUMENT)
    requires_vision = document.source.document_format == DocumentFormat.IMAGE
    if document.source.document_format == DocumentFormat.PDF:
        requires_vision = (
            AnomalyType.MISSING_NUMERIC_FACT in anomalies
            and bool(evidence.get("missing_numeric_contexts"))
        )
    if document.source.document_format == DocumentFormat.PPTX:
        requires_vision = AnomalyType.VISUAL_SEMANTICS in anomalies
    return ReviewTarget(
        target_id=_target_id(document.source.document_id, unit.locator, anomalies),
        document_id=document.source.document_id,
        document_format=document.source.document_format,
        unit_type=unit_type,
        locator=unit.locator,
        page_number=page,
        slide_number=slide,
        sheet_name=sheet,
        element_ids=[element.element_id for element in elements],
        asset_ids=[asset_id for element in elements for asset_id in element.asset_ids],
        anomalies=list(dict.fromkeys(anomalies)),
        requires_vision=requires_vision,
        evidence=evidence,
    )


def _element_target(
    document: ParsedDocument,
    element: DocumentElement,
    anomalies: list[AnomalyType],
    evidence: dict[str, object],
) -> ReviewTarget:
    location = element.source_location
    page = location.page_number if location else None
    slide = location.slide_number if location else None
    locator = f"element:{element.element_id}"
    unit_type = ReviewUnitType.REGION if location and location.bbox else (
        ReviewUnitType.SLIDE if slide else ReviewUnitType.PAGE if page else ReviewUnitType.DOCUMENT
    )
    return ReviewTarget(
        target_id=_target_id(document.source.document_id, locator, anomalies),
        document_id=document.source.document_id,
        document_format=document.source.document_format,
        unit_type=unit_type,
        locator=locator,
        page_number=page,
        slide_number=slide,
        bbox=location.bbox if location else None,
        element_ids=[element.element_id],
        asset_ids=element.asset_ids,
        anomalies=anomalies,
        requires_vision=True,
        evidence=evidence,
    )


def _document_image_target(document: ParsedDocument, parsed_chars: int) -> ReviewTarget:
    anomalies = [AnomalyType.EMPTY_UNIT if parsed_chars == 0 else AnomalyType.LOW_TEXT_COVERAGE]
    return ReviewTarget(
        target_id=_target_id(document.source.document_id, "image:1", anomalies),
        document_id=document.source.document_id,
        document_format=document.source.document_format,
        unit_type=ReviewUnitType.IMAGE,
        locator="image:1",
        anomalies=anomalies,
        requires_vision=True,
        evidence={"parsed_visible_chars": parsed_chars},
    )


def _parsed_units(document: ParsedDocument) -> dict[str, list[DocumentElement]]:
    result: dict[str, list[DocumentElement]] = defaultdict(list)
    for element in document.elements:
        result["document"].append(element)
        location = element.source_location
        if location and location.page_number:
            result[f"page:{location.page_number}"].append(element)
        if location and location.slide_number:
            result[f"slide:{location.slide_number}"].append(element)
        if location and location.sheet_name:
            result[f"sheet:{location.sheet_name}"].append(element)
    return result


def _element_text(elements: list[DocumentElement]) -> str:
    return "\n".join(
        [element.text for element in elements]
        + [cell.text for element in elements if element.table for cell in element.table.cells]
    )


def _has_unresolved_visual(elements: list[DocumentElement]) -> bool:
    return any(
        element.element_type == ElementType.FIGURE
        and (element.metadata.get("mineru_type") == "chart" or element.asset_ids)
        and not element.text.strip()
        for element in elements
    )


def _compact(value: str) -> str:
    return _COMPACT_RE.sub("", value).lower()


def _shingle_coverage(expected: str, actual: str, width: int = 8) -> tuple[float, list[str]]:
    if not expected:
        return 1.0, []
    if len(expected) < width:
        return (1.0 if expected in actual else 0.0), ([] if expected in actual else [expected])
    # 覆盖率不依赖阅读顺序，避免双栏、表格和页眉重排造成误报。
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    matched = sum(min(count, actual_counts[char]) for char, count in expected_counts.items())
    coverage = matched / sum(expected_counts.values())
    shingles = [expected[index:index + width] for index in range(0, len(expected) - width + 1, width)]
    missing = [value for value in shingles if value not in actual]
    return coverage, missing


def _missing_numeric_contexts(source: str, parsed: str, missing: list[str]) -> list[dict[str, str]]:
    """只有数值两侧语义锚点都存在、唯独数值缺失时，才认为定位证据充分。"""
    if not missing:
        return []
    actual = _compact(parsed)
    missing_set = set(missing)
    result: list[dict[str, str]] = []
    number_pattern = re.compile(
        r"(?<![0-9A-Za-z_.])[-+]?\d+(?:(?:,|，)\s*\d{3})*(?:\.\d+)?%?(?![0-9A-Za-z_.])"
    )
    for match in number_pattern.finditer(source):
        tokens = numeric_tokens(match.group(0))
        matched_tokens = tokens & missing_set
        if not matched_tokens:
            continue
        left_source = _compact(source[max(0, match.start() - 40):match.start()])
        right_source = _compact(source[match.end():match.end() + 40])
        left_candidates = [left_source[-width:] for width in range(10, 5, -1) if len(left_source) >= width]
        right_candidates = [right_source[:width] for width in range(10, 5, -1) if len(right_source) >= width]
        left = next((value for value in left_candidates if value in actual), "")
        right = next((value for value in right_candidates if value in actual), "")
        if left and right:
            result.append({
                "token": sorted(matched_tokens)[0],
                "left_anchor": left,
                "right_anchor": right,
                "source_context": source[max(0, match.start() - 30):match.end() + 30].replace("\n", " "),
            })
    return result


def _covered_by_longer_local_number(token: str, parsed: str) -> bool:
    """过滤 PDF 文字层把长数字从逗号或换行处拆开的伪缺失。"""
    if len(token.replace(".", "").lstrip("-")) < 3:
        return False
    for candidate in numeric_tokens(parsed):
        plain_token = token.lstrip("-")
        plain_candidate = candidate.lstrip("-")
        if len(plain_candidate) < len(plain_token) + 2:
            continue
        if plain_candidate.startswith(plain_token) or plain_candidate.endswith(plain_token):
            return True
    return False


def _target_id(document_id: str, locator: str, anomalies: list[AnomalyType]) -> str:
    raw = "|".join([document_id, locator, *sorted(value.value for value in anomalies)])
    return f"review_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _deduplicate(targets: list[ReviewTarget]) -> list[ReviewTarget]:
    # 同一页整体异常与其中图表区域可以并存；完全相同 target_id 只保留一次。
    return list({target.target_id: target for target in targets}.values())


__all__ = ["DocumentAnomalyDetector"]
