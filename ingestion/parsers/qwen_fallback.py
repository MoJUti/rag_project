"""按异常目标调用千问多模态，而不是按整份文档盲目补漏。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ingestion.document_model import (
    DocumentElement, DocumentFormat, ElementType, IssueSeverity, ParseIssue,
    ParseStatus, ParsedDocument, SourceLocation, stable_element_id,
)
from ingestion.parsers.native_extractors import NativeDocumentSnapshot, numeric_tokens
from ingestion.parsers.quality_gate import DocumentQualityGate
from ingestion.parsers.qwen_vision import QwenVisionClient
from ingestion.parsers.review_models import AnomalyType, ReviewFinding, ReviewTarget


class QwenDocumentReviewer:
    """只处理已定位且确实需要视觉判断的目标。"""

    def __init__(self, client: QwenVisionClient, max_targets: int = 8, dpi: int = 200) -> None:
        self.client = client
        self.max_targets = max_targets
        self.dpi = dpi

    def review(
        self,
        document: ParsedDocument,
        source_path: Path,
        native: NativeDocumentSnapshot,
        artifact_dir: Path,
        targets: list[ReviewTarget],
    ) -> tuple[ParsedDocument, list[ReviewFinding]]:
        elements = list(document.elements)
        findings: list[ReviewFinding] = []
        artifact_dir.mkdir(parents=True, exist_ok=True)
        selected_ids = {
            target.target_id for target in [value for value in targets if value.requires_vision][:self.max_targets]
        }

        for target in targets:
            if not target.requires_vision:
                findings.append(_unreviewed(target, "deterministic_repair_required"))
                continue
            if target.target_id not in selected_ids:
                findings.append(_unreviewed(target, "review_budget_exceeded"))
                continue
            image_path, render_details = _render_target(
                document, source_path, target, artifact_dir, self.dpi
            )
            if image_path is None:
                findings.append(_unreviewed(target, "render_unavailable"))
                continue
            result = self.client.extract([image_path], build_review_prompt(target))
            observed = result.content.get("observed_elements", [])
            observed = [item for item in observed if isinstance(item, dict)] if isinstance(observed, list) else []
            uncertain = result.content.get("uncertain_items", [])
            uncertain_items = [str(item) for item in uncertain] if isinstance(uncertain, list) else []
            resolved, unresolved = _assess(target, observed, uncertain_items)
            for item in observed:
                created = _review_element(document, target, item, len(elements), result.model)
                if created is not None:
                    elements.append(created)
            findings.append(ReviewFinding(
                target_id=target.target_id,
                status="resolved" if not unresolved else ("uncertain" if uncertain_items else "unresolved"),
                resolved_anomalies=resolved,
                unresolved_anomalies=unresolved,
                elements=observed,
                uncertain_items=uncertain_items,
                model=result.model,
                usage=result.usage,
                artifact_path=str(image_path.resolve()),
                render_details=render_details,
            ))

        unresolved_count = sum(bool(finding.unresolved_anomalies) for finding in findings)
        issue = ParseIssue(
            code="DOCUMENT_REVIEW_TARGETS",
            message=f"{len(targets)} 个精确异常目标中 {len(targets) - unresolved_count} 个已解决，{unresolved_count} 个仍需处理",
            severity=IssueSeverity.WARNING if unresolved_count else IssueSeverity.INFO,
            retryable=bool(unresolved_count),
            details={
                "targets": [target.model_dump(mode="json") for target in targets],
                "findings": [finding.model_dump(mode="json") for finding in findings],
                "enable_thinking": False,
            },
        )
        issues = [value for value in document.issues if value.code != "DOCUMENT_REVIEW_TARGETS"] + [issue]
        enriched = document.model_copy(update={
            "elements": elements,
            "root_element_ids": [element.element_id for element in elements],
            "issues": issues,
            "raw_artifact_paths": list(dict.fromkeys(
                document.raw_artifact_paths + [finding.artifact_path for finding in findings if finding.artifact_path]
            )),
        })
        checked = DocumentQualityGate().evaluate(enriched, native)
        if unresolved_count and checked.status == ParseStatus.SUCCESS:
            checked = checked.model_copy(update={"status": ParseStatus.PARTIAL})
        return checked, findings


def build_review_prompt(target: ReviewTarget) -> str:
    """异常类型决定核验重点，但不会把提取范围缩窄为数字。"""
    anomaly_names = "、".join(value.value for value in target.anomalies)
    evidence = json.dumps(target.evidence, ensure_ascii=False, sort_keys=True)
    return f"""你是文档解析质量复核器。当前只给你一个已经被程序定位的异常单元，不代表整份文档都有问题。
目标 ID：{target.target_id}
原文件格式：{target.document_format.value}
定位：{target.locator}
检测到的异常：{anomaly_names}
检测证据：{evidence}

请完整检查图中所有可检索语义，而不只是数字：
1. 标题、段落、列表、页眉页脚及其阅读顺序；
2. 表格的标题、表头、行列、合并关系和所有单元格；
3. 图表的标题、图例、坐标轴、系列、标签、数值、单位和各元素关系；
4. 公式、符号、上下标、脚注；
5. 图片中的文字及其与正文或图表的关系。

只依据图片逐字核验，不使用外部知识，不补猜模糊内容。无法确认的内容必须放进 uncertain_items。
只输出一个 JSON 对象，结构如下：
{{
  "target_id": "{target.target_id}",
  "observed_elements": [
    {{"type": "heading|paragraph|list|table|chart|formula|image|header|footer|other",
      "reading_order": 1, "verbatim_text": "可直接核验的完整原文",
      "table": {{"headers": [], "rows": []}},
      "chart": {{"title": "", "axes": [], "legend": [], "series": []}},
      "formula": "", "relations": [], "bbox": [0, 0, 0, 0], "confidence": 0.0}}
  ],
  "missing_or_wrong_from_parser": [], "uncertain_items": [], "overall_confidence": 0.0
}}
没有对应内容的字段使用空值或空数组。不要输出 Markdown 代码围栏或解释。"""


def attach_review_targets(document: ParsedDocument, targets: list[ReviewTarget]) -> ParsedDocument:
    """未配置视觉客户端时也保留精确目标，阻止异常文档伪装为 success。"""
    if not targets:
        return document
    issue = ParseIssue(
        code="DOCUMENT_REVIEW_TARGETS",
        message=f"发现 {len(targets)} 个精确异常目标，尚未复核",
        severity=IssueSeverity.WARNING,
        retryable=True,
        details={"targets": [target.model_dump(mode="json") for target in targets]},
    )
    issues = [value for value in document.issues if value.code != issue.code] + [issue]
    return document.model_copy(update={"issues": issues, "status": ParseStatus.PARTIAL})


def _render_target(
    document: ParsedDocument, source_path: Path, target: ReviewTarget, artifact_dir: Path, dpi: int
) -> tuple[Path | None, dict[str, Any]]:
    asset = _resolve_asset(document, target) if (
        target.unit_type.value == "region" or AnomalyType.VISUAL_SEMANTICS in target.anomalies
    ) else None
    if asset is not None:
        return asset, {"mode": "mineru_asset", "asset_ids": target.asset_ids}
    if target.document_format == DocumentFormat.IMAGE:
        return source_path, {"mode": "original_image"}
    if target.document_format != DocumentFormat.PDF or not target.page_number:
        return None, {"mode": "unavailable"}
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise RuntimeError("PDF 精确复核渲染需要安装 PyMuPDF") from exc
    output = artifact_dir / f"{target.target_id}.png"
    scale = dpi / 72
    with fitz.open(source_path) as pdf:
        if not 1 <= target.page_number <= pdf.page_count:
            return None, {"mode": "unavailable", "reason": "page_out_of_range"}
        page = pdf[target.page_number - 1]
        clip = None
        mode = "full_page"
        if target.bbox:
            clip = fitz.Rect(target.bbox.x0, target.bbox.y0, target.bbox.x1, target.bbox.y1)
            mode = "target_bbox"
        elif AnomalyType.MISSING_NUMERIC_FACT in target.anomalies:
            clip = _numeric_clip(page, [str(value) for value in target.evidence.get("missing_numeric_tokens", [])])
            mode = "numeric_context_crop" if clip else "full_page"
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        pixmap.save(output)
    details: dict[str, Any] = {"mode": mode, "page_number": target.page_number}
    if clip:
        details["clip"] = [clip.x0, clip.y0, clip.x1, clip.y1]
    return output, details


def _numeric_clip(page: Any, tokens: list[str]) -> Any | None:
    """利用 PDF 文字层把数值异常裁成局部区域；找不到时才回退整页。"""
    rectangles: list[Any] = []
    for token in tokens:
        candidates = [token]
        whole, dot, fraction = token.partition(".")
        if whole.lstrip("-").isdigit() and len(whole.lstrip("-")) >= 4:
            grouped = f"{int(whole):,}"
            candidates.append(grouped + (dot + fraction if dot else ""))
        for candidate in dict.fromkeys(candidates):
            rectangles.extend(page.search_for(candidate))
    if not rectangles:
        return None
    clip = rectangles[0]
    for rectangle in rectangles[1:]:
        clip |= rectangle
    page_rect = page.rect
    # 数值事实通常横跨一整行；保留全行避免把主语、单位或句尾截断。
    clip.x0 = page_rect.x0
    clip.x1 = page_rect.x1
    clip.y0 = max(page_rect.y0, clip.y0 - 100)
    clip.y1 = min(page_rect.y1, clip.y1 + 100)
    return clip


def _resolve_asset(document: ParsedDocument, target: ReviewTarget) -> Path | None:
    wanted = {asset.relative_path.replace("\\", "/") for asset in document.assets if asset.asset_id in target.asset_ids}
    for raw in document.raw_artifact_paths:
        path = Path(raw)
        normalized = str(path).replace("\\", "/")
        if path.is_file() and any(normalized.endswith(value) for value in wanted):
            return path
    return None


def _assess(
    target: ReviewTarget, observed: list[dict[str, Any]], uncertain: list[str]
) -> tuple[list[AnomalyType], list[AnomalyType]]:
    serialized = json.dumps(observed, ensure_ascii=False)
    compact = "".join(serialized.split())
    types = {str(item.get("type", "")) for item in observed}
    has_content = any(_semantic_text(item) for item in observed)
    resolved: list[AnomalyType] = []
    unresolved: list[AnomalyType] = []
    for anomaly in target.anomalies:
        ok = False
        if anomaly == AnomalyType.MISSING_NUMERIC_FACT:
            expected = set(target.evidence.get("missing_numeric_tokens", []))
            ok = bool(expected & numeric_tokens(serialized))
        elif anomaly in {AnomalyType.MISSING_TEXT, AnomalyType.LOW_TEXT_COVERAGE}:
            fragments = [str(value) for value in target.evidence.get("missing_fragments", [])]
            ok = has_content and (not fragments or any("".join(value.split()) in compact for value in fragments))
        elif anomaly == AnomalyType.EMPTY_UNIT:
            ok = has_content
        elif anomaly == AnomalyType.TABLE_STRUCTURE:
            ok = "table" in types
        elif anomaly == AnomalyType.FORMULA_STRUCTURE:
            ok = "formula" in types
        elif anomaly == AnomalyType.VISUAL_SEMANTICS:
            ok = bool(types & {"chart", "table", "formula", "image"}) and has_content
        elif anomaly == AnomalyType.READING_ORDER:
            orders = [item.get("reading_order") for item in observed]
            ok = bool(orders) and all(isinstance(value, int) for value in orders)
        if uncertain and anomaly not in {
            AnomalyType.MISSING_NUMERIC_FACT,
            AnomalyType.VISUAL_SEMANTICS,
        }:
            ok = False
        (resolved if ok else unresolved).append(anomaly)
    return resolved, unresolved


def _semantic_text(item: dict[str, Any]) -> str:
    values = [item.get("verbatim_text"), item.get("formula")]
    if item.get("table"):
        values.append(json.dumps(item["table"], ensure_ascii=False))
    if item.get("chart"):
        values.append(json.dumps(item["chart"], ensure_ascii=False))
    return " ".join(str(value) for value in values if value).strip()


def _review_element(
    document: ParsedDocument, target: ReviewTarget, item: dict[str, Any], order: int, model: str
) -> DocumentElement | None:
    text = _semantic_text(item)
    if not text:
        return None
    kind = {
        "heading": ElementType.HEADING, "paragraph": ElementType.PARAGRAPH,
        "list": ElementType.LIST, "formula": ElementType.FORMULA,
        "header": ElementType.HEADER, "footer": ElementType.FOOTER,
    }.get(str(item.get("type")), ElementType.PARAGRAPH)
    location = SourceLocation(
        page_number=target.page_number, slide_number=target.slide_number,
        sheet_name=target.sheet_name, cell_range=target.cell_range, bbox=target.bbox,
        parser_locator=f"qwen_review:{target.target_id}:{order}",
    )
    return DocumentElement(
        element_id=stable_element_id(document.source.document_id, kind, order, location.parser_locator or ""),
        element_type=kind, order=order, text=text, source_location=location,
        metadata={
            "source": "qwen_targeted_review", "model": model, "target_id": target.target_id,
            "observed_type": item.get("type"), "confidence": item.get("confidence"),
        },
    )


def _unreviewed(target: ReviewTarget, reason: str) -> ReviewFinding:
    return ReviewFinding(
        target_id=target.target_id, status="unreviewed",
        unresolved_anomalies=target.anomalies, uncertain_items=[reason],
    )


assess_review_content = _assess


__all__ = [
    "QwenDocumentReviewer", "assess_review_content",
    "attach_review_targets", "build_review_prompt",
]
