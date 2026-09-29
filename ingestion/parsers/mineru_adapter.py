"""将 MinerU 解压产物适配为统一 ParsedDocument。"""

from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestion.document_model import (
    AssetReference,
    BoundingBox,
    DocumentElement,
    DocumentFormat,
    ElementType,
    ParseIssue,
    ParseQuality,
    ParseStatus,
    ParsedDocument,
    ParserProvenance,
    QualityMetric,
    QualityStatus,
    SourceDocument,
    SourceLocation,
    IssueSeverity,
    config_fingerprint,
    sha256_file,
    stable_element_id,
)
from ingestion.parsers.html_table import parse_html_table, strip_html


_TYPE_MAP = {
    "title": ElementType.TITLE,
    "text": ElementType.PARAGRAPH,
    "list": ElementType.LIST,
    "table": ElementType.TABLE,
    "image": ElementType.FIGURE,
    "chart": ElementType.FIGURE,
    "equation": ElementType.FORMULA,
    "interline_equation": ElementType.FORMULA,
    "code": ElementType.CODE,
    "header": ElementType.HEADER,
    "footer": ElementType.FOOTER,
    "page_footnote": ElementType.FOOTER,
    "page_number": ElementType.FOOTER,
}


class MinerUResultAdapter:
    def adapt(
        self,
        *,
        source_path: Path,
        document_format: DocumentFormat,
        document_id: str,
        extracted_dir: Path,
        model_version: str,
        remote_batch_id: str | None = None,
        remote_task_id: str | None = None,
    ) -> ParsedDocument:
        started_at = datetime.now(timezone.utc)
        content_path = _find_content_list(extracted_dir)
        issues: list[ParseIssue] = []
        if content_path is not None:
            items = json.loads(content_path.read_text(encoding="utf-8"))
        else:
            markdown_path = _find_named(extracted_dir, "full.md")
            if markdown_path is None:
                return self._failed_document(
                    source_path, document_format, document_id, extracted_dir, model_version,
                    "MINERU_OUTPUT_MISSING", "MinerU 结果中没有 content_list.json 或 full.md",
                    remote_batch_id, remote_task_id, started_at,
                )
            items = [{"type": "text", "text": markdown_path.read_text(encoding="utf-8")}]
            issues.append(
                ParseIssue(
                    code="MINERU_CONTENT_LIST_MISSING",
                    message="缺少 content_list.json，已降级使用 full.md",
                    severity=IssueSeverity.WARNING,
                )
            )

        elements: list[DocumentElement] = []
        assets: dict[str, AssetReference] = {}
        for order, item in enumerate(items):
            element, item_assets = _convert_item(item, order, document_id, document_format, extracted_dir)
            if element is not None:
                elements.append(element)
            for asset in item_assets:
                assets.setdefault(asset.asset_id, asset)

        nonempty = any(element.text.strip() or element.table or element.asset_ids for element in elements)
        metric = QualityMetric(
            name="nonempty_content",
            score=1.0 if nonempty else 0.0,
            threshold=1.0,
            passed=nonempty,
            details={"element_count": len(elements)},
        )
        status = ParseStatus.SUCCESS if nonempty and not issues else (ParseStatus.PARTIAL if nonempty else ParseStatus.FAILED)
        quality_status = QualityStatus.PASSED if status == ParseStatus.SUCCESS else (
            QualityStatus.WARNING if status == ParseStatus.PARTIAL else QualityStatus.FAILED
        )
        source_hash = sha256_file(source_path)
        raw_paths = [str(path.resolve()) for path in sorted(extracted_dir.rglob("*")) if path.is_file()]
        return ParsedDocument(
            source=SourceDocument(
                document_id=document_id,
                filename=source_path.name,
                document_format=document_format,
                media_type=mimetypes.guess_type(source_path.name)[0] or "application/octet-stream",
                size_bytes=source_path.stat().st_size,
                sha256=source_hash,
                source_uri=str(source_path.resolve()),
            ),
            status=status,
            parser=ParserProvenance(
                parser_name="mineru",
                parser_version="v4-api",
                model_version=model_version,
                config_fingerprint=config_fingerprint({"model_version": model_version}),
                remote_batch_id=remote_batch_id,
                remote_task_id=remote_task_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            ),
            elements=elements,
            root_element_ids=[element.element_id for element in elements],
            assets=list(assets.values()),
            issues=issues,
            quality=ParseQuality(status=quality_status, overall_score=metric.score, metrics=[metric]),
            raw_artifact_paths=raw_paths,
        )

    def _failed_document(
        self,
        source_path: Path,
        document_format: DocumentFormat,
        document_id: str,
        extracted_dir: Path,
        model_version: str,
        code: str,
        message: str,
        batch_id: str | None,
        task_id: str | None,
        started_at: datetime,
    ) -> ParsedDocument:
        source_hash = sha256_file(source_path)
        return ParsedDocument(
            source=SourceDocument(
                document_id=document_id,
                filename=source_path.name,
                document_format=document_format,
                media_type=mimetypes.guess_type(source_path.name)[0] or "application/octet-stream",
                size_bytes=source_path.stat().st_size,
                sha256=source_hash,
                source_uri=str(source_path.resolve()),
            ),
            status=ParseStatus.FAILED,
            parser=ParserProvenance(
                parser_name="mineru",
                parser_version="v4-api",
                model_version=model_version,
                config_fingerprint=config_fingerprint({"model_version": model_version}),
                remote_batch_id=batch_id,
                remote_task_id=task_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            ),
            issues=[ParseIssue(code=code, message=message, severity=IssueSeverity.ERROR, retryable=True)],
            quality=ParseQuality(status=QualityStatus.FAILED, overall_score=0.0),
            raw_artifact_paths=[str(extracted_dir.resolve())],
        )


def _convert_item(
    item: dict[str, Any],
    order: int,
    document_id: str,
    document_format: DocumentFormat,
    extracted_dir: Path,
) -> tuple[DocumentElement | None, list[AssetReference]]:
    mineru_type = str(item.get("type", "unknown"))
    kind = _TYPE_MAP.get(mineru_type, ElementType.UNKNOWN)
    text_level = item.get("text_level")
    if mineru_type == "text" and isinstance(text_level, int):
        kind = ElementType.TITLE if text_level == 1 else ElementType.HEADING

    text = str(item.get("text") or item.get("content") or "")
    if mineru_type == "list":
        text = "\n".join(str(value) for value in item.get("list_items", []))
    if mineru_type == "table":
        captions = item.get("table_caption") or []
        caption = " ".join(str(value) for value in captions).strip() or None
        table_html = str(item.get("table_body") or "")
        table = parse_html_table(table_html, caption=caption)
        text = "\n".join(filter(None, [caption or "", " | ".join(cell.text for cell in table.cells)]))
    else:
        table = None
        text = strip_html(text)

    page_idx = item.get("page_idx")
    bbox_values = item.get("bbox")
    bbox = None
    if isinstance(bbox_values, list) and len(bbox_values) == 4:
        bbox = BoundingBox(x0=bbox_values[0], y0=bbox_values[1], x1=bbox_values[2], y1=bbox_values[3])
    location = SourceLocation(
        page_number=(page_idx + 1 if isinstance(page_idx, int) and document_format not in {DocumentFormat.PPT, DocumentFormat.PPTX} else None),
        slide_number=(page_idx + 1 if isinstance(page_idx, int) and document_format in {DocumentFormat.PPT, DocumentFormat.PPTX} else None),
        bbox=bbox,
        parser_locator=f"content_list:{order}",
    )

    asset_refs: list[AssetReference] = []
    asset_ids: list[str] = []
    for key in ("img_path", "image_path"):
        relative = item.get(key)
        if not relative:
            continue
        asset_path = (extracted_dir / str(relative)).resolve()
        try:
            asset_path.relative_to(extracted_dir.resolve())
        except ValueError:
            continue
        digest = sha256_file(asset_path) if asset_path.is_file() else None
        asset_id = f"asset_{digest}" if digest else f"asset_{stable_element_id(document_id, 'asset', order, str(relative))[3:]}"
        asset_ids.append(asset_id)
        asset_refs.append(
            AssetReference(
                asset_id=asset_id,
                media_type=mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream",
                relative_path=str(relative).replace("\\", "/"),
                sha256=digest,
                source_location=location,
            )
        )

    locator = f"page:{location.page_number or 0}:content:{order}"
    element = DocumentElement(
        element_id=stable_element_id(document_id, kind, order, locator),
        element_type=kind,
        order=order,
        text=text,
        level=text_level if kind in {ElementType.TITLE, ElementType.HEADING} and isinstance(text_level, int) else None,
        source_location=location,
        table=table,
        asset_ids=asset_ids,
        metadata={"mineru_type": mineru_type},
    )
    return element, asset_refs


def _find_content_list(root: Path) -> Path | None:
    matches = sorted(root.rglob("*content_list*.json"))
    return matches[0] if matches else None


def _find_named(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


__all__ = ["MinerUResultAdapter"]
