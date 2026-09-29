"""格式专属解析质量门控。"""

from __future__ import annotations

from ingestion.document_model import (
    DocumentFormat,
    IssueSeverity,
    ParseIssue,
    ParseQuality,
    ParseStatus,
    ParsedDocument,
    QualityMetric,
    QualityStatus,
)
from ingestion.parsers.native_extractors import NativeDocumentSnapshot, missing_numeric_tokens


class DocumentQualityGate:
    def evaluate(self, document: ParsedDocument, native: NativeDocumentSnapshot) -> ParsedDocument:
        parsed_text = "\n".join(
            [element.text for element in document.elements]
            + [cell.text for element in document.elements if element.table for cell in element.table.cells]
        )
        expected_numbers = native.numeric_tokens
        missing_numbers = sorted(missing_numeric_tokens(expected_numbers, parsed_text))
        numeric_score = (
            (len(expected_numbers) - len(missing_numbers)) / len(expected_numbers) if expected_numbers else 1.0
        )
        metrics_by_name = {metric.name: metric for metric in document.quality.metrics}
        metrics_by_name["critical_numeric_recall"] = (
            QualityMetric(
                name="critical_numeric_recall",
                score=numeric_score,
                threshold=1.0,
                passed=not missing_numbers,
                details={
                    "expected_unique": len(expected_numbers),
                    "matched_unique": len(expected_numbers) - len(missing_numbers),
                    "missing_count": len(missing_numbers),
                    "missing_tokens": missing_numbers[:100],
                },
            )
        )

        page_metric = _page_or_slide_metric(document, native)
        if page_metric is not None:
            metrics_by_name[page_metric.name] = page_metric
        metrics = list(metrics_by_name.values())

        issues = [issue for issue in document.issues if issue.code != "CRITICAL_NUMBERS_MISSING"]
        if missing_numbers:
            issues.append(
                ParseIssue(
                    code="CRITICAL_NUMBERS_MISSING",
                    message=f"原生内容中的 {len(missing_numbers)} 个唯一数字未在解析结果中找到",
                    severity=IssueSeverity.WARNING,
                    retryable=True,
                    details={"missing_tokens": missing_numbers[:100]},
                )
            )

        hard_failure = not document.elements or any(
            metric.name in {"nonempty_content", "page_coverage", "slide_coverage"} and not metric.passed
            for metric in metrics
        )
        has_warning = bool(missing_numbers) or any(
            issue.severity in {IssueSeverity.WARNING, IssueSeverity.ERROR} for issue in issues
        )
        quality_status = QualityStatus.FAILED if hard_failure else (
            QualityStatus.WARNING if has_warning else QualityStatus.PASSED
        )
        parse_status = ParseStatus.FAILED if hard_failure else (
            ParseStatus.PARTIAL if has_warning else ParseStatus.SUCCESS
        )
        scores = [metric.score for metric in metrics]
        quality = ParseQuality(
            status=quality_status,
            overall_score=sum(scores) / len(scores) if scores else None,
            metrics=metrics,
        )
        return document.model_copy(update={"status": parse_status, "issues": issues, "quality": quality})


def _page_or_slide_metric(
    document: ParsedDocument, native: NativeDocumentSnapshot
) -> QualityMetric | None:
    if document.source.document_format == DocumentFormat.PDF:
        expected = int(native.metadata.get("page_count", 0))
        actual = len({
            element.source_location.page_number
            for element in document.elements
            if element.source_location and element.source_location.page_number
        })
        name = "page_coverage"
    elif document.source.document_format == DocumentFormat.PPTX:
        expected = int(native.metadata.get("slide_count", 0))
        actual = len({
            element.source_location.slide_number
            for element in document.elements
            if element.source_location and element.source_location.slide_number
        })
        name = "slide_coverage"
    else:
        return None
    score = min(actual / expected, 1.0) if expected else 1.0
    return QualityMetric(
        name=name,
        score=score,
        threshold=1.0,
        passed=actual >= expected,
        details={"expected": expected, "actual": actual},
    )


__all__ = ["DocumentQualityGate"]
