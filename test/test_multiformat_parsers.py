from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from pydantic import SecretStr, ValidationError

from ingestion.document_model import (
    DocumentFormat,
    ElementType,
    ParseStatus,
    QualityStatus,
    stable_document_id,
    sha256_file,
)
from ingestion.parsers.base import ParseRequest
from ingestion.parsers.anomaly_detector import DocumentAnomalyDetector
from ingestion.parsers.local_text import LocalTextParser
from ingestion.parsers.mineru_adapter import MinerUResultAdapter
from ingestion.parsers.mineru_client import MinerUApiError, _safe_extract_zip
from ingestion.parsers.native_extractors import NativeDocumentSnapshot, NativeUnit
from ingestion.parsers.native_repair import NativeStructureRepairer
from ingestion.parsers.quality_gate import DocumentQualityGate
from ingestion.parsers.qwen_fallback import QwenDocumentReviewer, build_review_prompt
from ingestion.parsers.qwen_vision import QwenVisionClient, QwenVisionResult, QwenVisionSettings
from ingestion.parsers.review_models import AnomalyType, ReviewTarget, ReviewUnitType


class TestLocalTextParser(unittest.TestCase):
    def test_markdown_preserves_structure_and_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.md"
            path.write_text("# 标题\n\n正文 123。\n\n- 项目一\n- 项目二\n\n```python\nprint(1)\n```\n", encoding="utf-8")
            document_id = stable_document_id(sha256_file(path))
            parsed = asyncio.run(
                LocalTextParser().parse(
                    ParseRequest(
                        source_path=path,
                        document_format=DocumentFormat.MARKDOWN,
                        document_id=document_id,
                    )
                )
            )
            self.assertEqual(parsed.status, ParseStatus.SUCCESS)
            self.assertEqual(
                [element.element_type for element in parsed.elements],
                [ElementType.HEADING, ElementType.PARAGRAPH, ElementType.LIST, ElementType.CODE],
            )
            self.assertEqual(parsed.elements[0].source_location.line_start, 1)
            self.assertEqual(parsed.quality.status, QualityStatus.PASSED)


class TestMinerUAdapter(unittest.TestCase):
    def test_content_list_is_converted_to_stable_elements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            source.write_bytes(b"fake-pdf")
            extracted = root / "extracted"
            images = extracted / "images"
            images.mkdir(parents=True)
            (images / "chart.png").write_bytes(b"png")
            content = [
                {"type": "text", "text": "标题", "text_level": 1, "page_idx": 0, "bbox": [1, 2, 3, 4]},
                {
                    "type": "table",
                    "table_caption": ["统计表"],
                    "table_body": "<table><tr><th>A</th><th>B</th></tr><tr><td rowspan='2'>1</td><td>2</td></tr><tr><td>3</td></tr></table>",
                    "page_idx": 0,
                },
                {"type": "chart", "img_path": "images/chart.png", "page_idx": 1},
            ]
            (extracted / "sample_content_list.json").write_text(
                json.dumps(content, ensure_ascii=False), encoding="utf-8"
            )
            document_id = stable_document_id(sha256_file(source))
            parsed = MinerUResultAdapter().adapt(
                source_path=source,
                document_format=DocumentFormat.PDF,
                document_id=document_id,
                extracted_dir=extracted,
                model_version="vlm",
            )
            self.assertEqual(parsed.status, ParseStatus.SUCCESS)
            self.assertEqual(parsed.elements[0].element_type, ElementType.TITLE)
            self.assertEqual(parsed.elements[1].table.row_count, 3)
            self.assertEqual(parsed.elements[1].table.column_count, 2)
            self.assertEqual(parsed.elements[2].source_location.page_number, 2)
            self.assertEqual(len(parsed.assets), 1)


class TestQualityGate(unittest.TestCase):
    def test_missing_numeric_fact_marks_document_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            source.write_bytes(b"fake-pdf")
            extracted = root / "extracted"
            extracted.mkdir()
            (extracted / "content_list.json").write_text(
                json.dumps([{"type": "text", "text": "销售额为 100 元", "page_idx": 0}], ensure_ascii=False),
                encoding="utf-8",
            )
            document_id = stable_document_id(sha256_file(source))
            parsed = MinerUResultAdapter().adapt(
                source_path=source,
                document_format=DocumentFormat.PDF,
                document_id=document_id,
                extracted_dir=extracted,
                model_version="vlm",
            )
            native = NativeDocumentSnapshot(
                document_format=DocumentFormat.PDF,
                units=[NativeUnit(locator="page:1", text="销售额为 100 元，增长 20%", numeric_tokens={"100", "20"})],
                metadata={"page_count": 1},
            )
            checked = DocumentQualityGate().evaluate(parsed, native)
            self.assertEqual(checked.status, ParseStatus.PARTIAL)
            metric = next(value for value in checked.quality.metrics if value.name == "critical_numeric_recall")
            self.assertEqual(metric.score, 0.5)
            self.assertIn("20", metric.details["missing_tokens"])


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _FakeSession:
    def __init__(self) -> None:
        self.last_payload: dict | None = None

    def post(self, _url: str, **kwargs):
        self.last_payload = kwargs["json"]
        return _FakeResponse(
            {
                "model": "qwen3.8-max",
                "usage": {"total_tokens": 42},
                "choices": [{"message": {"content": "```json\n{\"facts\": []}\n```"}}],
            }
        )


class TestQwenVisionClient(unittest.TestCase):
    def test_thinking_cannot_be_enabled(self) -> None:
        with self.assertRaises(ValidationError):
            QwenVisionSettings(api_key=SecretStr("secret"), enable_thinking=True)

    def test_request_explicitly_disables_thinking_and_parses_fence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.png"
            image.write_bytes(b"png")
            session = _FakeSession()
            client = QwenVisionClient(
                QwenVisionSettings(api_key=SecretStr("secret")), session=session
            )
            result = client.extract([image], "extract")
            self.assertFalse(session.last_payload["enable_thinking"])
            self.assertEqual(result.content, {"facts": []})


class _FakeQwenClient:
    def extract(self, _images, _prompt):
        return QwenVisionResult(
            model="qwen3.8-max",
            content={
                "observed_elements": [
                    {
                        "type": "paragraph",
                        "reading_order": 1,
                        "verbatim_text": "销售额为 100 元，同比增长 20%，该指标用于年度经营分析并作为管理层决策的重要依据，同时需要在报告中完整披露计算口径和统计范围。",
                        "confidence": 0.99,
                    }
                ],
                "uncertain_items": [],
            },
            usage={"total_tokens": 10},
            raw_text='{"facts": []}',
        )


class TestQwenFallback(unittest.TestCase):
    def test_ppt_chart_gap_creates_slide_visual_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pptx"
            source.write_bytes(b"fake-pptx")
            extracted = root / "extracted"
            extracted.mkdir()
            (extracted / "content_list.json").write_text(
                json.dumps([{"type": "text", "text": "季度经营情况", "page_idx": 0}], ensure_ascii=False),
                encoding="utf-8",
            )
            document_id = stable_document_id(sha256_file(source))
            parsed = MinerUResultAdapter().adapt(
                source_path=source, document_format=DocumentFormat.PPTX,
                document_id=document_id, extracted_dir=extracted, model_version="vlm",
            )
            native = NativeDocumentSnapshot(
                document_format=DocumentFormat.PPTX,
                units=[NativeUnit(locator="slide:1", text="季度经营情况", chart_count=1)],
                metadata={"slide_count": 1},
            )
            targets = DocumentAnomalyDetector().detect(parsed, native)
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].slide_number, 1)
            self.assertTrue(targets[0].requires_vision)
            self.assertIn("visual_semantics", [value.value for value in targets[0].anomalies])

    def test_only_the_bad_page_becomes_a_review_target(self) -> None:
        import pymupdf as fitz

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "healthy page 100")
            page = pdf.new_page()
            page.insert_text((72, 72), "sales 100 growth 20")
            pdf.save(source)
            pdf.close()
            extracted = root / "extracted"
            extracted.mkdir()
            (extracted / "content_list.json").write_text(
                json.dumps([
                    {"type": "text", "text": "这是完整的正常页面内容，包含销售额 100 元和全部说明文字。", "page_idx": 0},
                    {"type": "text", "text": "销售额为 100 元，同比增长，该指标用于年度经营分析并作为管理层决策的重要依据，同时需要在报告中完整披露计算口径和统计范围。", "page_idx": 1},
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            document_id = stable_document_id(sha256_file(source))
            parsed = MinerUResultAdapter().adapt(
                source_path=source,
                document_format=DocumentFormat.PDF,
                document_id=document_id,
                extracted_dir=extracted,
                model_version="vlm",
            )
            native = NativeDocumentSnapshot(
                document_format=DocumentFormat.PDF,
                units=[
                    NativeUnit(locator="page:1", text="这是完整的正常页面内容，包含销售额 100 元和全部说明文字。", numeric_tokens={"100"}),
                    NativeUnit(
                        locator="page:2",
                        text="销售额为 100 元，同比增长 20%，该指标用于年度经营分析并作为管理层决策的重要依据，同时需要在报告中完整披露计算口径和统计范围。",
                        numeric_tokens={"100", "20"},
                    ),
                ],
                metadata={"page_count": 2},
            )
            checked = DocumentQualityGate().evaluate(parsed, native)
            targets = DocumentAnomalyDetector().detect(checked, native)
            self.assertEqual([target.page_number for target in targets], [2])
            enriched, findings = QwenDocumentReviewer(_FakeQwenClient()).review(
                checked, source, native, root / "qwen", targets
            )
            self.assertEqual(findings[0].status, "resolved")
            self.assertEqual(enriched.status, ParseStatus.SUCCESS)
            self.assertTrue(any(element.metadata.get("source") == "qwen_targeted_review" for element in enriched.elements))
            self.assertTrue(any(issue.code == "DOCUMENT_REVIEW_TARGETS" for issue in enriched.issues))

    def test_prompt_covers_all_semantic_types(self) -> None:
        from ingestion.parsers.review_models import AnomalyType, ReviewTarget, ReviewUnitType

        target = ReviewTarget(
            target_id="review_1", document_id="doc_1", document_format=DocumentFormat.PPTX,
            unit_type=ReviewUnitType.SLIDE, locator="slide:3", slide_number=3,
            anomalies=[AnomalyType.VISUAL_SEMANTICS], requires_vision=True,
        )
        prompt = build_review_prompt(target)
        for term in ("标题", "段落", "列表", "表格", "图表", "公式", "阅读顺序"):
            self.assertIn(term, prompt)


class TestNativeStructureRepair(unittest.TestCase):
    def test_pdf_low_coverage_page_is_repaired_from_text_blocks(self) -> None:
        import pymupdf as fitz

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            expected = (
                "Annual report table contains revenue 12345 cost 6789 profit 5556 "
                "and complete explanatory notes for deterministic recovery."
            )
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_textbox(fitz.Rect(72, 72, 520, 180), expected, fontsize=11)
            pdf.save(source)
            pdf.close()
            extracted = root / "extracted"
            extracted.mkdir()
            (extracted / "content_list.json").write_text(
                json.dumps([{"type": "text", "text": "1", "page_idx": 0}]), encoding="utf-8"
            )
            document_id = stable_document_id(sha256_file(source))
            parsed = MinerUResultAdapter().adapt(
                source_path=source, document_format=DocumentFormat.PDF,
                document_id=document_id, extracted_dir=extracted, model_version="vlm",
            )
            native = NativeDocumentSnapshot(
                document_format=DocumentFormat.PDF,
                units=[NativeUnit(locator="page:1", text=expected, numeric_tokens={"12345", "6789", "5556"})],
                metadata={"page_count": 1},
            )
            targets = DocumentAnomalyDetector().detect(parsed, native)
            self.assertEqual(len(targets), 1)
            self.assertFalse(targets[0].requires_vision)
            repaired, findings = NativeStructureRepairer().repair(parsed, source, targets)
            remaining = DocumentAnomalyDetector().detect(repaired, native)
            self.assertEqual(remaining, [])
            self.assertGreater(findings[0]["added_element_count"], 0)

    def test_xlsx_formulas_are_written_back_with_cell_locations(self) -> None:
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "数据"
            sheet["A1"] = 10
            sheet["A2"] = 20
            sheet["A3"] = "=SUM(A1:A2)"
            workbook.save(source)
            workbook.close()
            extracted = root / "extracted"
            extracted.mkdir()
            (extracted / "content_list.json").write_text(
                json.dumps([{"type": "text", "text": "10 20 30", "page_idx": 0}]), encoding="utf-8"
            )
            document_id = stable_document_id(sha256_file(source))
            parsed = MinerUResultAdapter().adapt(
                source_path=source, document_format=DocumentFormat.XLSX,
                document_id=document_id, extracted_dir=extracted, model_version="vlm",
            )
            target = ReviewTarget(
                target_id="review_formula", document_id=document_id,
                document_format=DocumentFormat.XLSX, unit_type=ReviewUnitType.SHEET,
                locator="sheet:数据", sheet_name="数据",
                anomalies=[AnomalyType.FORMULA_STRUCTURE], requires_vision=False,
            )
            repaired, findings = NativeStructureRepairer().repair(parsed, source, [target])
            tables = [element for element in repaired.elements if element.element_type == ElementType.TABLE]
            self.assertEqual(len(tables), 1)
            formula_cells = [cell for cell in tables[0].table.cells if cell.formula]
            self.assertEqual(len(formula_cells), 1)
            self.assertEqual(formula_cells[0].formula, "=SUM(A1:A2)")
            self.assertEqual(tables[0].source_location.cell_range, "A1:A3")
            self.assertEqual(findings[0]["status"], "repaired")


class TestSafeZip(unittest.TestCase):
    def test_rejects_path_traversal(self) -> None:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../escape.txt", "bad")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MinerUApiError):
                _safe_extract_zip(stream.getvalue(), Path(directory))


if __name__ == "__main__":
    unittest.main()
