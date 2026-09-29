"""对已下载的 MinerU 产物执行统一模型适配与原生质量门控。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ingestion.document_model import DocumentFormat, sha256_file, stable_document_id
from ingestion.parsers.mineru_adapter import MinerUResultAdapter
from ingestion.parsers.anomaly_detector import DocumentAnomalyDetector
from ingestion.parsers.native_extractors import extract_native_snapshot
from ingestion.parsers.quality_gate import DocumentQualityGate
from ingestion.parsers.routing import detect_document_format


NATIVE_FORMATS = {DocumentFormat.PDF, DocumentFormat.DOCX, DocumentFormat.PPTX, DocumentFormat.XLSX}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.run_root / "summary.json").read_text(encoding="utf-8"))
    source_by_hash = {
        sha256_file(path): path
        for path in (args.dataset_root / "documents").rglob("*")
        if path.is_file()
    }
    adapter = MinerUResultAdapter()
    gate = DocumentQualityGate()
    detector = DocumentAnomalyDetector()
    records: list[dict] = []

    for record in summary["records"]:
        source = source_by_hash.get(record["sha256"])
        if source is None:
            records.append({"index": record["index"], "error": "source_not_found"})
            continue
        document_format = detect_document_format(source)
        extracted = Path(record["output_dir"])
        parsed = adapter.adapt(
            source_path=source,
            document_format=document_format,
            document_id=stable_document_id(record["sha256"]),
            extracted_dir=extracted,
            model_version=summary["model_version"],
            remote_batch_id=summary["batch_id"],
        )
        native = None
        if document_format in NATIVE_FORMATS:
            native = extract_native_snapshot(source, document_format)
            parsed = gate.evaluate(parsed, native)
        detection_snapshot = native or extract_native_snapshot(source, document_format)
        review_targets = detector.detect(parsed, detection_snapshot)
        metrics = {metric.name: metric.model_dump(mode="json") for metric in parsed.quality.metrics}
        records.append(
            {
                "index": record["index"],
                "filename": source.name,
                "format": document_format.value,
                "status": parsed.status.value,
                "quality_status": parsed.quality.status.value,
                "element_count": len(parsed.elements),
                "asset_count": len(parsed.assets),
                "issue_codes": [issue.code for issue in parsed.issues],
                "review_target_count": len(review_targets),
                "vision_target_count": sum(target.requires_vision for target in review_targets),
                "review_targets": [target.model_dump(mode="json") for target in review_targets],
                "metrics": metrics,
                "native": {
                    "table_count": native.table_count,
                    "formula_count": native.formula_count,
                    "merged_range_count": native.merged_range_count,
                    "metadata": native.metadata,
                } if native else None,
            }
        )

    by_format: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        if "format" in record:
            by_format[record["format"]][record["status"]] += 1
    report = {
        "run_root": str(args.run_root),
        "document_count": len(records),
        "status_counts": dict(Counter(record.get("status", "error") for record in records)),
        "by_format": {key: dict(value) for key, value in sorted(by_format.items())},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("document_count", "status_counts", "by_format")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
