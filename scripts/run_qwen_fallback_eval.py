"""对离线检测出的精确异常目标执行千问复核并保存最终统一文档。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv

from ingestion.document_model import ParsedDocument, sha256_file, stable_document_id
from ingestion.parsers.anomaly_detector import DocumentAnomalyDetector
from ingestion.parsers.mineru_adapter import MinerUResultAdapter
from ingestion.parsers.native_extractors import extract_native_snapshot
from ingestion.parsers.native_repair import NativeStructureRepairer
from ingestion.parsers.quality_gate import DocumentQualityGate
from ingestion.parsers.qwen_fallback import (
    QwenDocumentReviewer, assess_review_content, attach_review_targets,
)
from ingestion.parsers.qwen_vision import QwenVisionClient, QwenVisionSettings
from ingestion.parsers.routing import detect_document_format


def localize_cached_artifacts(findings: dict[str, dict], output_dir: Path) -> None:
    """把复用结果依赖的图片复制到本轮目录，保证最终报告可独立归档。"""
    review_dir = output_dir / "review_targets"
    for finding in findings.values():
        raw_path = finding.get("artifact_path")
        if not raw_path:
            continue
        source = Path(raw_path)
        if not source.is_file():
            raise FileNotFoundError(f"复用的复核图片不存在: {source}")
        review_dir.mkdir(parents=True, exist_ok=True)
        destination = review_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        finding["artifact_path"] = str(destination.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--document", action="append", default=[], help="可选：只处理指定 SHA256 前缀")
    parser.add_argument("--dry-run", action="store_true", help="只生成目标清单，不调用模型")
    parser.add_argument("--reuse-report", type=Path, help="复用旧报告中已解决的模型结果，只重试未解决目标")
    args = parser.parse_args()

    reused_records: dict[str, dict] = {}
    if args.reuse_report:
        previous = json.loads(args.reuse_report.read_text(encoding="utf-8"))
        reused_records = {record["document_id"]: record for record in previous.get("records", [])}

    load_dotenv()
    settings = None if args.dry_run else QwenVisionSettings.from_env()
    reviewer = None if args.dry_run else QwenDocumentReviewer(
        QwenVisionClient(settings), max_targets=args.max_targets
    )
    adapter = MinerUResultAdapter()
    gate = DocumentQualityGate()
    detector = DocumentAnomalyDetector()
    repairer = NativeStructureRepairer()
    summary = json.loads((args.run_root / "summary.json").read_text(encoding="utf-8"))
    source_by_hash = {
        sha256_file(path): path
        for path in (args.dataset_root / "documents").rglob("*")
        if path.is_file()
    }
    records: list[dict] = []

    for record in summary["records"]:
        if args.document and not any(record["sha256"].lower().startswith(value.lower()) for value in args.document):
            continue
        source = source_by_hash.get(record["sha256"])
        if source is None:
            continue
        document_format = detect_document_format(source)
        document_id = stable_document_id(record["sha256"])
        output_dir = args.output_root / document_id
        reused = reused_records.get(document_id)
        if reused:
            old_targets = {
                target["target_id"]: target for target in reused.get("targets", [])
            }
            for finding in reused.get("findings", []):
                old_target = old_targets.get(finding.get("target_id"))
                if not old_target or not finding.get("model"):
                    continue
                from ingestion.parsers.review_models import ReviewTarget
                target_model = ReviewTarget.model_validate(old_target)
                resolved, unresolved = assess_review_content(
                    target_model,
                    finding.get("elements", []),
                    finding.get("uncertain_items", []),
                )
                finding["resolved_anomalies"] = [value.value for value in resolved]
                finding["unresolved_anomalies"] = [value.value for value in unresolved]
                finding["status"] = "resolved" if not unresolved else (
                    "uncertain" if finding.get("uncertain_items") else "unresolved"
                )
        cached_findings = {
            finding["target_id"]: finding
            for finding in (reused.get("findings", []) if reused else [])
            if finding.get("status") == "resolved"
            and not (
                "missing_numeric_fact" in finding.get("resolved_anomalies", [])
                and finding.get("render_details", {}).get("mode") == "mineru_asset"
            )
        }
        localize_cached_artifacts(cached_findings, output_dir)
        reused_path = Path(reused["parsed_document"]) if reused and reused.get("parsed_document") else None
        parsed = adapter.adapt(
            source_path=source,
            document_format=document_format,
            document_id=document_id,
            extracted_dir=Path(record["output_dir"]),
            model_version=summary["model_version"],
            remote_batch_id=summary["batch_id"],
        )
        if reused_path and reused_path.is_file() and cached_findings:
            previous_document = ParsedDocument.model_validate_json(reused_path.read_text(encoding="utf-8"))
            cached_elements = [
                element for element in previous_document.elements
                if element.metadata.get("source") == "qwen_targeted_review"
                and element.metadata.get("target_id") in cached_findings
            ]
            merged_elements = [*parsed.elements, *cached_elements]
            parsed = parsed.model_copy(update={
                "elements": merged_elements,
                "root_element_ids": [element.element_id for element in merged_elements],
            })
        native = extract_native_snapshot(source, document_format)
        before = gate.evaluate(parsed, native)
        initial_targets = detector.detect(before, native)
        repaired, repair_findings = repairer.repair(before, source, initial_targets)
        before = gate.evaluate(repaired, native)
        targets = detector.detect(before, native)
        print(json.dumps({
            "event": "review_targets_detected", "filename": source.name,
            "targets": [target.model_dump(mode="json") for target in targets],
            "model": settings.model if settings else None, "enable_thinking": False,
        }, ensure_ascii=False), flush=True)

        pending_targets = [target for target in targets if target.target_id not in cached_findings]
        if reviewer and pending_targets:
            after, new_findings = reviewer.review(
                before, source, native, output_dir / "review_targets", pending_targets
            )
            findings = [*cached_findings.values(), *[value.model_dump(mode="json") for value in new_findings]]
        else:
            after = attach_review_targets(before, pending_targets)
            findings = list(cached_findings.values())
        output_dir.mkdir(parents=True, exist_ok=True)
        parsed_path = output_dir / "parsed_document.json"
        parsed_path.write_text(after.model_dump_json(indent=2), encoding="utf-8")
        report_targets = [target.model_dump(mode="json") for target in targets]
        known_target_ids = {target["target_id"] for target in report_targets}
        for old_target in (reused.get("targets", []) if reused else []):
            if old_target.get("target_id") in cached_findings and old_target.get("target_id") not in known_target_ids:
                report_targets.append(old_target)
        records.append({
            "filename": source.name,
            "document_id": document_id,
            "format": document_format.value,
            "before_status": before.status.value,
            "after_status": after.status.value,
            "target_count": len(report_targets),
            "initial_target_count": len(initial_targets),
            "repair_findings": repair_findings,
            "vision_target_count": sum(bool(target.get("requires_vision")) for target in report_targets),
            "targets": report_targets,
            "findings": [
                finding if isinstance(finding, dict) else finding.model_dump(mode="json")
                for finding in findings
            ],
            "parsed_document": str(parsed_path.resolve()),
        })

    report = {
        "model": settings.model if settings else None,
        "enable_thinking": False,
        "dry_run": args.dry_run,
        "records": records,
    }
    report_path = args.output_root / "final_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "report": str(report_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
