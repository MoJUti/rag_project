"""把通过验收的解析实验从 logs 提升为后续节点使用的评测工件。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ingestion.document_model import ParsedDocument
from ingestion.storage import StorageLayout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--storage-root", type=Path)
    args = parser.parse_args()

    layout = StorageLayout(args.storage_root.resolve()) if args.storage_root else StorageLayout.from_env()
    evaluation = layout.evaluation(args.dataset_id)
    evaluation.ensure()
    report = json.loads(args.report.read_text(encoding="utf-8"))

    for record in report.get("records", []):
        document_id = record["document_id"]
        old_parsed = Path(record["parsed_document"])
        document = ParsedDocument.model_validate_json(old_parsed.read_text(encoding="utf-8"))
        asset_root = evaluation.assets / document_id
        artifact_files = [Path(value) for value in document.raw_artifact_paths if Path(value).is_file()]

        for asset in document.assets:
            source = _find_asset(asset.relative_path, artifact_files, document.raw_artifact_paths)
            if source is None:
                raise FileNotFoundError(f"找不到资产 {document_id}: {asset.relative_path}")
            destination = asset_root / asset.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        promoted_review_paths: list[str] = []
        for finding in record.get("findings", []):
            raw = finding.get("artifact_path")
            if not raw:
                continue
            source = Path(raw)
            destination = asset_root / "review_targets" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            finding["artifact_path"] = str(destination.resolve())
            promoted_review_paths.append(str(destination.resolve()))

        promoted = document.model_copy(update={
            "raw_artifact_paths": [str(asset_root.resolve()), *promoted_review_paths],
        })
        parsed_dir = evaluation.parsed / document_id
        parsed_dir.mkdir(parents=True, exist_ok=True)
        parsed_path = parsed_dir / "parsed_document.json"
        parsed_path.write_text(promoted.model_dump_json(indent=2), encoding="utf-8")
        record["parsed_document"] = str(parsed_path.resolve())

    report["promoted_from"] = str(args.report.resolve())
    report["evaluation_root"] = str(evaluation.root.resolve())
    report_path = evaluation.reports / "node3_final_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path.resolve())}, ensure_ascii=False))


def _find_asset(
    relative_path: str,
    artifact_files: list[Path],
    raw_paths: list[str],
) -> Path | None:
    normalized = Path(relative_path).as_posix().lower()
    for path in artifact_files:
        if path.as_posix().lower().endswith(normalized):
            return path
    for raw in raw_paths:
        root = Path(raw)
        if root.is_dir():
            candidate = root / relative_path
            if candidate.is_file():
                return candidate
    return None


if __name__ == "__main__":
    main()
