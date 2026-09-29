"""多格式解析执行编排。"""

from __future__ import annotations

from pathlib import Path

from ingestion.document_model import ParsedDocument
from ingestion.parsers.anomaly_detector import DocumentAnomalyDetector
from ingestion.parsers.base import ParseRequest
from ingestion.parsers.local_text import LocalTextParser
from ingestion.parsers.mineru_adapter import MinerUResultAdapter
from ingestion.parsers.mineru_client import MinerUClient
from ingestion.parsers.native_extractors import extract_native_snapshot
from ingestion.parsers.native_repair import NativeStructureRepairer
from ingestion.parsers.quality_gate import DocumentQualityGate
from ingestion.parsers.qwen_fallback import QwenDocumentReviewer, attach_review_targets
from ingestion.parsers.routing import ParseBatch, ParserRoute, build_parse_batches, plan_document
from ingestion.storage import KnowledgeBaseStorage, StorageLayout


class ParserExecutionService:
    def __init__(
        self,
        mineru_client: MinerUClient | None = None,
        qwen_reviewer: QwenDocumentReviewer | None = None,
        storage_layout: StorageLayout | None = None,
    ) -> None:
        self.local_parser = LocalTextParser()
        self.mineru_client = mineru_client
        self.qwen_reviewer = qwen_reviewer
        self.adapter = MinerUResultAdapter()
        self.quality_gate = DocumentQualityGate()
        self.anomaly_detector = DocumentAnomalyDetector()
        self.native_repairer = NativeStructureRepairer()
        self.storage = KnowledgeBaseStorage(storage_layout)

    async def parse_paths(self, paths: list[Path]) -> list[ParsedDocument]:
        planned = [plan_document(path) for path in paths]
        batches = build_parse_batches(planned)
        documents: list[ParsedDocument] = []
        for batch in batches:
            if batch.route == ParserRoute.LOCAL_TEXT:
                for item in batch.documents:
                    documents.append(
                        await self.local_parser.parse(
                            ParseRequest(
                                source_path=item.source_path,
                                document_format=item.document_format,
                                document_id=item.document_id,
                            )
                        )
                    )
            else:
                documents.extend(await self._parse_mineru_batch(batch))

        source_by_id = {item.document_id: item.source_path for item in planned}
        persisted: list[ParsedDocument] = []
        for document in documents:
            stored_source = self.storage.save_source(
                source_by_id[document.source.document_id], document.source.document_id,
            )
            updated_source = document.source.model_copy(update={"source_uri": str(stored_source)})
            updated = document.model_copy(update={"source": updated_source})
            self.storage.save_parsed_document(updated)
            persisted.append(updated)
        return persisted

    async def _parse_mineru_batch(self, batch: ParseBatch) -> list[ParsedDocument]:
        if self.mineru_client is None:
            raise RuntimeError("远程文档解析需要配置 MinerUClient")
        batch_id, urls = await self.mineru_client.submit_batch(batch)
        await self.mineru_client.upload_batch(batch, urls)
        results = await self.mineru_client.wait_for_batch(batch_id)
        by_data_id = {result.data_id: result for result in results if result.data_id}
        by_name = {result.file_name: result for result in results}
        documents: list[ParsedDocument] = []
        for item in batch.documents:
            result = by_data_id.get(item.document_id) or by_name.get(item.source_path.name)
            if result is None:
                raise RuntimeError(f"MinerU 批次结果缺少文件: {item.source_path.name}")
            destination = self.storage.layout.asset_dir(item.document_id) / "mineru"
            await self.mineru_client.download_and_extract(result, destination)
            parsed = self.adapter.adapt(
                source_path=item.source_path,
                document_format=item.document_format,
                document_id=item.document_id,
                extracted_dir=destination,
                model_version=batch.model_version or "vlm",
                remote_batch_id=batch_id,
            )
            native = extract_native_snapshot(item.source_path, item.document_format)
            checked = self.quality_gate.evaluate(parsed, native)
            initial_targets = self.anomaly_detector.detect(checked, native)
            checked, _ = self.native_repairer.repair(
                checked, item.source_path, initial_targets,
            )
            checked = self.quality_gate.evaluate(checked, native)
            targets = self.anomaly_detector.detect(checked, native)
            if targets and self.qwen_reviewer is not None:
                checked, _ = self.qwen_reviewer.review(
                    checked, item.source_path, native,
                    self.storage.layout.asset_dir(item.document_id) / "review_targets", targets,
                )
            elif targets:
                checked = attach_review_targets(checked, targets)
            documents.append(checked)
        return documents


__all__ = ["ParserExecutionService"]
