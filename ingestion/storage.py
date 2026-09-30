"""正式知识库文件的本地持久化目录约定。"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ingestion.document_model import ParsedDocument


@dataclass(frozen=True)
class StorageLayout:
    """集中管理正式数据路径；评测运行产物仍由脚本显式写入 logs。"""

    root: Path

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "StorageLayout":
        base = (project_root or Path(__file__).resolve().parents[1]).resolve()
        configured = Path(os.getenv("RAG_STORAGE_ROOT", "storage"))
        root = configured if configured.is_absolute() else base / configured
        return cls(root=root.resolve())

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def parsed(self) -> Path:
        return self.root / "parsed"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def chunks(self) -> Path:
        return self.root / "chunks"

    @property
    def indexes(self) -> Path:
        return self.root / "indexes"

    @property
    def evaluations(self) -> Path:
        return self.root / "evaluations"

    def ensure(self) -> None:
        for path in (self.sources, self.parsed, self.assets, self.chunks, self.indexes, self.evaluations):
            path.mkdir(parents=True, exist_ok=True)

    def source_dir(self, document_id: str) -> Path:
        return self.sources / document_id

    def parsed_dir(self, document_id: str) -> Path:
        return self.parsed / document_id

    def asset_dir(self, document_id: str) -> Path:
        return self.assets / document_id

    def chunk_dir(self, document_id: str) -> Path:
        return self.chunks / document_id

    def evaluation(self, dataset_id: str) -> "EvaluationStorageLayout":
        if not dataset_id or any(value in dataset_id for value in ("/", "\\", "..")):
            raise ValueError("dataset_id 必须是单个安全目录名")
        return EvaluationStorageLayout(self.evaluations / dataset_id)


@dataclass(frozen=True)
class EvaluationStorageLayout:
    root: Path

    @property
    def parsed(self) -> Path:
        return self.root / "parsed"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def chunks(self) -> Path:
        return self.root / "chunks"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def indexes(self) -> Path:
        return self.root / "indexes"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> None:
        for path in (self.parsed, self.assets, self.chunks, self.embeddings, self.indexes, self.reports):
            path.mkdir(parents=True, exist_ok=True)


class KnowledgeBaseStorage:
    def __init__(self, layout: StorageLayout | None = None) -> None:
        self.layout = layout or StorageLayout.from_env()
        self.layout.ensure()

    def save_source(self, source_path: Path, document_id: str) -> Path:
        destination_dir = self.layout.source_dir(document_id)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source_path.name
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
        return destination.resolve()

    def save_parsed_document(self, document: ParsedDocument) -> Path:
        destination_dir = self.layout.parsed_dir(document.source.document_id)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "parsed_document.json"
        _atomic_write_text(destination, document.model_dump_json(indent=2))
        return destination.resolve()


def _atomic_write_text(destination: Path, content: str) -> None:
    """同目录临时文件加原子替换，避免中断后留下半个 JSON。"""
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = ["EvaluationStorageLayout", "KnowledgeBaseStorage", "StorageLayout"]
