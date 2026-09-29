"""切块策略注册、自动推荐与旧法律策略适配。"""

from __future__ import annotations

import re

from ingestion.chunk_models import ChunkType, KnowledgeChunk
from ingestion.chunking import ChunkingConfig, FixedTokenChunker, StructureAwareChunker, TokenCounter
from ingestion.document_model import ParsedDocument
from ingestion.legal_chunker import build_chunks_from_article_units, parse_legal_article_units


_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零〇0-9]+\s*条")


class LegalArticleChunker:
    strategy = "legal_article_v1"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.counter = TokenCounter()

    def chunk(self, document: ParsedDocument) -> list[KnowledgeChunk]:
        text = "\n".join(element.text for element in document.elements if element.text.strip())
        units = parse_legal_article_units(text)
        if not units:
            return StructureAwareChunker(self.config).chunk(document)
        texts, metadata = build_chunks_from_article_units(
            units, max_chars=self.config.max_tokens * 2, overlap_articles=0,
        )
        result: list[KnowledgeChunk] = []
        from ingestion.chunk_models import stable_chunk_id
        for index, (value, old_metadata) in enumerate(zip(texts, metadata)):
            result.append(KnowledgeChunk(
                chunk_id=stable_chunk_id(document.source.document_id, self.strategy, index, value),
                document_id=document.source.document_id,
                strategy=self.strategy,
                chunk_type=ChunkType.TEXT,
                text=value,
                token_count=self.counter.count(value),
                section_path=[part for part in (
                    old_metadata.get("part"), old_metadata.get("chapter"), old_metadata.get("section")
                ) if part],
                element_ids=[],
                source_locations=[],
                metadata={
                    **old_metadata,
                    "chunking_config": self.config.model_dump(),
                    "config_hash": self.config.fingerprint(),
                },
            ))
        return result


STRATEGIES = {
    FixedTokenChunker.strategy: FixedTokenChunker,
    StructureAwareChunker.strategy: StructureAwareChunker,
    LegalArticleChunker.strategy: LegalArticleChunker,
}


def build_chunker(strategy: str, config: ChunkingConfig | None = None):
    if strategy not in STRATEGIES:
        raise ValueError(f"未知切块策略: {strategy}; 可选: {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](config)


def recommend_strategy(document: ParsedDocument) -> str:
    text = "\n".join(element.text for element in document.elements[:200] if element.text)
    return LegalArticleChunker.strategy if len(_ARTICLE_RE.findall(text)) >= 3 else StructureAwareChunker.strategy


__all__ = ["LegalArticleChunker", "STRATEGIES", "build_chunker", "recommend_strategy"]
