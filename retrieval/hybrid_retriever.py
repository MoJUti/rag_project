from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from retrieval.tokenizers import TOKENIZERS, char_tokenize


AVAILABLE_RETRIEVAL_STRATEGIES = {
    "dense": "仅向量召回",
    "bm25": "仅BM25关键词召回",
    "hybrid_rrf": "向量与BM25融合召回",
}


def _char_tokenize(text: str) -> list[str]:
    """保留旧评测脚本的字符词元入口。"""
    return char_tokenize(text)


def _doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    if md.get("chunk_id"):
        return str(md["chunk_id"])
    return "|".join(
        [
            str(md.get("source", "")),
            str(md.get("chapter", "")),
            str(md.get("article_no", "")),
            str(md.get("chunk_article_end", "")),
            hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest(),
        ]
    )


def rrf_fuse(
    ranked_lists: list[list[Document]],
    top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """根据各路名次倒数融合；同一路重复切块只计一次分数。"""
    score_map: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for docs in ranked_lists:
        seen: set[str] = set()
        for idx, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            doc_map[key] = doc
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (rrf_k + idx)

    ranked_keys = sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)
    return [doc_map[k] for k in ranked_keys[:top_k]]


@dataclass
class HybridRetrieverService:
    """提供向量、词法和融合候选，最终截断交给检索管线处理。"""

    get_vector_docs: Callable[[str, int], list[Document]]
    get_all_docs: Callable[[], list[Document]]
    get_document_count: Callable[[], int] | None = None
    vector_k: int = 20
    bm25_k: int = 20
    final_k: int = 6
    rrf_k: int = 60
    strategy_id: str = "hybrid_rrf"
    tokenizer_id: str = "char"

    _bm25_retriever: BM25Retriever | None = None
    _bm25_built_on_count: int = -1

    def __post_init__(self) -> None:
        if self.strategy_id not in AVAILABLE_RETRIEVAL_STRATEGIES:
            raise ValueError(f"不支持的召回策略：{self.strategy_id}")
        if self.tokenizer_id not in TOKENIZERS:
            raise ValueError(f"不支持的BM25词元策略：{self.tokenizer_id}")
        if min(self.vector_k, self.bm25_k, self.final_k, self.rrf_k) < 1:
            raise ValueError("召回数量、最终数量和RRF常数必须大于0")

    def invalidate(self) -> None:
        """同数量切块替换时，由写入方主动通知缓存失效。"""
        self._bm25_retriever = None
        self._bm25_built_on_count = -1

    def _ensure_bm25(self) -> None:
        if self.get_document_count is not None and self._bm25_retriever is not None:
            if self.get_document_count() == self._bm25_built_on_count:
                return
        all_docs = self.get_all_docs()
        count = len(all_docs)
        if count == 0:
            self._bm25_retriever = None
            self._bm25_built_on_count = 0
            return

        if self._bm25_retriever is not None and self._bm25_built_on_count == count:
            return

        retriever = BM25Retriever.from_documents(all_docs, preprocess_func=TOKENIZERS[self.tokenizer_id])
        retriever.k = self.bm25_k
        self._bm25_retriever = retriever
        self._bm25_built_on_count = count

    def retrieve(self, query: str) -> list[Document]:
        return self.get_candidates(query, self.final_k)

    def get_candidates(self, query: str, limit: int) -> list[Document]:
        """返回未按生成预算截断的候选，避免重排只能看到最后6条。"""
        if limit < 1:
            raise ValueError("候选数量必须大于0")
        if self.strategy_id == "dense":
            return self.get_vector_docs(query, limit)

        self._ensure_bm25()
        bm25_docs: list[Document] = []
        if self._bm25_retriever is not None:
            lexical_limit = limit if self.strategy_id == "bm25" else self.bm25_k
            self._bm25_retriever.k = lexical_limit
            bm25_docs = self._bm25_retriever.invoke(query)[:lexical_limit]

        if self.strategy_id == "bm25":
            return bm25_docs

        vector_docs = self.get_vector_docs(query, self.vector_k)

        if not vector_docs and not bm25_docs:
            return []

        return rrf_fuse([vector_docs, bm25_docs], top_k=limit, rrf_k=self.rrf_k)
