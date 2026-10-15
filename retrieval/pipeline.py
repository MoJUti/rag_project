"""组合查询处理、候选召回、融合、重排和证据扩展，并返回逐阶段记录。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from langchain_core.documents import Document

from retrieval.hybrid_retriever import _doc_key, rrf_fuse
from retrieval.qwen_reranker import QwenTextReranker, RerankResult


class CandidateRetriever(Protocol):
    """新召回插件只需返回有稳定来源标识的候选文档。"""

    def get_candidates(self, query: str, limit: int) -> list[Document]: ...


class Reranker(Protocol):
    """重排只返回输入下标，防止模型生成的文本冒充原始证据。"""

    def rerank(self, query: str, documents: list[str]) -> RerankResult: ...

    def close(self) -> None: ...


RERANKER_FACTORIES: dict[str, Callable[[], Reranker]] = {
    "qwen3.7-text-rerank": QwenTextReranker,
}


def original_query(query: str) -> list[str]:
    """默认用原始提问，后续可替换为改写或问题拆解插件。"""
    return [query]


def keep_evidence(query: str, documents: list[Document]) -> list[Document]:
    """默认保留命中切块，后续可替换为父块或邻接块扩展。"""
    return documents


@dataclass
class StageTrace:
    """单次请求的阶段记录，可直接转为JSON供界面展示。"""

    stage: str
    elapsed_ms: float
    chunk_ids: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    total_tokens: int = 0
    request_id: str = ""
    added_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    documents: list[Document]
    trace: list[StageTrace]


@dataclass
class RetrievalPipeline:
    """插件通过统一输入输出组合；每次请求独立记录，不共享调试状态。"""

    retriever: CandidateRetriever
    final_k: int = 10
    candidate_k: int = 40
    max_queries: int = 4
    rerank_strategy: str = "none"
    query_transform: Callable[[str], list[str]] = original_query
    evidence_expander: Callable[[str, list[Document]], list[Document]] = keep_evidence
    fusion: Callable[[list[list[Document]], int], list[Document]] = rrf_fuse

    def __post_init__(self) -> None:
        if self.final_k < 1 or self.candidate_k < self.final_k:
            raise ValueError("最终数量必须大于0，候选数量不能小于最终数量")
        if self.max_queries < 1:
            raise ValueError("最大子问题数量必须大于0")
        if self.rerank_strategy != "none" and self.rerank_strategy not in RERANKER_FACTORIES:
            raise ValueError(f"不支持的重排策略：{self.rerank_strategy}")

    def run(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise ValueError("检索问题不能为空")
        trace: list[StageTrace] = []
        started = time.perf_counter()
        transformed_queries = self.query_transform(query)
        if not isinstance(transformed_queries, list) or not transformed_queries:
            raise ValueError("查询处理插件必须返回非空问题列表")
        if any(not isinstance(item, str) or not item.strip() for item in transformed_queries):
            raise ValueError("查询处理插件返回了无效问题")
        queries = list(dict.fromkeys(item.strip() for item in transformed_queries))
        if len(queries) > self.max_queries:
            raise ValueError(f"查询处理插件超出子问题预算：{len(queries)}/{self.max_queries}")
        trace.append(StageTrace("query", self._elapsed(started), queries=queries))

        # 关闭重排时沿用最终数量；开启后先保留候选池，再按生成预算截断。
        limit = self.final_k if self.rerank_strategy == "none" else self.candidate_k
        ranked_lists: list[list[Document]] = []
        for transformed in queries:
            started = time.perf_counter()
            documents = self.retriever.get_candidates(transformed, limit)
            ranked_lists.append(documents)
            trace.append(StageTrace("recall", self._elapsed(started),
                                    [_doc_key(doc) for doc in documents], [transformed]))

        started = time.perf_counter()
        if len(ranked_lists) == 1:
            documents = list({_doc_key(doc): doc for doc in ranked_lists[0]}.values())[:limit]
        else:
            documents = self.fusion(ranked_lists, limit)
        trace.append(StageTrace("fusion", self._elapsed(started), [_doc_key(doc) for doc in documents]))

        if documents and self.rerank_strategy != "none":
            started = time.perf_counter()
            reranker = RERANKER_FACTORIES[self.rerank_strategy]()
            try:
                ranked = reranker.rerank(query, [doc.page_content for doc in documents])
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(f"重排阶段失败（{self.rerank_strategy}）：{exc}") from exc
            finally:
                reranker.close()
            if sorted(ranked.indices) != list(range(len(documents))):
                raise RuntimeError("重排插件返回的候选下标不完整或重复")
            documents = [documents[index] for index in ranked.indices]
            trace.append(StageTrace("rerank", self._elapsed(started),
                                    [_doc_key(doc) for doc in documents],
                                    total_tokens=ranked.total_tokens, request_id=ranked.request_id))

        started = time.perf_counter()
        seeds = documents[:self.final_k]
        seed_keys = {_doc_key(doc) for doc in seeds}
        documents = self.evidence_expander(query, seeds)
        final_keys = [_doc_key(doc) for doc in documents]
        trace.append(StageTrace("evidence", self._elapsed(started), final_keys,
                                added_chunk_ids=[key for key in final_keys if key not in seed_keys]))
        return RetrievalResult(documents, trace)

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)
