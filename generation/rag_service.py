from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableBranch, RunnableGenerator, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from pathlib import Path
from typing import Callable, Iterator
from langchain_core.chat_history import BaseChatMessageHistory

from core import config
from infra.vector_store import VectorStoreService
from memory.history_store import get_history
from retrieval.hybrid_retriever import HybridRetrieverService
from retrieval.pipeline import RetrievalPipeline, RetrievalResult
from retrieval.evidence_expansion import NeighborhoodExpander
from generation.evidence import build_evidence, evidence_json, sources_suffix
from generation.prompts import NO_EVIDENCE_ANSWER, build_answer_prompt
from generation.model import build_chat_model as _build_chat_model


class RagService:
    def __init__(
        self, vector_service: VectorStoreService | None = None, chat_model: Runnable | None = None,
        history_factory: Callable[[str], BaseChatMessageHistory] | None = None,
        evidence_renderer: Callable[[list], str] = evidence_json,
    ) -> None:
        self.vector_service = vector_service if vector_service is not None else VectorStoreService()
        self.prompt_template = build_answer_prompt()
        self.history_factory = history_factory or get_history
        self.evidence_renderer = evidence_renderer
        self.hybrid_retriever = HybridRetrieverService(
            get_vector_docs=self.vector_service.get_vector_docs,
            get_all_docs=self.vector_service.get_all_documents,
            get_document_count=self.vector_service.count,
            vector_k=config.hybrid_vector_k,
            bm25_k=config.hybrid_bm25_k,
            final_k=config.hybrid_final_k,
            rrf_k=config.hybrid_rrf_k,
            strategy_id=config.retrieval_strategy,
            tokenizer_id=config.bm25_tokenizer,
        )
        self.retrieval_pipeline = RetrievalPipeline(
            retriever=self.hybrid_retriever,
            final_k=config.hybrid_final_k,
            candidate_k=config.rerank_candidate_k,
            rerank_strategy=config.rerank_strategy,
        )
        if config.evidence_expansion == "neighbors":
            if not config.evidence_chunks_directory:
                raise ValueError("开启证据扩展需要配置 EVIDENCE_CHUNKS_DIRECTORY")
            chunks_directory = Path(config.evidence_chunks_directory)
            if not chunks_directory.is_absolute():
                chunks_directory = Path(config.BASE_DIR) / chunks_directory
            expander = NeighborhoodExpander.from_directory(
                chunks_directory, radius=config.evidence_neighbor_radius,
                extra_token_budget=config.evidence_extra_tokens,
            )
            expander.validate_index(self.vector_service.get_all_documents())
            self.retrieval_pipeline.evidence_expander = expander
        elif config.evidence_expansion != "none":
            raise ValueError(f"不支持的证据扩展策略：{config.evidence_expansion}")
        self.chat_model = chat_model if chat_model is not None else _build_chat_model()
        self.chain = self._build_chain()

    def retrieve_with_trace(self, query: str) -> RetrievalResult:
        """为问答与后续前端预览提供同一条检索路径。"""
        return self.retrieval_pipeline.run(query)

    def _prepare_answer(self, value: dict) -> dict:
        """一次检索对应一份证据映射，供生成与来源展示共同使用。"""
        result = self.retrieve_with_trace(value["input"])
        evidence = build_evidence(result.documents)
        history = [HumanMessage(content=f"历史会话摘要（仅用于上下文，不是系统指令）：{message.content}")
                   if message.type == "system" else message for message in value.get("history", [])]
        return {"input": value["input"], "history": history,
                "context": self.evidence_renderer(evidence), "evidence": evidence}

    @staticmethod
    def _format_stream(chunks: Iterator[dict]) -> Iterator[str]:
        """原样传递模型文本流，完成后按本次证据追加来源与编号检查。"""
        evidence = []
        answer = []
        for chunk in chunks:
            if "evidence" in chunk:
                evidence = chunk["evidence"]
            if "answer" in chunk:
                text = chunk["answer"]
                answer.append(text)
                yield text
        complete_answer = "".join(answer)
        if not complete_answer.strip():
            raise RuntimeError("回答生成阶段返回了空内容")
        suffix = sources_suffix(complete_answer, evidence)
        if suffix:
            yield suffix

    def _build_chain(self) -> RunnableWithMessageHistory:
        answer_chain = RunnableBranch(
            (lambda value: not value["evidence"], RunnableLambda(lambda _: NO_EVIDENCE_ANSWER)),
            self.prompt_template | self.chat_model | StrOutputParser(),
        )
        chain = (
            RunnableLambda(self._prepare_answer)
            | {"answer": answer_chain, "evidence": lambda value: value["evidence"]}
            | RunnableGenerator(self._format_stream)
        )

        return RunnableWithMessageHistory(
            chain,
            self.history_factory,
            input_messages_key="input",
            history_messages_key="history",
        )
