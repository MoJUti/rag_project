"""按单次请求采集流程；观察不重复检索，不共享其他会话状态。"""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Callable, Iterator
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.config import merge_configs

from core import config
from generation.prompts import SYSTEM_PROMPT


def settings_snapshot() -> dict:
    """仅列可展示的配置，不读取密钥、鉴权头或完整环境变量。"""
    return {name: getattr(config, name) for name in (
        "vector_store_backend", "collection_name", "embedding_model_name", "embedding_dimensions",
        "retrieval_strategy", "bm25_tokenizer", "hybrid_vector_k", "hybrid_bm25_k", "hybrid_final_k",
        "rerank_strategy", "evidence_expansion", "query_resolution_strategy", "chat_model_name",
        "chat_fallback_provider", "memory_history_max_tokens", "memory_summary_max_tokens")}


class RequestTraceRecorder(BaseCallbackHandler):
    """回调只保存本请求快照；模型阶段按实际系统提示词区分，不把消歧计作回答。"""

    def __init__(self, on_prepared: Callable[[dict], None] | None = None) -> None:
        self.prepared: dict | None = None
        self.model_calls: dict[str, dict] = {}
        self.on_prepared = on_prepared

    def on_custom_event(self, name: str, data: Any, **kwargs: Any) -> None:
        if name == "rag_prepared" and isinstance(data, dict):
            self.prepared = deepcopy(data)
            if self.on_prepared is not None:
                self.on_prepared(deepcopy(data))

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: Any, **kwargs: Any) -> None:
        first = messages[0][0] if messages and messages[0] else None
        phase = "answer" if first is not None and first.type == "system" and first.content == SYSTEM_PROMPT else "other"
        self.model_calls[str(run_id)] = {"phase": phase, "status": "started"}

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        generations = response.generations
        if not generations or not generations[0]:
            return
        generation = generations[0][0]
        metadata = dict(getattr(getattr(generation, "message", None), "response_metadata", {}) or {})
        metadata.update(generation.generation_info or {})
        call = self.model_calls.setdefault(str(run_id), {"phase": "unknown"})
        call.update(status="completed", **{key: deepcopy(metadata[key]) for key in (
            "provider", "model_name", "request_id", "finish_reason") if key in metadata})
        call["usage"] = deepcopy(metadata.get("token_usage") or metadata.get("usage") or {})

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        call = self.model_calls.setdefault(str(run_id), {"phase": "unknown"})
        # 不把第三方异常里潜在的鉴权信息复制进可导出界面记录。
        call.update(status="failed", error_type=type(error).__name__)


def stream_with_trace(service: Any, query: str, run_config: dict, report: dict,
                      on_prepared: Callable[[dict], None] | None = None) -> Iterator[str]:
    """原样流式输出，成功、中断和异常都留下本次记录；不重复调用回答链。"""
    recorder = RequestTraceRecorder(on_prepared)
    report.update(schema_version=1, request_id=uuid4().hex, original_query=query, settings=settings_snapshot())
    started = time.perf_counter()
    first_text, characters, status, error_type = None, 0, "interrupted", None
    try:
        for chunk in service.chain.stream({"input": query}, merge_configs(run_config, {"callbacks": [recorder]})):
            if chunk and first_text is None:
                first_text = time.perf_counter() - started
            characters += len(chunk)
            yield chunk
        status = "completed"
    except Exception as exc:
        # 仅在请求边界记录类型后原样抛出，不吞错、不切换模型或补跑检索。
        status, error_type = "failed", type(exc).__name__
        raise
    finally:
        report.update(status=status, error_type=error_type, first_text_seconds=first_text,
                      total_seconds=time.perf_counter() - started, output_characters=characters,
                      prepared=recorder.prepared, model_calls=list(recorder.model_calls.values()))
