"""FastAPI入口：复用既有RAG和会话存档，仅向本机提供接口。"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from core import config
from generation.model_pool import CompatibleChatModel
from generation.observability import settings_snapshot, stream_with_trace
from generation.rag_service import RagService
from memory.file_history import ContextChatMessageHistory, FileChatMessageHistory, session_paths

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str
    provider: Literal["zhipu", "qwen"] | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空")
        return value


class SessionStore:
    """读取完整存档，回答链另取压缩视图；不删除或自动迁移旧会话。"""

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def history(self, session_id: str) -> FileChatMessageHistory:
        session_paths(self.directory, session_id)
        return FileChatMessageHistory(session_id, self.directory)

    def exists(self, session_id: str) -> bool:
        legacy, archive = session_paths(self.directory, session_id)
        return legacy.is_file() or archive.is_file()

    def create(self) -> dict:
        session_id = f"chat_{uuid4().hex[:12]}"
        self.history(session_id).add_messages([])
        return {"id": session_id, "title": "新对话", "message_count": 0}

    def messages(self, session_id: str) -> list[dict]:
        return [{"role": "user" if m.type == "human" else "assistant", "content": str(m.content)}
                for m in self.history(session_id).messages if m.type in {"human", "ai"}]

    def sessions(self) -> list[dict]:
        root = Path(self.directory)
        paths = {p.name: p for p in root.glob("chat_*") if p.is_file()}
        paths.update({p.stem: p for p in (root / ".memory_v2").glob("chat_*.json") if p.is_file()})
        rows = []
        for key, path in sorted(paths.items(), key=lambda pair: pair[1].stat().st_mtime, reverse=True):
            messages = self.messages(key)
            title = next((m["content"].strip()[:28] for m in messages if m["role"] == "user"), "新对话")
            rows.append({"id": key, "title": title, "message_count": len(messages)})
        return rows


class SessionLease:
    """同一会话拒绝并行问答；断开连接后等待工作线程真正结束再释放。"""

    def __init__(self, lock: Lock) -> None:
        self.lock, self.guard = lock, Lock()
        self.started, self.released = False, False

    def start(self) -> None:
        with self.guard:
            self.started = True

    def release(self, only_unstarted: bool = False) -> None:
        with self.guard:
            if not self.released and not (only_unstarted and self.started):
                self.released = True
                self.lock.release()


def build_service(provider: str, history_factory: Callable) -> RagService:
    """新版默认GLM/low，千问仍可选；不修改旧Streamlit模型池配置。"""
    if provider == "zhipu":
        model = CompatibleChatModel(config.zhipu_chat_model, config.zhipu_base_url,
                                   os.getenv("ZHIPU_API_KEY", "").strip(), provider="zhipu")
    else:
        model = CompatibleChatModel(config.chat_model_name, config.embedding_base_url,
                                   os.getenv("DASHSCOPE_API_KEY", "").strip(), provider="qwen")
    return RagService(chat_model=model, history_factory=history_factory)


def encode_event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(service_factory: Callable = build_service, store: SessionStore | None = None) -> FastAPI:
    app = FastAPI(title="知识库工作台 API", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    sessions = store or SessionStore(config.chat_history_directory)
    services, locks = {}, {}
    service_guard, locks_guard = Lock(), Lock()

    @app.middleware("http")
    async def local_origin_guard(request: Request, call_next: Callable):
        origin = request.headers.get("origin")
        try:
            allowed = not origin or urlsplit(origin).hostname in {"localhost", "127.0.0.1"}
        except ValueError:
            allowed = False
        if not allowed:
            return JSONResponse({"detail": "本地工作台拒绝来自外部站点的请求"}, status_code=403)
        return await call_next(request)

    def require_session(session_id: str) -> None:
        try:
            exists = sessions.exists(session_id)
        except ValueError as exc:
            raise HTTPException(422, "会话ID不合法") from exc
        if not exists:
            raise HTTPException(404, "会话不存在，请新建对话")

    def get_service(provider: str):
        with service_guard:
            if provider not in services:
                services[provider] = service_factory(provider, lambda sid: ContextChatMessageHistory(sessions.history(sid)))
            return services[provider]

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "app": "knowledge-workbench"}

    @app.get("/api/config")
    def get_config() -> dict:
        return {"settings": settings_snapshot(), "default_provider": config.api_chat_provider,
                "providers": [{"id": "zhipu", "label": "GLM · low", "model": config.zhipu_chat_model,
                               "configured": bool(os.getenv("ZHIPU_API_KEY", "").strip())},
                              {"id": "qwen", "label": "千问 · 关闭思考", "model": config.chat_model_name,
                               "configured": bool(os.getenv("DASHSCOPE_API_KEY", "").strip())}]}

    @app.get("/api/sessions")
    def list_sessions() -> list[dict]:
        return sessions.sessions()

    @app.post("/api/sessions", status_code=201)
    def new_session() -> dict:
        return sessions.create()

    @app.get("/api/sessions/{session_id}/messages")
    def messages(session_id: str) -> list[dict]:
        require_session(session_id)
        return sessions.messages(session_id)

    @app.post("/api/sessions/{session_id}/chat")
    def chat(session_id: str, body: ChatRequest):
        require_session(session_id)
        provider = body.provider or config.api_chat_provider
        if provider not in {"zhipu", "qwen"}:
            raise HTTPException(500, "API_CHAT_PROVIDER只能为zhipu或qwen")
        with locks_guard:
            lock = locks.setdefault(session_id, Lock())
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "该会话仍有回答正在处理，请稍后再试")
        lease, stopped, events = SessionLease(lock), Event(), Queue()

        def work() -> None:
            started = time.perf_counter()
            report = {"request_id": uuid4().hex, "status": "failed", "prepared": None,
                      "model_calls": [], "first_text_seconds": None, "total_seconds": 0.0}
            try:
                service = get_service(provider)
                initialization_seconds = time.perf_counter() - started
                stream = stream_with_trace(service, body.query, config.build_session_config(session_id), report,
                    on_prepared=lambda data: events.put(("prepared", data)))
                try:
                    for text in stream:
                        if stopped.is_set():
                            break
                        events.put(("token", {"text": text}))
                finally:
                    stream.close()
                if report.get("first_text_seconds") is not None:
                    report["first_text_seconds"] += initialization_seconds
                report["service_initialization_seconds"] = initialization_seconds
                report["total_seconds"] = time.perf_counter() - started
                if not stopped.is_set():
                    events.put(("done", {"trace": report}))
            except Exception as exc:
                logger.error("问答请求失败：会话=%s，类型=%s", session_id, type(exc).__name__)
                detail = str(exc) if isinstance(exc, (RuntimeError, ValueError)) else "请检查本地后端日志"
                for name in ("ZHIPU_API_KEY", "DASHSCOPE_API_KEY", "MINERU_API_TOKEN", "QDRANT_API_KEY"):
                    secret = os.getenv(name, "").strip()
                    if secret:
                        detail = detail.replace(secret, "[密钥已隐藏]")
                detail = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[密钥已隐藏]", detail)
                report.update(status="failed", error_type=type(exc).__name__, total_seconds=time.perf_counter() - started)
                events.put(("error", {"message": f"回答未完成：{detail}", "trace": report}))
            finally:
                lease.release()
                events.put(("end", {}))

        def stream_events():
            try:
                # 先标记工作已接管，再启动线程，避免断开连接时后台清理误释放其会话锁。
                lease.start()
                try:
                    Thread(target=work, daemon=True).start()
                except RuntimeError:
                    lease.release()
                    raise
                yield encode_event("start", {"session_id": session_id, "provider": provider})
                while True:
                    try:
                        name, data = events.get(timeout=1)
                    except Empty:
                        yield ": keepalive\n\n"
                        continue
                    if name == "end":
                        break
                    yield encode_event(name, data)
            finally:
                stopped.set()

        return StreamingResponse(stream_events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            background=BackgroundTask(lease.release, only_unstarted=True))

    dist = ROOT / "frontend/dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/")
        def missing_frontend():
            return JSONResponse({"detail": "前端尚未构建，请在frontend目录执行npm run build"}, status_code=503)
    return app


app = create_app()
