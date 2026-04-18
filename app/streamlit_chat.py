import os
import sys
from datetime import datetime
from html import escape
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import config
from generation.rag_service import RagService
from memory.history_store import (
    delete_history,
    get_history,
    is_session_pinned,
    list_session_ids,
    toggle_session_pinned,
)

load_dotenv()


def new_session_id() -> str:
    return f"chat_{uuid4().hex[:12]}"


def default_messages() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": "你好，我是一个法律问答助手，我有什么可以帮助你？"}]


def load_messages_from_history(session_id: str) -> list[dict[str, str]]:
    """从历史存储加载会话消息，并过滤内部摘要消息。

    说明：
    - history_store 会写入 type=system 的内部摘要消息，用于给模型提供长期记忆。
    - UI 展示层不应把这些内部摘要暴露给用户，因此这里做过滤。
    """
    # history: 指定会话的历史对象。
    history = get_history(session_id)
    # messages: Streamlit 前端使用的 role/content 结构列表。
    messages = []
    for msg in history.messages:
        # summary_tag: 用于识别内部摘要消息的前缀。
        summary_tag = config.memory_summary_tag
        if msg.type == "system" and str(msg.content).startswith(summary_tag):
            continue

        # role: LangChain 消息类型映射为前端角色类型。
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})
    return messages or default_messages()


def session_label(session_id: str) -> str:
    history = get_history(session_id)
    first_user_message = ""
    for msg in history.messages:
        if msg.type == "human" and isinstance(msg.content, str):
            first_user_message = msg.content.strip()
            break
    preview = first_user_message[:15] if first_user_message else "新会话"
    return preview


def _consume_query_action() -> None:
    action = st.query_params.get("action")
    sid = st.query_params.get("sid")
    if not action:
        return

    session_ids = list_session_ids()
    active_sid = st.session_state.get("active_session_id")

    if action == "new":
        created_id = new_session_id()
        st.session_state["active_session_id"] = created_id
        st.session_state["message"] = default_messages()
    elif action == "open" and sid and sid in session_ids:
        st.session_state["active_session_id"] = sid
        st.session_state["message"] = load_messages_from_history(sid)
    elif action == "delete" and sid:
        delete_history(sid)
        remaining = list_session_ids()
        if not remaining:
            created_id = new_session_id()
            st.session_state["active_session_id"] = created_id
            st.session_state["message"] = default_messages()
        else:
            if active_sid == sid:
                st.session_state["active_session_id"] = remaining[0]
                st.session_state["message"] = load_messages_from_history(remaining[0])
    elif action == "pin" and sid:
        toggle_session_pinned(sid)

    st.query_params.clear()
    st.rerun()


def _render_sidebar_history(active_session_id: str, session_ids: list[str]) -> None:
    rows: list[str] = []
    for sid in session_ids:
        label = escape(session_label(sid))
        active_cls = "session-item active" if sid == active_session_id else "session-item"
        pin_text = "取消置顶" if is_session_pinned(sid) else "置顶"
        pin_mark = "📌 " if is_session_pinned(sid) else ""
        rows.append(
            (
                f'<div class="{active_cls}">'
                f'<a class="session-link" href="?action=open&sid={sid}" target="_self">{pin_mark}{label}</a>'
                '<details class="session-menu">'
                '<summary>&#8942;</summary>'
                '<div class="menu-panel">'
                f'<a href="?action=pin&sid={sid}" target="_self">{pin_text}</a>'
                f'<a class="danger" href="?action=delete&sid={sid}" target="_self">删除</a>'
                "</div>"
                "</details>"
                "</div>"
            )
        )

    html = (
        '<div class="history-shell">'
        '<a class="new-chat-btn" href="?action=new" target="_self">⊕ 新建会话</a>'
        '<div class="history-header">🕘 历史会话</div>'
        '<div class="history-list">'
        + "".join(rows)
        + "</div>"
        "</div>"
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


import markdown
from markupsafe import escape

def render_bubble(role: str, content: str) -> None:
    # safe_content = escape(content).replace("\n", "<br>") # Removed to support markdown
    # Convert markdown to HTML while explicitly allowing tables, code blocks, lists
    md_html = markdown.markdown(
        content,
        extensions=[
            "fenced_code",
            "tables",
            "nl2br",
            "sane_lists"
        ]
    )
    
    cls = "bubble-user" if role == "user" else "bubble-assistant"
    align = "row-user" if role == "user" else "row-assistant"
    st.markdown(
        (
            f'<div class="chat-row {align}">' 
            f'<div class="chat-bubble {cls}">{md_html}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="智能法律问答系统", page_icon="⚖️", layout="centered")

st.markdown(
    """
<style>
    [data-testid="stSidebar"] {
        background: #f0f4f9;
        border-right: none;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.1rem;
        padding-bottom: 0.8rem;
    }
    .history-shell {
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
    }
    .new-chat-btn {
        display: block;
        text-decoration: none !important;
        border: 1px solid #d4dae3;
        color: #1f1f1f !important;
        border-radius: 12px;
        padding: 0.5rem 0.75rem;
        font-weight: 600;
        background: #f9fafb;
        margin-bottom: 1rem;
    }
    .new-chat-btn:hover {
        border-color: #b9c4d4;
        background: #ffffff;
    }
    .history-header {
        font-size: 0.92rem;
        color: #444746 !important;
        margin-bottom: 0.55rem;
        font-weight: 500;
        padding-left: 0.2rem;
    }
    .history-list {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }
    .session-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 50px;
        padding: 0;
        margin-bottom: 0.15rem;
    }
    .session-item:hover {
        background: #e1e5ea;
    }
    .session-item.active {
        background: #d3e3fd;
    }
    .session-item.active .session-link {
        color: #041e49 !important;
        font-weight: 500;
    }
    .session-link {
        text-decoration: none !important;
        color: #444746 !important;
        font-size: 0.9rem;
        padding: 0.55rem 0.8rem;
        border-radius: 50px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        width: 84%;
        display: inline-block;
    }
    .session-menu {
        position: relative;
        width: 1.8rem;
        text-align: center;
        opacity: 0;
        transition: opacity 0.16s ease;
    }
    .session-item:hover .session-menu,
    .session-menu[open] {
        opacity: 1;
    }
    .session-menu summary {
        list-style: none;
        cursor: pointer;
        color: #444746 !important;
        font-size: 1.2rem;
        line-height: 1;
        border-radius: 50%;
        padding: 0.15rem;
        margin-right: 0.4rem;
        font-weight: bold;
    }
    .session-menu summary:hover {
        background: rgba(31, 31, 31, 0.08);
    }
    .session-menu summary::-webkit-details-marker {
        display: none;
    }
    .menu-panel {
        position: absolute;
        right: 0.1rem;
        top: 1.55rem;
        background: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        min-width: 6.3rem;
        z-index: 1000;
        overflow: hidden;
    }
    .menu-panel a {
        display: block;
        padding: 0.46rem 0.7rem;
        text-decoration: none !important;
        color: #1f1f1f !important;
        font-size: 0.88rem;
    }
    .menu-panel a:hover {
        background: #f3f4f6;
    }
    .menu-panel a.danger {
        color: #dc2626;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 860px;
        margin: 0 auto;
    }
    .chat-wrap {
        padding: 0.3rem 0.2rem 0.6rem 0.2rem;
    }
    .chat-row {
        display: flex;
        margin: 0.4rem 0;
        width: 100%;
    }
    .row-user {
        justify-content: flex-end;
    }
    .row-assistant {
        justify-content: flex-start;
    }
    .chat-bubble {
        max-width: 72%;
        border-radius: 14px;
        padding: 0.65rem 0.85rem;
        line-height: 1.55;
        font-size: 0.96rem;
        font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
        word-break: break-word;
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.06);
    }
    .bubble-user {
        background: linear-gradient(135deg, #f7f7f8 0%, #eceef1 100%);
        color: #1f2937;
        border-bottom-right-radius: 4px;
    }
    .bubble-assistant {
        background: #ffffff;
        color: #1f2937;
        border-bottom-left-radius: 4px;
    }
    .chat-bubble p {
        margin: 0 0 0.6rem 0;
    }
    .chat-bubble p:last-child {
        margin-bottom: 0;
    }
    .chat-bubble h1, .chat-bubble h2, .chat-bubble h3, .chat-bubble h4 {
        margin: 0.4rem 0 0.6rem 0;
        font-weight: 600;
        line-height: 1.3;
    }
    .chat-bubble h1 { font-size: 1.15rem; }
    .chat-bubble h2 { font-size: 1.1rem; }
    .chat-bubble h3 { font-size: 1.05rem; }
    .chat-bubble ul, .chat-bubble ol {
        margin: 0 0 0.6rem 0;
        padding-left: 1.5rem;
    }
    .chat-bubble pre {
        background: #f1f3f4;
        padding: 0.6rem;
        border-radius: 6px;
        overflow-x: auto;
        margin: 0 0 0.6rem 0;
    }
    .chat-bubble code {
        font-family: Consolas, Monaco, "Andale Mono", monospace;
        background: rgba(0, 0, 0, 0.05);
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .chat-bubble pre code {
        padding: 0;
        background: transparent;
    }
    .chat-bubble blockquote {
        margin: 0 0 0.6rem 0;
        padding-left: 1rem;
        border-left: 3px solid #ccc;
        color: #666;
    }
    .chat-bubble table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 0.6rem;
    }
    .chat-bubble th, .chat-bubble td {
        border: 1px solid #d1d5db;
        padding: 0.3rem 0.5rem;
    }
    .session-caption {
        color: #5f6368;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }
    @media (max-width: 768px) {
        .chat-bubble {
            max-width: 88%;
            font-size: 0.93rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("智能法律问答系统")
st.caption("刑法问答助手")
st.divider()

if "session_ids" not in st.session_state:
    sessions = list_session_ids()
    st.session_state["session_ids"] = sessions or [new_session_id()]

if "active_session_id" not in st.session_state:
    st.session_state["active_session_id"] = st.session_state["session_ids"][0]

if st.session_state["active_session_id"] not in st.session_state["session_ids"]:
    st.session_state["session_ids"].append(st.session_state["active_session_id"])

_consume_query_action()

# Refresh sessions list for UI
disk_sessions = list_session_ids() or []
if st.session_state["active_session_id"] not in disk_sessions:
    disk_sessions.insert(0, st.session_state["active_session_id"])
st.session_state["session_ids"] = disk_sessions

_render_sidebar_history(
    active_session_id=st.session_state["active_session_id"],
    session_ids=st.session_state["session_ids"],
)

session_config = config.build_session_config(st.session_state["active_session_id"])

if "message" not in st.session_state:
    st.session_state["message"] = load_messages_from_history(st.session_state["active_session_id"])

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
for message in st.session_state["message"]:
    render_bubble(message["role"], message["content"])
st.markdown("</div>", unsafe_allow_html=True)

prompt = st.chat_input("请输入你的问题")
if prompt:
    st.session_state["message"].append({"role": "user", "content": prompt})
    render_bubble("user", prompt)

    ai_res_list = []
    with st.spinner("思考中..."):
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        with st.empty():
            ai_text = st.write_stream(capture(res_stream, ai_res_list))

        final_ai_text = ai_text if isinstance(ai_text, str) else "".join(ai_res_list)
        st.session_state["message"].append({"role": "assistant", "content": final_ai_text})
        st.rerun()
