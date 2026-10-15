"""通用知识库问答约束，与具体模型和检索策略分离。"""

import hashlib
import json
from string import Template

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from core.prompt_loader import load_prompt, render_prompt


NO_EVIDENCE_ANSWER = "当前知识库未检索到可用资料，无法据此回答。请补充相关文档或更具体的问题。"

SYSTEM_PROMPT = render_prompt("generation/knowledge_answer.md", {})
USER_PROMPT = load_prompt("generation/knowledge_question.md", ("context", "input"))


def answer_prompt_sha256() -> str:
    """系统规则和用户模板共同参与评测版本，避免只修改包装模板时误用旧缓存。"""
    content = json.dumps({"system": SYSTEM_PROMPT, "user": USER_PROMPT}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_answer_prompt() -> ChatPromptTemplate:
    """文件控制正文；消息角色与历史插入位置仍由代码管理。"""
    # 先转义静态花括号，再生成框架占位符，避免模板中的JSON示例成为额外变量。
    user_template = Template(USER_PROMPT).substitute(context="\x00context\x00", input="\x00input\x00")
    user_template = user_template.replace("{", "{{").replace("}", "}}")
    user_template = user_template.replace("\x00context\x00", "{context}").replace("\x00input\x00", "{input}")
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("user", user_template),
    ])
