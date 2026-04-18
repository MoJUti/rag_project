from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model

from core import config
from infra.vector_store import VectorStoreService
from memory.history_store import get_history
from retrieval.hybrid_retriever import HybridRetrieverService


def _print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagService:
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                   "system",
                    "你是一位拥有多年实务经验的**刑法专业助手**。你的任务是根据提供的检索证据，为用户提供准确、严谨且易于阅读的法律分析。"

                    "【核心原则】"
                    " **重点突出**：在回复中，使用**加粗**标注关键法律术语、罪名、量刑时间等核心信息，方便用户快速抓取重点。"
                    " **知识范围边界**：如果你判断用户的问题**不属于刑法领域**（如离婚财产分割、租房合同纠纷、劳动补偿等），请不要强制套用证据，应明确提示领域差异，仅给出**通识性的建议与参考方向**，并建议咨询专业律师。"
                    " **证据闭环**：刑法问题必须严格挂载法条编号。如，（依据《刑法》第196条）。若资料缺失，诚实说明，严禁幻觉。"

                    "【输出结构要求】"

                    "---"
                    "### ⚖️ 初步判定"
                    "（结论先行。用通俗且加粗的文字给出定性判断。如果是非刑法问题，请直接在此处说明领域并给出通识建议。）"

                    "### 🔍 深度分析"
                    "（刑法问题专用：要在罪名旁边加上法条。不要枯燥列条号，要结合具体案情解释。示例：**【信用卡诈骗罪】**（依据《刑法》第196条），您的行为属于**数额较大**范畴...）"

                    "### ⚠️ 风险预警与待核实点"
                    "（列出影响量刑的关键变量，如：**是否自首**、**主观目的**、**犯罪未遂**等。若证据中缺少具体金额标准，必须在此明确提示。）"

                    "### 📜 免责声明"
                    "（本回复仅供学习参考，不构成法律意见。法律事务复杂，请务必咨询专业律师。）"

                    "【语气与格式控制】"
                    "- 使用** Markdown 列表**和**分割线**保持页面整洁。"
                    "- 语气要专业但不冰冷，适当使用‘您的行为可能涉及’、‘建议您重点关注’等表达。"
                    "- 严格禁止输出证据库以外的法条（非刑法建议除外）。"
                    "参考资料如下：{context}。\\n并且我提供用户的对话历史记录，如下：",
                ),
                MessagesPlaceholder("history"),
                (
                    "user",
                    "请回答用户提问：{input}。"
                    "如果问题涉及个案定性或量刑，请强调需由司法机关和律师结合具体事实判断。",
                ),
            ]
        )
        self.hybrid_retriever = HybridRetrieverService(
            get_vector_docs=self.vector_service.get_vector_docs,
            get_all_docs=self.vector_service.get_all_documents,
            vector_k=config.hybrid_vector_k,
            bm25_k=config.hybrid_bm25_k,
            final_k=config.hybrid_final_k,
            rrf_k=config.hybrid_rrf_k,
        )
        self.chat_model = _build_chat_model()
        self.chain = self._build_chain()

    def _build_chain(self):
        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            evidence_lines = []
            for i, doc in enumerate(docs, start=1):
                meta = doc.metadata or {}
                source = meta.get("source", "")
                chapter = meta.get("chapter", "")
                article = meta.get("article_no", "")
                evidence_lines.append(
                    f"source={source} chapter={chapter} article={article}\\n"
                    f"原文片段：{doc.page_content}"
                )
            return "\\n\\n".join(evidence_lines)

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def hybrid_retrieve(query: str) -> list[Document]:
            return self.hybrid_retriever.retrieve(query)

        def format_for_prompt_template(value: dict):
            return {
                "input": value["input"]["input"],
                "context": value["context"],
                "history": value["input"]["history"],
            }

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever)
                | RunnableLambda(hybrid_retrieve)
                | format_document,
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | _print_prompt
            | self.chat_model
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )


def _build_chat_model():
    """优先使用 LangChain v1 的统一模型初始化语法。"""
    try:
        return init_chat_model(
            model=config.chat_model_name,
            model_provider="tongyi",
        )
    except Exception:
        # 若本地依赖未完成迁移，回退到社区模型实现，保证可运行。
        return ChatTongyi(model=config.chat_model_name)
