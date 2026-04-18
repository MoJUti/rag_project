import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

md5_path = os.path.join(BASE_DIR, "md5.text")

# Chroma
collection_name = "rag"
persist_directory = os.path.join(BASE_DIR, "chroma_db")
chat_history_directory = os.path.join(BASE_DIR, "chat_history")

# Spliter
chunk_size = 600
chunk_overlap = 150
chunk_overlap_articles = 0
separators = [
    "\\n第.{1,5}条\\s+",
    "\\n第.{1,5}章",
    "\\n\\n",
    "\\n",
    "。",
]

max_spliter_char_number = 1000

# Retrieval
retrieval_top_k = 5
retrieval_source_filter = ""
hybrid_vector_k = 20
hybrid_bm25_k = 20
hybrid_final_k = 6
hybrid_rrf_k = 60

# Backward compatible alias
similarity_threshold = retrieval_top_k

embedding_model_name = "text-embedding-v4"
# 主力推理模型，用于生成最终给用户的专业法律回答
chat_model_name = "qwen3-max"
# 轻量级推理模型，用于生成对话摘要、意图识别等
light_model_name = "qwen-turbo"  # 用于对话摘要、意图识别等轻量级内部任务

# Memory compression strategy config
# 1) 滑动窗口：保留最近 N 轮对话（1 轮 = 用户 1 条 + 助手 1 条）
memory_keep_recent_rounds = 3
# 2) 摘要触发：当累计轮数超过 M 时触发摘要压缩
memory_summary_trigger_rounds = 5
# 3) Token 预算：压缩后的历史消息估算 token 不超过该值
memory_history_max_tokens = 4000
# 4) 摘要最长字符数：约束摘要体积，防止反向膨胀
memory_summary_max_chars = 1500
# 5) 内部摘要开关：允许按环境快速关闭摘要，仅保留滑动窗口
memory_summary_enabled = True
# 6) 内部摘要消息前缀：用于程序识别，展示层可据此隐藏
memory_summary_tag = "[SESSION_SUMMARY]"
# 7) 压缩调试日志开关：为 True 时在控制台打印压缩过程信息
memory_compression_debug = True


def build_session_config(session_id: str):
    return {
        "configurable": {
            "session_id": session_id,
        }
    }


session_config = build_session_config("user_001")
