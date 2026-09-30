import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

md5_path = os.path.join(BASE_DIR, "md5.text")

# Vector store（默认继续使用现有 Chroma 本地库；可切换 HTTP/Qdrant）
vector_store_backend = os.getenv("VECTOR_STORE", "chroma").strip().lower()
vector_schema_version = os.getenv("VECTOR_SCHEMA_VERSION", "1").strip()
# 旧 rag 集合是 1024 维。新集合名显式带模型和维度，避免误混入 2048 维向量。
collection_name = os.getenv(
    "VECTOR_COLLECTION",
    "rag_text_embedding_v4_2048",
).strip()
chroma_mode = os.getenv("CHROMA_MODE", "local").strip().lower()
persist_directory = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    os.path.join(BASE_DIR, "chroma_db"),
)
if not os.path.isabs(persist_directory):
    persist_directory = os.path.join(BASE_DIR, persist_directory)
chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1").strip()
chroma_port = int(os.getenv("CHROMA_PORT", "18000"))
chroma_ssl = os.getenv("CHROMA_SSL", "false").strip().lower() in {"1", "true", "yes", "on"}
chroma_hnsw_m = int(os.getenv("CHROMA_HNSW_M", "32"))
chroma_hnsw_ef_construction = int(os.getenv("CHROMA_HNSW_EF_CONSTRUCTION", "200"))
chroma_hnsw_ef_search = int(os.getenv("CHROMA_HNSW_EF_SEARCH", "200"))

qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:16333").strip()
qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
qdrant_hnsw_m = int(os.getenv("QDRANT_HNSW_M", "32"))
qdrant_hnsw_ef_construction = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCTION", "200"))
qdrant_hnsw_ef_search = int(os.getenv("QDRANT_HNSW_EF_SEARCH", "300"))
qdrant_indexing_threshold_kb = int(os.getenv("QDRANT_INDEXING_THRESHOLD_KB", "1000"))
qdrant_full_scan_threshold_kb = int(os.getenv("QDRANT_FULL_SCAN_THRESHOLD_KB", "10"))

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

embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
embedding_base_url = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://maas.qianwenaiapi.com/compatible-mode/v1",
).strip()
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
