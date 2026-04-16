import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

md5_path = os.path.join(BASE_DIR, "md5.text")

# Chroma
collection_name="rag"
persist_directory=os.path.join(BASE_DIR, "chroma_db")
chat_history_directory = os.path.join(BASE_DIR, "chat_history")

# spliter
chunk_size= 600
chunk_overlap= 150
chunk_overlap_articles= 0  # 是否允许跨条重叠，设置为0表示不允许跨条重叠
separators =[
    "\n第.{1,5}条\s+",  # 使用正则匹配“第[一二三...]条 ”，通常后面跟着空格或换行
    "\n第.{1,5}章",     # 按章切分也是一种逻辑
    "\n\n", 
    "\n", 
    "。"
]

max_spliter_char_number= 1000  # 文本分割阈值

# 检索参数
retrieval_top_k = 5
retrieval_source_filter = ""  # 例如: "刑法.txt"；空字符串表示不过滤

# 兼容旧字段名
similarity_threshold = retrieval_top_k

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

def build_session_config(session_id: str):
    return {
        "configurable": {
            "session_id": session_id,
        }
    }


session_config = build_session_config("user_001")