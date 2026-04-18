"""混合召回效果预览脚本。

运行：
    .\\.venv\\Scripts\\python.exe test/preview_hybrid_recall.py
"""

import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import config
from generation.rag_service import RagService

load_dotenv()


if __name__ == "__main__":
    query = "抢劫和盗窃有什么区别"
    rag = RagService()
    docs = rag.hybrid_retriever.retrieve(query)

    print("=== hybrid config ===")
    print(
        {
            "vector_k": config.hybrid_vector_k,
            "bm25_k": config.hybrid_bm25_k,
            "final_k": config.hybrid_final_k,
            "rrf_k": config.hybrid_rrf_k,
        }
    )
    print("=== query ===")
    print(query)
    print("=== docs ===", len(docs))

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        print(f"\n[{i}] source={meta.get('source','')} chapter={meta.get('chapter','')} article={meta.get('article_no','')}")
        print(doc.page_content[:220].replace("\n", " "))
