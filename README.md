# 智能客服 - 基于 RAG 的大语言模型知识库问答系统

本项目是一个基于 **Retrieval-Augmented Generation (RAG)** 架构的本地知识库问答系统。通过整合文档切片、向量化存储和检索，结合大语言模型，实现带上下文记忆的智能客服功能。

## 🌟 核心特性
- **基于文档检索问答 (RAG)**：通过上传本地知识文档（使用 `Chroma` 向量数据库存储），让大模型基于私有知识进行精准回答。
- **持久化聊天记录**：问答对话会被自动记录到本地（`chat_history/` 目录），即使页面刷新也能还原历史对话，支持独立用户的 Session 管理。
- **Streamlit 双端 Web 界面**：
  - `app_upload.py`：文档上传与知识库构建系统。
  - `app_chat.py`：智能客服对话界面。
- **阿里云通义千问 (DashScope) 接入**：采用 `qwen3-max` 语言模型和 `text-embedding-v4` 向量大模型。

## 📁 目录结构

```text
├── app_chat.py            # 对话界面 (Streamlit)
├── app_upload.py          # 文档上传页面 (Streamlit)
├── config_data.py         # 全局配置文件 (包含 chunk 大小、模型选择、集合名等)
├── file_history_store.py  # 本地聊天记录存储服务 (JSON持久化)
├── knowledge_base.py      # 知识库核心服务 (文件加载、切片、写入向量库)
├── rag.py                 # RAG 检索问答核心链服务 (Retrieval QA Chain)
├── vector_stores.py       # 向量数据库操作封装服务
├── requirements.txt       # 项目 Python 依赖列表
├── .env                   # 环境变量配置文件 (需手动创建)
├── chroma_db/             # (运行后生成) Chroma 向量数据库文件保存目录
└── chat_history/          # (运行后生成) 根据 session 隔离的聊天记录文件目录
```

## 🛠️ 安装与配置

### 1. 环境准备
推荐使用虚拟环境进行安装（由于最新版依赖库的兼容性，建议使用 Python 3.10 或 3.11）：
```powershell
# 如果使用 uv 工具：
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# 如果使用传统 pip 工具：
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 环境变量配置
在项目根目录 `KnowledgeBase-RAG-LLM-System` 下新建一个 `.env` 文件，如果之前没创建过，需要加入以下内容：

```env
# 阿里云百炼平台的 API Key (必须提供)
DASHSCOPE_API_KEY=sk-xxxx你的真实阿里云密钥xxxx

# API 请求地址
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 彻底关闭 ChromaDB 内部的匿名遥测数据收集 (防止新版本兼容报错)
ANONYMIZED_TELEMETRY=False
```

## 🚀 运行应用

本项目采用前后端分离式运行，提供两个入口。你可以开启两个 PowerShell 终端分别运行：

### 界面一：构建知识库（知识库文档上传）
打开文档上传页面，将你准备好的知识库文档处理写入向量数据库：
```powershell
streamlit run app_upload.py
```

### 界面二：智能客服对话界面
知识库加载完成后，运行聊天应用，向 AI 进行提问测试检索效果：
```powershell
streamlit run app_chat.py
```

## ⚙️ 自定义配置
你可以在 `config_data.py` 文件中修改系统的一些关键配置参数：
- `chunk_size` & `chunk_overlap`: 文本切片控制，调整向量化精度。
- `collection_name`: ChromaDB 里的集合名。
- `embedding_model_name`: 文本嵌入模型（默认 `text-embedding-v4`）。
- `chat_model_name`: 提供对话反馈的大本模型（默认 `qwen3-max`）。
- `similarity_threshold`: 向量库匹配 Top K 数量阈值。
- `session_config`: 修改内部对话缓存的 session ID 取值。