"""
配置文件 - API keys 和常量定义
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============ 搜索配置 ============
TOP_K_RESULTS = 5  # 返回多个搜索结果供用户选择

# ============ LLM 配置（OpenAI） ============
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ============ Embedding 配置（阿里云 DashScope） ============
EMBEDDING_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_USE_OPENAI_API = os.getenv("EMBEDDING_USE_OPENAI_API", "true").lower() == "true"

# ============ Tavily 搜索配置 ============
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ============ 向量数据库配置 ============
CHROMA_PERSIST_DIR = "data/knowledge_base"
COLLECTION_NAME = "learning_documents"

# ============ 文档存储配置 ============
DOCUMENTS_DIR = "data/documents"
KB_SUMMARY_FILE = "data/kb_summary.txt"

# ============ 检索配置 ============
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# ============ RRF 融合配置 ============
RRF_K = 60
