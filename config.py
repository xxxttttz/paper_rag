"""
全局配置。
所有可调参数集中在这里，方便后续做混合检索/换模型时统一修改。
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---- OpenAI ----
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# EMBEDDING_MODEL = "text-embedding-3-small"
# EMBEDDING_DIM = 1536  # text-embedding-3-small 的输出维度
# CHAT_MODEL = "gpt-4o-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
) or None  # 留空环境变量则不走任何自定义地址，回退到OpenAI官方
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # 换模型时必须同步改这个，否则Milvus建表维度会对不上
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")


# ---- Milvus Lite ----
# Milvus Lite 是嵌入式版本，直接指定一个本地文件路径即可，无需部署服务
MILVUS_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "paper_rag.db")
COLLECTION_NAME = "paper_chunks"

# ---- 文档处理 ----
PDF_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
CHUNK_SIZE = 400        # 每个chunk的目标token数
CHUNK_OVERLAP = 80      # 相邻chunk的重叠token数

# ---- 检索 ----
TOP_K = 5
USE_HYBRID_SEARCH = True   # 是否启用向量+BM25混合检索
VECTOR_WEIGHT = 0.7        # 混合检索中向量分数的权重（BM25权重为 1 - VECTOR_WEIGHT）

# ---- 对话 ----
MAX_HISTORY_MESSAGES = 6   # 发送给模型的最近消息数（3轮问答）
CONVERSATION_TITLE_LENGTH = 30
