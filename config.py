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
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl-flash")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", OPENAI_API_KEY)
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "")


# ---- Milvus Lite ----
# Milvus Lite 是嵌入式版本，直接指定一个本地文件路径即可，无需部署服务
MILVUS_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "paper_rag.db")
COLLECTION_NAME = "paper_chunks"
IMAGE_COLLECTION_NAME = "paper_images"

# ---- 文档处理 ----
PDF_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
CHUNK_SIZE = 400        # 每个chunk的目标token数
CHUNK_OVERLAP = 80      # 相邻chunk的重叠token数
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 150
MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_PAGE_RENDER_DPI = int(os.getenv("IMAGE_PAGE_RENDER_DPI", "144"))
RENDER_PAGE_IMAGE_FALLBACK = os.getenv(
    "RENDER_PAGE_IMAGE_FALLBACK", "true"
).lower() in {"1", "true", "yes", "on"}

# ---- 检索 ----
TOP_K = 5
USE_HYBRID_SEARCH = True   # 是否启用向量+BM25混合检索
VECTOR_WEIGHT = 0.7        # 混合检索中向量分数的权重（BM25权重为 1 - VECTOR_WEIGHT）
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() in {
    "1", "true", "yes", "on"
}
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "20"))
IMAGE_EMBEDDING_MODEL = os.getenv("IMAGE_EMBEDDING_MODEL", "qwen3-vl-embedding")
IMAGE_EMBEDDING_DIM = int(os.getenv("IMAGE_EMBEDDING_DIM", "1024"))
IMAGE_TOP_K = int(os.getenv("IMAGE_TOP_K", "2"))
IMAGE_QUERY_KEYWORDS = (
    "图",
    "图片",
    "图表",
    "曲线",
    "柱状图",
    "折线图",
    "散点图",
    "流程图",
    "架构图",
    "示意图",
    "截图",
    "figure",
    "fig.",
    "chart",
    "plot",
    "diagram",
    "image",
)

# ---- 对话 ----
MAX_HISTORY_MESSAGES = 6   # 发送给模型的最近消息数（3轮问答）
CONVERSATION_TITLE_LENGTH = 30
