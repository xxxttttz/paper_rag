"""
image_retriever.py
-------------------
用用户的文字问题，去 paper_images collection 里检索最相关的图表。
检索原理：文字和图片都经过同一个 qwen3-vl-embedding 模型，
落在同一个向量空间里，所以"文字问题"和"图片"之间的cosine相似度是有意义的——这就是跨模态检索（cross-modal retrieval）的核心。
"""

from pymilvus import MilvusClient

import config
from image_ingest import embed_text_query


def retrieve_images(query: str, top_k: int = None):
    """
    返回 [{image_path, source, page, score}, ...]，按相关度排序。
    如果collection还没建过（用户没跑过image_ingest），返回空列表。
    """
    top_k = top_k or config.IMAGE_TOP_K
    mc = MilvusClient(uri=config.MILVUS_DB_PATH)

    if not mc.has_collection(config.IMAGE_COLLECTION_NAME):
        return []

    query_vector = embed_text_query(query)
    results = mc.search(
        collection_name=config.IMAGE_COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["image_path", "source", "page"],
    )
    hits = results[0]
    return [
        {
            "image_path": h["entity"]["image_path"],
            "source": h["entity"]["source"],
            "page": h["entity"]["page"],
            "score": h["distance"],
        }
        for h in hits
    ]