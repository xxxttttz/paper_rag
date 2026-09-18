"""
retriever.py
------------
负责从Milvus Lite里检索与用户问题最相关的文本块。

1. 粗筛（召回）：向量检索 + BM25关键词检索各取一部分候选，按加权分数合并
   （BM25能补上向量检索容易漏掉的专有名词/缩写匹配，例如论文里的模型名、协议名）
2. 精排（可选，由 config.ENABLE_RERANK 控制）：
   粗筛阶段的向量/BM25分数都是"分别编码再比较"（bi-encoder），速度快但不够精准；
   精排用cross-encoder重排序模型（问题和候选文本一起输入模型），对粗筛出的候选
   重新打分排序，只取最终TopK喂给生成模型。
"""

from rank_bm25 import BM25Okapi
import dashscope
from openai import OpenAI
from pymilvus import MilvusClient

import config

client_openai = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
)
dashscope.api_key = config.DASHSCOPE_API_KEY
if config.DASHSCOPE_BASE_URL:
    dashscope.base_http_api_url = config.DASHSCOPE_BASE_URL


def embed_query(query: str):
    resp = client_openai.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[query],
    )
    return resp.data[0].embedding



def _vector_search(mc: MilvusClient, query_vector, top_k: int):
    results = mc.search(
        collection_name=config.COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "source", "page"],
    )
    hits = results[0]
    return [
        {
            "text": h["entity"]["text"],
            "source": h["entity"]["source"],
            "page": h["entity"]["page"],
            "score": h["distance"],  # cosine相似度，越大越相关
        }
        for h in hits
    ]




def _load_all_chunks(mc: MilvusClient):
    """取出全部chunk用于BM25建索引（论文数量小，10-30篇量级完全可以全量加载）"""
    return mc.query(
        collection_name=config.COLLECTION_NAME,
        filter="",
        output_fields=["id", "text", "source", "page"],
        limit=10000,
    )



def _bm25_search(all_chunks, query: str, top_k: int):
    corpus = [c["text"].split() for c in all_chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(all_chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "text": c["text"],
            "source": c["source"],
            "page": c["page"],
            "score": float(score),
        }
        for c, score in ranked
    ]


def _normalize(items, key="score"):
    """把一组分数线性归一化到 [0, 1]，方便和另一路分数加权融合"""
    if not items:
        return items
    scores = [it[key] for it in items]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi > lo else 1.0
    for it in items:
        it["norm_score"] = (it[key] - lo) / span
    return items

def rerank(query: str, candidates: list[dict], top_n: int):
    """Use a cross-encoder to reorder recalled text candidates."""
    if not candidates:
        return candidates

    top_n = min(max(top_n, 1), len(candidates))
    documents = [c["text"] for c in candidates]
    resp = dashscope.TextReRank.call(
        model=config.RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,  # 不需要模型把原文吐回来，我们自己按index对应即可
        api_key=config.DASHSCOPE_API_KEY,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"重排序失败: {resp.code} {resp.message}")

    reranked = []
    for item in resp.output["results"]:
        index = item["index"]
        if not 0 <= index < len(candidates):
            continue
        original = candidates[index]
        reranked.append({
            **original,
            "retrieval_score": original["score"],
            "score": float(item["relevance_score"]),
        })
    return reranked

def retrieve(query: str, top_k: int = None):
    """
    对外的统一检索入口。
    返回按相关度排序的 [{text, source, page, score}, ...]
    """
    top_k = top_k or config.TOP_K
    candidate_k = (
        max(top_k, config.RERANK_CANDIDATE_K)
        if config.ENABLE_RERANK
        else top_k
    )
    mc = MilvusClient(uri=config.MILVUS_DB_PATH)

    query_vector = embed_query(query)
    vector_hits = _vector_search(mc, query_vector, top_k=candidate_k)

    if not config.USE_HYBRID_SEARCH:
        candidates = vector_hits[:candidate_k]
        if config.ENABLE_RERANK and candidates:
            try:
                return rerank(query, candidates, top_n=top_k)
            except Exception as error:
                print(f"[警告] Rerank 失败，回退到向量排序: {error}")
        return candidates[:top_k]

    all_chunks = _load_all_chunks(mc)
    bm25_hits = _bm25_search(all_chunks, query, top_k=candidate_k)

    vector_hits = _normalize(vector_hits)
    bm25_hits = _normalize(bm25_hits)

    # 按 (source, page, text) 做key合并两路结果
    merged = {}
    for h in vector_hits:
        key = (h["source"], h["page"], h["text"])
        merged[key] = merged.get(key, 0) + config.VECTOR_WEIGHT * h["norm_score"]
    for h in bm25_hits:
        key = (h["source"], h["page"], h["text"])
        merged[key] = merged.get(key, 0) + (1 - config.VECTOR_WEIGHT) * h["norm_score"]

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:candidate_k]
    candidates = [
        {"source": k[0], "page": k[1], "text": k[2], "score": v}
        for k, v in ranked
    ]

    if config.ENABLE_RERANK and candidates:
        try:
            return rerank(query, candidates, top_n=top_k)
        except Exception as error:
            print(f"[警告] Rerank 失败，回退到混合排序: {error}")

    return candidates[:top_k]





