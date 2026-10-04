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
import json
import dashscope
from openai import OpenAI
from pymilvus import MilvusClient

import config
import database
from evidence_selector import select_diverse_evidence, select_requirement_evidence
from graph_retriever import enrich_with_graph
from milvus_store import active_collection_name, create_milvus_client
from search_tokenizer import tokenize_search_text

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



def _vector_search(
    mc: MilvusClient, query_vector, top_k: int,
    collection_name: str | None = None, source: str | None = None,
):
    collection_name = collection_name or config.COLLECTION_NAME
    mc.load_collection(collection_name)
    results = mc.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["id", "text", "source", "page"],
        **({"filter": "source == " + json.dumps(source, ensure_ascii=False)} if source is not None else {}),
    )
    hits = results[0]
    chunks = []
    for hit in hits:
        chunk = {
            "text": hit["entity"]["text"],
            "source": hit["entity"]["source"],
            "page": hit["entity"]["page"],
            "score": hit["distance"],  # cosine相似度，越大越相关
        }
        chunk_id = hit["entity"].get("id") or hit.get("id")
        if chunk_id is not None:
            chunk["id"] = chunk_id
        chunks.append(chunk)
    return chunks




def _load_all_chunks(mc: MilvusClient, collection_name: str | None = None):
    """取出全部chunk用于BM25建索引（论文数量小，10-30篇量级完全可以全量加载）"""
    collection_name = collection_name or config.COLLECTION_NAME
    mc.load_collection(collection_name)
    return mc.query(
        collection_name=collection_name,
        filter="",
        output_fields=["id", "text", "source", "page"],
        limit=10000,
    )



def _bm25_search(all_chunks, query: str, top_k: int):
    if not all_chunks:
        return []
    query_tokens = tokenize_search_text(query)
    if not query_tokens:
        return []
    corpus = [tokenize_search_text(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        ((chunk, score) for chunk, score in zip(all_chunks, scores) if score > 0),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]
    return [
        {
            **({"id": c["id"]} if "id" in c else {}),
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


def _record_trace(trace: dict | None, stage: str, hits: list[dict]) -> None:
    """Record only retrieval metadata, never candidate text."""
    if trace is not None:
        trace[stage] = [
            {
                "source": hit["source"], "page": hit["page"],
                "score": float(hit["score"]) if hit["score"] is not None else 0.0,
            }
            for hit in hits
        ]


def _fuse_hits(vector_hits: list[dict], bm25_hits: list[dict], limit: int) -> list[dict]:
    merged = {}
    for hits, weight in (
        (_normalize(vector_hits), config.VECTOR_WEIGHT),
        (_normalize(bm25_hits), 1 - config.VECTOR_WEIGHT),
    ):
        for hit in hits:
            key = hit.get("id") or (hit["source"], hit["page"], hit["text"])
            if key not in merged:
                merged[key] = {
                    "text": hit["text"], "source": hit["source"], "page": hit["page"], "score": 0.0,
                    **({"id": hit["id"]} if hit.get("id") is not None else {}),
                }
            merged[key]["score"] += weight * hit["norm_score"]
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:limit]


class ComparisonIndexError(RuntimeError):
    """A selected document is unavailable in the active text index."""


def retrieve_comparison_evidence(query: str, sources: list[str], per_source_k: int = 5) -> list[dict]:
    """Pin one index, recall within each paper, then rerank once with source quotas."""
    mc = create_milvus_client()
    try:
        if not mc.has_collection(config.COLLECTION_NAME):
            raise ComparisonIndexError("尚未构建文本索引，请先构建知识库。")
        physical = active_collection_name(mc, config.COLLECTION_NAME)
        chunks = _load_all_chunks(mc, physical)
        known = {chunk["source"] for chunk in chunks}
        missing = [source for source in sources if source not in known]
        if missing:
            raise ComparisonIndexError("所选论文不在当前文本索引中：" + "、".join(missing) + "。请重新构建索引。")

        vector = embed_query(query)
        candidate_k = max(per_source_k, config.RERANK_CANDIDATE_K)
        candidates = []
        for source in sources:
            vector_hits = _vector_search(mc, vector, candidate_k, physical, source=source)
            # Defense in depth: never send an unexpected source to the model.
            vector_hits = [hit for hit in vector_hits if hit["source"] == source]
            if config.USE_HYBRID_SEARCH:
                paper_chunks = [chunk for chunk in chunks if chunk["source"] == source]
                bm25_hits = _bm25_search(paper_chunks, query, candidate_k)
                candidates.extend(_fuse_hits(vector_hits, bm25_hits, candidate_k))
            else:
                candidates.extend(vector_hits[:candidate_k])
        ranked = rerank(query, candidates, len(candidates)) if config.ENABLE_RERANK and candidates else candidates
        selected = []
        for source in sources:
            selected.extend(select_diverse_evidence(
                [hit for hit in ranked if hit["source"] == source], per_source_k,
            ))
        return selected
    finally:
        mc.close()

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

def retrieve(
    query: str,
    top_k: int = None,
    trace: dict | None = None,
    *,
    reranked_output: list[dict] | None = None,
):
    """
    对外的统一检索入口。
    返回按相关度排序的 [{text, source, page, score}, ...]
    reranked_output 仅供本进程内的配对评测捕获完整排名；trace 仍不含原文。
    """
    mc = create_milvus_client()
    try:
        graph_enabled = config.ENABLE_GRAPH_RETRIEVAL and bool(database.DATABASE_URL)
        physical_name = (
            active_collection_name(mc, config.COLLECTION_NAME) if graph_enabled else None
        )
        result = _retrieve_with_client(
            mc, query, top_k, trace=trace, collection_name=physical_name,
            reranked_output=reranked_output,
        )
        if graph_enabled:
            try:
                result, graph_hits = enrich_with_graph(
                    mc, database.DATABASE_URL, physical_name, query, result,
                    top_k or config.TOP_K, config.GRAPH_EVIDENCE_K,
                )
                if trace is not None:
                    trace.pop("final", None)
                _record_trace(trace, "graph", graph_hits)
                _record_trace(trace, "final", result)
            except Exception as error:
                # The graph is an optional enhancement; text retrieval remains usable.
                print(f"[警告] 图检索失败，保留文本检索结果: {error}")
        return result
    finally:
        mc.close()


def _retrieve_with_client(
    mc: MilvusClient,
    query: str,
    top_k: int | None = None,
    trace: dict | None = None,
    collection_name: str | None = None,
    reranked_output: list[dict] | None = None,
):
    top_k = top_k or config.TOP_K
    candidate_k = (
        max(top_k, config.RERANK_CANDIDATE_K)
        if config.ENABLE_RERANK
        else top_k
    )
    query_vector = embed_query(query)
    if collection_name is None:
        vector_hits = _vector_search(mc, query_vector, top_k=candidate_k)
    else:
        vector_hits = _vector_search(mc, query_vector, top_k=candidate_k, collection_name=collection_name)
    _record_trace(trace, "vector", vector_hits)

    if not config.USE_HYBRID_SEARCH:
        candidates = vector_hits[:candidate_k]
        _record_trace(trace, "candidates", candidates)
        if config.ENABLE_RERANK and candidates:
            try:
                rerank_n = (
                    len(candidates)
                    if config.ENABLE_DIVERSE_RERANK or config.ENABLE_REQUIREMENT_SELECTION
                    or reranked_output is not None
                    else top_k
                )
                reranked = rerank(query, candidates, top_n=rerank_n)
                if reranked_output is not None:
                    reranked_output.extend(reranked)
                    _record_trace(trace, "reranked", reranked)
                if config.ENABLE_REQUIREMENT_SELECTION:
                    _record_trace(trace, "reranked", reranked)
                    final = select_requirement_evidence(query, reranked, top_k)
                elif config.ENABLE_DIVERSE_RERANK:
                    _record_trace(trace, "reranked", reranked)
                    final = select_diverse_evidence(reranked, top_k)
                else:
                    final = reranked[:top_k]
                _record_trace(trace, "final", final)
                return final
            except Exception as error:
                if reranked_output is not None:
                    raise
                print(f"[警告] Rerank 失败，回退到向量排序: {error}")
        final = candidates[:top_k]
        _record_trace(trace, "final", final)
        return final

    if collection_name is None:
        all_chunks = _load_all_chunks(mc)
    else:
        all_chunks = _load_all_chunks(mc, collection_name=collection_name)
    bm25_hits = _bm25_search(all_chunks, query, top_k=candidate_k)
    _record_trace(trace, "bm25", bm25_hits)

    candidates = _fuse_hits(vector_hits, bm25_hits, candidate_k)
    _record_trace(trace, "candidates", candidates)

    if config.ENABLE_RERANK and candidates:
        try:
            rerank_n = (
                len(candidates)
                if config.ENABLE_DIVERSE_RERANK or config.ENABLE_REQUIREMENT_SELECTION
                or reranked_output is not None
                else top_k
            )
            reranked = rerank(query, candidates, top_n=rerank_n)
            if reranked_output is not None:
                reranked_output.extend(reranked)
                _record_trace(trace, "reranked", reranked)
            if config.ENABLE_REQUIREMENT_SELECTION:
                _record_trace(trace, "reranked", reranked)
                final = select_requirement_evidence(query, reranked, top_k)
            elif config.ENABLE_DIVERSE_RERANK:
                _record_trace(trace, "reranked", reranked)
                final = select_diverse_evidence(reranked, top_k)
            else:
                final = reranked[:top_k]
            _record_trace(trace, "final", final)
            return final
        except Exception as error:
            if reranked_output is not None:
                raise
            print(f"[警告] Rerank 失败，回退到混合排序: {error}")

    final = candidates[:top_k]
    _record_trace(trace, "final", final)
    return final





