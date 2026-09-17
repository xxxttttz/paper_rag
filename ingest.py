"""
ingest.py
---------
把 data/pdfs/ 目录下的论文PDF处理成向量，写入Milvus Lite。

流程：
1. 遍历PDF，用pdfplumber逐页提取文本，并记录来源文件名+页码
2. 按 CHUNK_SIZE / CHUNK_OVERLAP 做滑动窗口分块（按token数切，用tiktoken估算）
3. 调用OpenAI embedding接口批量生成向量
4. 写入Milvus Lite的一个collection，字段：id, text, source, page, vector

用法：
    python ingest.py
"""

import os
import glob
import uuid

import pdfplumber
import tiktoken
from openai import OpenAI
from pymilvus import MilvusClient, DataType

import config

encoder = tiktoken.get_encoding("cl100k_base")
client_openai = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
)


def extract_pages(pdf_path: str):
    """逐页提取文本，返回 [(page_no, text), ...]"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int):
    """按token数做滑动窗口切分，返回文本片段列表"""
    tokens = encoder.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(tokens):
        piece = tokens[start:start + chunk_size]
        chunks.append(encoder.decode(piece))
        start += step
    return chunks



def build_chunks_from_pdfs(pdf_dir: str):
    """遍历目录下所有PDF，返回 [{id, text, source, page}, ...]"""
    records = []
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        print(f"[警告] 在 {pdf_dir} 下没有找到PDF文件，请先放入论文。")
        return records

    for pdf_path in pdf_files:
        source = os.path.basename(pdf_path)
        print(f"正在处理: {source}")
        pages = extract_pages(pdf_path)
        for page_no, text in pages:
            for chunk in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                records.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": source,
                    "page": page_no,
                })
    print(f"共生成 {len(records)} 个文本块")
    return records       


def embed_texts(texts: list[str], batch_size: int = 10):
    """批量调用OpenAI embedding接口，返回向量列表"""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client_openai.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
        )
        all_vectors.extend([item.embedding for item in resp.data])
        print(f"  已完成embedding: {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_vectors




def build_collection(mc: MilvusClient):
    """创建（或重建）collection"""
    if mc.has_collection(config.COLLECTION_NAME):
        mc.drop_collection(config.COLLECTION_NAME)

    schema = mc.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=8192)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("page", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)

    index_params = mc.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    mc.create_collection(
        collection_name=config.COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"已创建collection: {config.COLLECTION_NAME}")





def main():
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY，请在 .env 文件或环境变量中设置后再运行。"
        )

    os.makedirs(config.PDF_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MILVUS_DB_PATH), exist_ok=True)

    records = build_chunks_from_pdfs(config.PDF_DIR)
    if not records:
        return

    print("正在生成embedding...")
    vectors = embed_texts([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v

    print("正在写入Milvus Lite...")
    mc = MilvusClient(uri=config.MILVUS_DB_PATH)
    build_collection(mc)
    mc.insert(collection_name=config.COLLECTION_NAME, data=records)
    print(f"完成！共写入 {len(records)} 条数据到 {config.MILVUS_DB_PATH}")




if __name__ == "__main__":
    main()
