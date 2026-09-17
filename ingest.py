"""
ingest.py
---------


流程：
1. 用PyMuPDF4LLM逐页把PDF转成结构化Markdown——正确处理双栏排版的阅读顺序，
   并把表格识别成规范的Markdown表格语法（而不是打散成一堆散乱文字）
2. 按"段落/表格"这样的结构边界做分块（结构感知分块），而不是无脑按token数滑窗切：
   - 表格块无论多大，永远整体放进同一个chunk，绝不会被从中间切断
   - 普通文本块如果单独就超过CHUNK_SIZE，才退化成旧的token滑窗切分
3. 调用通义千问embedding接口批量生成向量
4. 写入Milvus Lite的一个collection，字段：id, text, source, page, vector

用法：
    python ingest.py
"""

import os
import glob
import re
import uuid

import pymupdf4llm
import tiktoken
from openai import OpenAI
from pymilvus import MilvusClient, DataType

import config

encoder = tiktoken.get_encoding("cl100k_base")
client_openai = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)


def extract_pages_markdown(pdf_path: str):
    """
    用PyMuPDF4LLM逐页转成Markdown。

    """
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    return [
        (p["metadata"]["page_number"], p["text"])
        for p in pages
        if p["text"].strip()
    ]


def split_into_blocks(md_text: str):
    """
    按空行把Markdown切成段落级"块"。
    关键点：Markdown表格内部各行之间没有空行，所以一个表格天然会被当成
    一个完整的块保留下来，不会在这一步就被拆散。
    """
    raw_blocks = re.split(r"\n\s*\n", md_text)
    return [b.strip() for b in raw_blocks if b.strip()]


def is_table_block(block: str) -> bool:
    """判断一个块是不是Markdown表格：至少有2行以 `|` 开头"""
    lines = [l for l in block.splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    return sum(1 for l in lines if l.strip().startswith("|")) >= 2


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


def split_table_block(table: str, chunk_size: int) -> list[str]:
    """按行拆分超长 Markdown 表格，并在每个分片中重复表头。"""
    lines = [line for line in table.splitlines() if line.strip()]
    if len(lines) < 3:
        return chunk_text(table, chunk_size, 0)

    header = lines[:2]
    rows = lines[2:]
    header_text = "\n".join(header)
    if len(encoder.encode(header_text)) >= chunk_size:
        return chunk_text(table, chunk_size, 0)

    chunks = []
    current_rows = []
    for row in rows:
        candidate = "\n".join(header + current_rows + [row])
        if len(encoder.encode(candidate)) <= chunk_size:
            current_rows.append(row)
            continue

        if current_rows:
            chunks.append("\n".join(header + current_rows))
            current_rows = []

        single_row = "\n".join(header + [row])
        if len(encoder.encode(single_row)) <= chunk_size:
            current_rows.append(row)
        else:
            # 极端情况下单行本身超长，只能退回 Token 切分以满足模型和数据库限制。
            chunks.extend(chunk_text(single_row, chunk_size, 0))

    if current_rows:
        chunks.append("\n".join(header + current_rows))
    return chunks


def chunk_markdown(md_text: str, chunk_size: int, overlap: int) -> list[str]:
    """按 Markdown 结构合并段落，并只在块过长时执行安全拆分。"""
    chunks = []
    current_blocks = []

    def flush_current() -> None:
        if current_blocks:
            chunks.append("\n\n".join(current_blocks))
            current_blocks.clear()

    for block in split_into_blocks(md_text):
        block_tokens = len(encoder.encode(block))
        if block_tokens > chunk_size:
            flush_current()
            if is_table_block(block):
                chunks.extend(split_table_block(block, chunk_size))
            else:
                chunks.extend(chunk_text(block, chunk_size, overlap))
            continue

        candidate = "\n\n".join(current_blocks + [block])
        if current_blocks and len(encoder.encode(candidate)) > chunk_size:
            flush_current()
        current_blocks.append(block)

    flush_current()
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
        pages = extract_pages_markdown(pdf_path)
        for page_no, text in pages:
            for chunk in chunk_markdown(
                text,
                config.CHUNK_SIZE,
                config.CHUNK_OVERLAP,
            ):
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
