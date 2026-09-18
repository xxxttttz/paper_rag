"""
image_ingest.py
----------------
从PDF里抽取图表（不含文字段落），用通义千问的多模态embedding模型
（qwen3-vl-embedding）生成图片向量，写入一个独立的Milvus collection。

跟文本的ingest.py是并行的两条流水线：
    文本: PDF -> md文件-> 分块 -> text-embedding-v4  -> paper_chunks   collection
    图片: PDF -> 抽图 -> qwen3-vl-embedding -> paper_images   collection

之所以要单独一个collection，是因为两者的向量模型不同、维度也可能不同，
Milvus的一个collection只能存同一个维度的向量。

用法：
    python image_ingest.py
"""

import os
import glob
import base64
import hashlib
import uuid

import pymupdf as fitz  # PyMuPDF；用新的导入名避免过期API警告
import dashscope
from pymilvus import MilvusClient, DataType

import config

dashscope.api_key = config.DASHSCOPE_API_KEY
if config.DASHSCOPE_BASE_URL:
    dashscope.base_http_api_url = config.DASHSCOPE_BASE_URL


def extract_images(pdf_path: str):
    """
    用PyMuPDF遍历每一页，抽取内嵌的图片，过滤掉太小的图标/logo。
    返回 [{page, image_bytes, ext}, ...]
    """
    results = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            seen_xrefs = set()
            page_has_image = False
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                base_image = doc.extract_image(xref)
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                image_bytes = base_image["image"]
                if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
                    continue
                if len(image_bytes) > config.MAX_IMAGE_BYTES:
                    print(f"  跳过超过大小限制的图片: 第{page_index + 1}页 xref={xref}")
                    continue
                results.append({
                    "page": page_index + 1,
                    "image_bytes": image_bytes,
                    "ext": base_image.get("ext", "png"),
                })
                page_has_image = True

            # 很多论文图表是 PDF 矢量对象，page.get_images() 无法抽取。
            # 对没有合格位图的页面渲染整页，确保图表仍能进入视觉索引。
            if not page_has_image and config.RENDER_PAGE_IMAGE_FALLBACK:
                scale = config.IMAGE_PAGE_RENDER_DPI / 72
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    alpha=False,
                )
                image_bytes = pixmap.tobytes("png")
                if len(image_bytes) <= config.MAX_IMAGE_BYTES:
                    results.append({
                        "page": page_index + 1,
                        "image_bytes": image_bytes,
                        "ext": "png",
                    })
                else:
                    print(f"  跳过超过大小限制的页面渲染图: 第{page_index + 1}页")
    return results


def save_images_from_pdfs(pdf_dir: str, image_dir: str):
    """
    遍历目录下所有PDF，抽图并保存到本地文件，返回图片记录列表（还不含向量）。
    返回 [{id, image_path, source, page}, ...]
    """
    os.makedirs(image_dir, exist_ok=True)
    records = []
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        print(f"[警告] 在 {pdf_dir} 下没有找到PDF文件。")
        return records

    for pdf_path in pdf_files:
        source = os.path.basename(pdf_path)
        stem = os.path.splitext(source)[0]
        print(f"正在抽图: {source}")
        images = extract_images(pdf_path)
        for i, img in enumerate(images):
            digest = hashlib.sha1(img["image_bytes"]).hexdigest()[:12]
            filename = f"{stem}_p{img['page']}_i{i}_{digest}.{img['ext']}"
            path = os.path.join(image_dir, filename)
            with open(path, "wb") as f:
                f.write(img["image_bytes"])
            records.append({
                "id": str(uuid.uuid4()),
                "image_path": os.path.relpath(path, os.path.dirname(__file__)),
                "source": source,
                "page": img["page"],
            })
    print(f"共抽取 {len(records)} 张图")
    return records


def _image_to_data_url(image_path: str) -> str:
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(__file__), image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "png"
    mime_ext = {"jpg": "jpeg", "tif": "tiff"}.get(ext, ext)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime_ext};base64,{b64}"


def embed_image(image_path: str):
    """调用通义千问多模态embedding，把一张本地图片转成向量"""
    data_url = _image_to_data_url(image_path)
    resp = dashscope.MultiModalEmbedding.call(
        model=config.IMAGE_EMBEDDING_MODEL,
        input=[{"image": data_url}],
        dimension=config.IMAGE_EMBEDDING_DIM,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"图片embedding失败: {resp.code} {resp.message}")
    # 响应结构: resp.output["embeddings"][0]["embedding"]
    return resp.output["embeddings"][0]["embedding"]


def embed_text_query(text: str):
    """把文字问题用同一个多模态模型embedding，落在跟图片同一个向量空间里，
    这样才能用"文字问题"去检索"图片"（跨模态检索的关键）"""
    resp = dashscope.MultiModalEmbedding.call(
        model=config.IMAGE_EMBEDDING_MODEL,
        input=[{"text": text}],
        dimension=config.IMAGE_EMBEDDING_DIM,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"文本embedding失败: {resp.code} {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def build_image_collection(mc: MilvusClient):
    if mc.has_collection(config.IMAGE_COLLECTION_NAME):
        mc.drop_collection(config.IMAGE_COLLECTION_NAME)

    schema = mc.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("image_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("page", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=config.IMAGE_EMBEDDING_DIM)

    index_params = mc.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    mc.create_collection(
        collection_name=config.IMAGE_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"已创建collection: {config.IMAGE_COLLECTION_NAME}")


def rebuild_image_index(mc: MilvusClient | None = None) -> list[dict]:
    """抽取、向量化并重建图片 collection，返回写入的图片记录。"""
    records = save_images_from_pdfs(config.PDF_DIR, config.IMAGE_DIR)
    mc = mc or MilvusClient(uri=config.MILVUS_DB_PATH)
    if not records:
        if mc.has_collection(config.IMAGE_COLLECTION_NAME):
            mc.drop_collection(config.IMAGE_COLLECTION_NAME)
        return []

    print("正在生成图片embedding（逐张调用，数量少可以接受）...")
    for i, r in enumerate(records, start=1):
        r["vector"] = embed_image(r["image_path"])
        print(f"  已完成: {i}/{len(records)}")

    print("正在写入Milvus Lite...")
    build_image_collection(mc)
    mc.insert(collection_name=config.IMAGE_COLLECTION_NAME, data=records)
    print(f"完成！共写入 {len(records)} 张图到 {config.MILVUS_DB_PATH}")
    return records


def main():
    if not config.DASHSCOPE_API_KEY:
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY / OPENAI_API_KEY，请先在 .env 中配置。")
    rebuild_image_index()


if __name__ == "__main__":
    main()
