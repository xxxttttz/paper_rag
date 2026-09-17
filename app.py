"""Streamlit interface for the persistent multi-conversation paper RAG app."""

import os

import streamlit as st
from pymilvus import MilvusClient

import chat_service
import config
import database
from ingest import build_chunks_from_pdfs, build_collection, embed_texts


st.set_page_config(page_title="论文检索 RAG 助手", page_icon="📄", layout="wide")

if not config.OPENAI_API_KEY:
    st.error("未检测到 OPENAI_API_KEY，请在 .env 文件中配置后重启应用。")
    st.stop()

chat_service.initialize()
os.makedirs(config.PDF_DIR, exist_ok=True)


def ensure_active_conversation() -> str:
    conversations = database.list_conversations()
    known_ids = {item["id"] for item in conversations}
    active_id = st.session_state.get("active_conversation_id")
    if active_id not in known_ids:
        active_id = conversations[0]["id"] if conversations else chat_service.create_conversation()
        st.session_state.active_conversation_id = active_id
    return active_id


def render_citations(message_id: str) -> None:
    citations = database.get_citations(message_id)
    if not citations:
        return
    with st.expander(f"📎 查看 {len(citations)} 个参考片段"):
        for index, citation in enumerate(citations, start=1):
            score = citation.get("score")
            score_text = f" · 相关度: {score:.3f}" if score is not None else ""
            st.markdown(
                f"**片段 {index}** · 来源: `{citation['source']}` "
                f"· 第 {citation['page']} 页{score_text}"
            )
            text = citation["chunk_text"]
            st.text(text[:800] + ("..." if len(text) > 800 else ""))


def delete_active_conversation() -> None:
    conversation_id = st.session_state.get("active_conversation_id")
    if conversation_id:
        database.delete_conversation(conversation_id)
    st.session_state.pop("active_conversation_id", None)
    st.session_state.confirm_delete_conversation = False


active_conversation_id = ensure_active_conversation()

with st.sidebar:
    st.header("💬 对话记录")
    if st.button("＋ 新建会话", use_container_width=True, type="primary"):
        st.session_state.active_conversation_id = chat_service.create_conversation()
        st.rerun()

    for conversation in database.list_conversations():
        selected = conversation["id"] == active_conversation_id
        label = ("● " if selected else "") + conversation["title"]
        if st.button(
            label,
            key=f"conversation_{conversation['id']}",
            use_container_width=True,
            disabled=selected,
        ):
            st.session_state.active_conversation_id = conversation["id"]
            st.rerun()

    st.checkbox("确认删除当前会话", key="confirm_delete_conversation")
    st.button(
        "删除当前会话",
        use_container_width=True,
        disabled=not st.session_state.confirm_delete_conversation,
        on_click=delete_active_conversation,
    )

    st.divider()
    st.header("📚 知识库管理")
    uploaded_files = st.file_uploader(
        "上传 PDF 论文（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        saved_count = 0
        for uploaded_file in uploaded_files:
            save_path = os.path.join(config.PDF_DIR, os.path.basename(uploaded_file.name))
            with open(save_path, "wb") as output:
                output.write(uploaded_file.getbuffer())
            saved_count += 1
        st.success(f"已保存 {saved_count} 个文件")

    existing_pdfs = sorted(
        filename
        for filename in os.listdir(config.PDF_DIR)
        if filename.lower().endswith(".pdf")
    )
    st.caption(f"当前知识库：{len(existing_pdfs)} 篇 PDF")
    for filename in existing_pdfs:
        st.caption(f"• {filename}")

    if st.button("🔨 重新构建知识库索引", use_container_width=True):
        try:
            with st.spinner("正在解析 PDF、生成向量并写入 Milvus..."):
                records = build_chunks_from_pdfs(config.PDF_DIR)
                if not records:
                    st.warning("知识库为空，请先上传 PDF。")
                else:
                    vectors = embed_texts([record["text"] for record in records])
                    for record, vector in zip(records, vectors):
                        record["vector"] = vector
                    milvus = MilvusClient(uri=config.MILVUS_DB_PATH)
                    build_collection(milvus)
                    milvus.insert(collection_name=config.COLLECTION_NAME, data=records)
                    st.success(f"索引构建完成，共 {len(records)} 个文本块")
        except Exception as error:
            st.error(f"索引构建失败：{error}")

    config.USE_HYBRID_SEARCH = st.toggle(
        "启用向量 + BM25 混合检索",
        value=config.USE_HYBRID_SEARCH,
    )


conversation = database.get_conversation(active_conversation_id)
st.title("📄 论文检索 RAG 助手")
st.caption(f"当前会话：{conversation['title']} · 回答仅依据知识库中的论文片段")

messages = database.get_messages(active_conversation_id)
if not messages:
    st.info("上传并构建论文索引后，在下方输入问题开始对话。")

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_citations(message["id"])

question = st.chat_input("询问论文内容……")
if question:
    if not os.path.exists(config.MILVUS_DB_PATH):
        st.error("还没有构建知识库索引，请先上传 PDF 并重新构建索引。")
    else:
        try:
            with st.spinner("正在理解问题、检索论文并生成回答..."):
                chat_service.ask(active_conversation_id, question)
            st.rerun()
        except Exception as error:
            st.error(f"回答生成失败：{error}")
