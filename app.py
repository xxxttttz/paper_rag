"""Streamlit interface for the persistent multi-conversation paper RAG app."""

import os

import streamlit as st

import api_client
import chat_service
import config
import database
from milvus_store import collection_exists


st.set_page_config(page_title="论文检索 RAG 助手", page_icon="📄", layout="wide")

if not config.OPENAI_API_KEY:
    st.error("未检测到 OPENAI_API_KEY，请在 .env 文件中配置后重启应用。")
    st.stop()

chat_service.initialize()
os.makedirs(config.PDF_DIR, exist_ok=True)

INGESTION_STAGE_LABELS = {
    "queued": "等待 worker 处理",
    "started": "任务已开始",
    "parsing": "正在解析 PDF",
    "embedding_text": "正在生成文本向量",
    "writing_text": "正在写入文本索引",
    "embedding_images": "正在生成图片向量",
    "completed": "索引构建完成",
}
INGESTION_STAGE_PROGRESS = {
    "queued": 5,
    "started": 10,
    "parsing": 20,
    "embedding_text": 40,
    "writing_text": 65,
    "embedding_images": 80,
    "completed": 100,
}


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
    if citations:
        with st.expander(f"📎 查看 {len(citations)} 个文本片段"):
            for index, citation in enumerate(citations, start=1):
                score = citation.get("score")
                score_text = f" · 相关度: {score:.3f}" if score is not None else ""
                st.markdown(
                    f"**片段 {index}** · 来源: `{citation['source']}` "
                    f"· 第 {citation['page']} 页{score_text}"
                )
                text = citation["chunk_text"]
                st.text(text[:800] + ("..." if len(text) > 800 else ""))

    image_citations = database.get_image_citations(message_id)
    if image_citations:
        with st.expander(f"🖼️ 查看 {len(image_citations)} 个图片引用"):
            for image in image_citations:
                image_path = image["image_path"]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(os.path.dirname(__file__), image_path)
                score = image.get("score")
                score_text = f" · 相关度: {score:.3f}" if score is not None else ""
                st.caption(
                    f"{image['source']} · 第 {image['page']} 页{score_text}"
                )
                if os.path.exists(image_path):
                    st.image(image_path, use_container_width=True)
                else:
                    st.warning("图片文件已不存在，请重新构建知识库索引。")


def delete_active_conversation() -> None:
    conversation_id = st.session_state.get("active_conversation_id")
    if conversation_id:
        database.delete_conversation(conversation_id)
    st.session_state.pop("active_conversation_id", None)
    st.session_state.confirm_delete_conversation = False


def _finish_ingestion_job(job: dict) -> None:
    st.session_state.pop("ingestion_job_id", None)
    st.session_state.ingestion_last_job = job
    st.rerun()


@st.fragment(run_every="2s")
def render_active_ingestion_job(job_id: str) -> None:
    try:
        job = api_client.get_ingestion_job(job_id)
    except api_client.PaperRAGAPIError as error:
        if error.status_code == 404:
            _finish_ingestion_job({"job_id": job_id, "status": "expired"})
            return
        st.warning(f"暂时无法获取索引任务状态：{error}")
        return

    status = job["status"]
    stage = job.get("stage") or status
    stage_label = INGESTION_STAGE_LABELS.get(stage, stage)
    progress = INGESTION_STAGE_PROGRESS.get(stage, 10)

    if status == "failed":
        st.error(f"索引构建失败：{job.get('error') or '未知错误'}")
        _finish_ingestion_job(job)
        return
    if status in {"canceled", "stopped"}:
        st.warning(f"索引任务已{status}")
        _finish_ingestion_job(job)
        return
    if status == "finished":
        _finish_ingestion_job(job)
        return

    st.progress(progress, text=stage_label)
    st.caption(f"任务 ID：{job_id}")


def render_last_ingestion_job(job: dict) -> None:
    status = job.get("status")
    result = job.get("result") or {}
    if status == "finished":
        image_error = result.get("image_error")
        if image_error:
            st.warning(
                f"文本索引已完成（{result.get('text_chunks', 0)} 个文本块），"
                f"但图片索引失败：{image_error}"
            )
        else:
            st.success(
                f"索引构建完成：{result.get('text_chunks', 0)} 个文本块，"
                f"{result.get('images', 0)} 张图片"
            )
    elif status == "failed":
        st.error(f"索引构建失败：{job.get('error') or '未知错误'}")
    elif status == "expired":
        st.warning("任务记录已过期或不存在，可以重新提交构建；此前任务的完成结果无法确认。")
    else:
        st.warning(f"索引任务已结束：{status}")


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

    active_ingestion_job_id = st.session_state.get("ingestion_job_id")
    if st.button(
        "🔨 重新构建知识库索引",
        use_container_width=True,
        disabled=bool(active_ingestion_job_id) or not existing_pdfs,
    ):
        try:
            job = api_client.create_ingestion_job()
            st.session_state.ingestion_job_id = job["job_id"]
            st.session_state.pop("ingestion_last_job", None)
            st.rerun()
        except api_client.PaperRAGAPIError as error:
            detail = error.detail
            if error.status_code == 409 and isinstance(detail, dict):
                st.session_state.ingestion_job_id = detail.get("job_id")
                st.warning("已有索引任务正在运行，已恢复进度跟踪。")
                st.rerun()
            else:
                st.error(str(error))

    active_ingestion_job_id = st.session_state.get("ingestion_job_id")
    if active_ingestion_job_id:
        render_active_ingestion_job(active_ingestion_job_id)
    elif last_job := st.session_state.get("ingestion_last_job"):
        render_last_ingestion_job(last_job)
        if st.button("清除任务状态", use_container_width=True):
            st.session_state.pop("ingestion_last_job", None)
            st.rerun()

    config.USE_HYBRID_SEARCH = st.toggle(
        "启用向量 + BM25 混合检索",
        value=config.USE_HYBRID_SEARCH,
    )
    config.ENABLE_RERANK = st.toggle(
        "启用 Rerank 精排",
        value=config.ENABLE_RERANK,
        help="对粗筛候选进行二次排序，可提高相关性，但会增加少量调用费用。",
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
    try:
        if not collection_exists(config.COLLECTION_NAME):
            st.error("还没有构建知识库索引，请先上传 PDF 并重新构建索引。")
        else:
            with st.spinner("正在理解问题、检索论文并生成回答..."):
                chat_service.ask(active_conversation_id, question)
            st.rerun()
    except Exception as error:
        st.error(f"回答生成失败：{error}")
