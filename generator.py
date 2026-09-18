"""
generator.py
------------
把检索到的文本块拼成prompt，调用OpenAI Chat接口生成回答，
并要求模型在回答中标注引用来源（论文名 + 页码）。
"""

import base64
import os

from openai import OpenAI

import config

client_openai = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
)

SYSTEM_PROMPT = """你是一个论文知识库问答助手。
请只根据下面提供的【参考文本片段】和【参考图片】回答用户问题，不要编造参考材料之外的信息。
历史对话只用于理解用户的指代和上下文，不能作为事实依据。
如果参考片段不足以回答问题，请明确说"根据现有资料无法回答这个问题"。
回答时请在相关内容后面用 [来源: 文件名, 第N页] 的格式标注引用。
如果提供了参考图片，可以分析其中的图表、结构和视觉信息；图片引用同样使用文件名和页码。
"""

REWRITE_PROMPT = """请结合历史对话，把用户的最新问题改写成一个可以独立用于论文检索的问题。
只输出改写后的问题，不要回答问题，不要补充解释。
如果最新问题本身已经完整，原样输出即可。
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"【参考片段{i}】(来源: {c['source']}, 第{c['page']}页)\n{c['text']}"
        )
    return "\n\n".join(parts)


def _image_to_data_url(image_path: str) -> str:
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(__file__), image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "png"
    mime_ext = {"jpg": "jpeg", "tif": "tiff"}.get(ext, ext)
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/{mime_ext};base64,{encoded}"




def rewrite_query(query: str, history: list[dict] | None = None) -> str:
    """Resolve references in a follow-up question before document retrieval."""
    if not history:
        return query

    history_text = "\n".join(
        f"{('用户' if item['role'] == 'user' else '助手')}：{item['content']}"
        for item in history
    )
    resp = client_openai.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {
                "role": "user",
                "content": f"历史对话：\n{history_text}\n\n最新问题：{query}",
            },
        ],
        temperature=0,
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    return rewritten or query


def generate_answer(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    images: list[dict] | None = None,
) -> str:
    context = build_context(chunks)
    text_context = context or "（没有检索到文本片段）"
    user_prompt = f"参考片段：\n{text_context}\n\n用户问题：{query}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history or []:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append(
                {"role": item["role"], "content": item["content"]}
            )
    valid_images = []
    for image in images or []:
        try:
            valid_images.append((image, _image_to_data_url(image["image_path"])))
        except (OSError, ValueError):
            continue

    if valid_images:
        image_sources = "\n".join(
            f"参考图片{i}: {item['source']}, 第{item['page']}页"
            for i, (item, _) in enumerate(valid_images, start=1)
        )
        content = [
            {
                "type": "text",
                "text": f"{user_prompt}\n\n参考图片来源：\n{image_sources}",
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": data_url}}
            for _, data_url in valid_images
        )
        messages.append({"role": "user", "content": content})
        model = config.VISION_MODEL
    else:
        messages.append({"role": "user", "content": user_prompt})
        model = config.CHAT_MODEL

    resp = client_openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content
