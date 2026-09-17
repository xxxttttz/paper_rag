"""
generator.py
------------
把检索到的文本块拼成prompt，调用OpenAI Chat接口生成回答，
并要求模型在回答中标注引用来源（论文名 + 页码）。
"""

from openai import OpenAI

import config

client_openai = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
)

SYSTEM_PROMPT = """你是一个论文知识库问答助手。
请只根据下面提供的【参考片段】回答用户问题，不要编造参考片段之外的信息。
历史对话只用于理解用户的指代和上下文，不能作为事实依据。
如果参考片段不足以回答问题，请明确说"根据现有资料无法回答这个问题"。
回答时请在相关内容后面用 [来源: 文件名, 第N页] 的格式标注引用。
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
) -> str:
    context = build_context(chunks)
    user_prompt = f"参考片段：\n{context}\n\n用户问题：{query}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history or []:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append(
                {"role": item["role"], "content": item["content"]}
            )
    messages.append({"role": "user", "content": user_prompt})

    resp = client_openai.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content
