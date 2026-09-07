"""Streamlit 极简 Web 界面：上传 PDF → 建索引 → 聊天。

启动：
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 让 Streamlit 在任意 cwd 下都能 import ragdoc
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ragdoc.config import settings  # noqa: E402
from ragdoc.embeddings import describe_backend, get_embeddings  # noqa: E402
from ragdoc.loaders import SplitConfig, load_and_split  # noqa: E402
from ragdoc.llm import get_chat_llm  # noqa: E402
from ragdoc.vectorstore import build_index, build_retriever, load_index, save_index  # noqa: E402

st.set_page_config(page_title="星海 RAG 问答", page_icon="📚", layout="wide")
st.title("📚 星海科技内部知识库问答")

with st.sidebar:
    st.header("1. 数据")
    uploaded = st.file_uploader("上传 PDF（不上传就用现有索引）", type=["pdf"])
    use_existing = st.button("使用已有索引")
    st.divider()
    st.header("2. 切分参数")
    chunk_size = st.slider("chunk_size", 200, 1200, settings.chunk_size, 50)
    chunk_overlap = st.slider("chunk_overlap", 0, 300, settings.chunk_overlap, 10)
    st.divider()
    st.header("3. 检索参数")
    k = st.slider("top_k", 1, 10, settings.top_k)
    search_type = st.selectbox("search_type", ["mmr", "similarity"], 0)
    st.divider()
    st.caption(f"Embedding：{settings.embedding_backend} / {settings.embedding_model}")
    st.caption(f"LLM：{settings.llm_model}")


@st.cache_resource(show_spinner=False)
def build_index_from_pdf(pdf_bytes: bytes, c_size: int, c_overlap: int) -> str:
    """建索引并保存到 settings.index_dir，返回 chunk 数。"""
    pdf_path = settings.data_dir / "uploaded.pdf"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)
    docs = load_and_split(pdf_path, SplitConfig(c_size, c_overlap))
    embeddings = get_embeddings(settings)
    vs = build_index(docs, embeddings)
    save_index(vs, settings.index_dir)
    return len(docs), describe_backend(embeddings)


@st.cache_resource(show_spinner=False)
def load_existing_index():
    embeddings = get_embeddings(settings)
    return load_index(settings.index_dir, embeddings)


def get_retriever(c_size: int, c_overlap: int, k_: int, stype: str):
    if uploaded is not None:
        with st.spinner("建索引中..."):
            n_chunks, backend = build_index_from_pdf(uploaded.read(), c_size, c_overlap)
        st.success(f"已建索引：{n_chunks} 个 chunk（{backend}）")
    elif use_existing:
        st.info("已加载现有索引。")
    vs = load_existing_index()
    return build_retriever(vs, top_k=k_, search_type=stype,
                           fetch_k=settings.fetch_k, lambda_mult=settings.lambda_mult)


if "history" not in st.session_state:
    st.session_state.history = []

try:
    retriever = get_retriever(chunk_size, chunk_overlap, k, search_type)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if not settings.has_llm_key:
    st.error(
        "未配置 DEEPSEEK_API_KEY。请在项目根目录创建 .env 文件：\n\n"
        "```\nDEEPSEEK_API_KEY=sk-xxx\n```\n"
        "然后刷新页面。"
    )
    st.stop()

llm = get_chat_llm(settings)

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander(f"参考资料（{len(turn['sources'])} 段）"):
                for i, s in enumerate(turn["sources"], 1):
                    st.caption(s)

q = st.chat_input("向手册提问，例如：去深圳出差住宿标准是多少？")
if q:
    from ragdoc.chains.rag import rag_answer  # noqa: PLC0415

    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            res = rag_answer(retriever, llm, q)
        st.markdown(res.answer)
        st.caption(f"耗时 {res.latency_ms:.0f}ms · 召回 {len(res.contexts)} 段")
        if res.contexts:
            with st.expander("参考资料"):
                for i, d in enumerate(res.contexts, 1):
                    sec = d.metadata.get("section", "")
                    st.markdown(f"**[{i}] {sec}**")
                    st.caption(d.page_content)
    st.session_state.history.append({"role": "user", "content": q})
    st.session_state.history.append({
        "role": "assistant",
        "content": res.answer,
        "sources": [d.page_content for d in res.contexts],
    })
