"""FAISS 向量库的构建 / 落盘 / 加载 / 检索。"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
STORE_FILE = "index.pkl"


def build_index(docs: list[Document], embeddings: Embeddings) -> FAISS:
    if not docs:
        raise ValueError("文档列表为空，无法建索引")
    vs = FAISS.from_documents(docs, embeddings)
    logger.info("FAISS 索引构建完成：%d 个向量", len(docs))
    return vs


def save_index(vs: FAISS, index_dir: str | Path) -> Path:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    logger.info("索引已保存到 %s", index_dir)
    return index_dir


def load_index(index_dir: str | Path, embeddings: Embeddings) -> FAISS:
    index_dir = Path(index_dir)
    if not (index_dir / INDEX_FILE).exists():
        raise FileNotFoundError(
            f"索引不存在：{index_dir}。请先运行：python -m ragdoc.cli ingest --pdf <你的PDF>"
        )
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )


def build_retriever(
    vs: FAISS,
    top_k: int = 4,
    search_type: str = "mmr",
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> VectorStoreRetriever:
    search_kwargs: dict = {"k": top_k}
    if search_type == "mmr":
        search_kwargs.update({"fetch_k": max(fetch_k, top_k), "lambda_mult": lambda_mult})
    return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def retrieve_with_score(vs: FAISS, query: str, k: int = 4) -> list[tuple[Document, float]]:
    """带相似度分数的检索，用于调试与可解释性。"""
    return vs.similarity_search_with_score(query, k=k)


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        section = d.metadata.get("section", "")
        page = d.metadata.get("page", "")
        head = f"[{i}]" + (f" {section}" if section else "")
        if page != "":
            head += f" (第{int(page) + 1}页)"
        parts.append(f"{head}\n{d.page_content}")
    return "\n\n".join(parts)


__all__ = [
    "FAISS",
    "build_index",
    "save_index",
    "load_index",
    "build_retriever",
    "retrieve_with_score",
    "format_docs",
    "BaseRetriever",
]
