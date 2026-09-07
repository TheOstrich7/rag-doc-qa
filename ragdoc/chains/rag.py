"""RAG 主链路：检索 -> 组装上下文 -> 大模型生成。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from ..vectorstore import format_docs
from .prompts import rag_prompt

logger = logging.getLogger(__name__)

NO_ANSWER = "根据现有资料无法回答该问题。"


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[Document] = field(default_factory=list)
    latency_ms: float = 0.0
    refused: bool = False

    @property
    def context_text(self) -> str:
        return format_docs(self.contexts)

    @property
    def citations(self) -> list[str]:
        out = []
        for d in self.contexts:
            sec = d.metadata.get("section", "")
            page = d.metadata.get("page", "")
            out.append(f"{sec or '未知章节'}"
                       f"{f' 第{int(page) + 1}页' if page != '' else ''}")
        return out


def build_generation_chain(llm: BaseChatModel) -> Runnable:
    """纯生成部分（LCEL）：{context, question} -> str"""
    return rag_prompt() | llm | StrOutputParser()


def rag_answer(
    retriever: BaseRetriever | None,
    llm: BaseChatModel,
    question: str,
    docs: list[Document] | None = None,
) -> RagResult:
    """一次完整的 RAG 问答，返回答案 + 被召回的原文（便于溯源与评测）。

    docs 允许外部传入，评测时可以复用同一次检索结果，避免重复检索。
    """
    t0 = time.perf_counter()
    if docs is None:
        if retriever is None:
            raise ValueError("retriever 与 docs 不能同时为空")
        docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = build_generation_chain(llm)
    answer = chain.invoke({"context": context or "（无参考资料）", "question": question})
    latency = (time.perf_counter() - t0) * 1000

    answer = (answer or "").strip()
    result = RagResult(
        question=question,
        answer=answer,
        contexts=list(docs),
        latency_ms=latency,
        refused=NO_ANSWER in answer,
    )
    logger.debug("RAG 完成：%.0fms，召回 %d 段", latency, len(docs))
    return result
