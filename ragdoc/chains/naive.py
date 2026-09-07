"""基线链路：不给任何参考资料，直接问大模型（第[2]步的对照组）。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .prompts import naive_prompt


@dataclass
class NaiveResult:
    question: str
    answer: str
    latency_ms: float = 0.0


def naive_answer(llm: BaseChatModel, question: str) -> NaiveResult:
    t0 = time.perf_counter()
    chain = naive_prompt() | llm | StrOutputParser()
    answer = (chain.invoke({"question": question}) or "").strip()
    return NaiveResult(question=question, answer=answer, latency_ms=(time.perf_counter() - t0) * 1000)
