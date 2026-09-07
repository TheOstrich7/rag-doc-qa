"""评测指标。

分两类，避免"只靠大模型自评"这种不可复现的做法：
  1) 确定性指标：检索召回率@k、MRR、关键词覆盖率、拒答率、延迟 —— 纯字符串比对，可复现
  2) LLM-as-Judge：答案正确率、忠实度（幻觉检测） —— 用 DeepSeek 打分，输出结构化 JSON
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from ..chains.prompts import correctness_prompt, faithfulness_prompt

logger = logging.getLogger(__name__)

REFUSAL_MARKERS = [
    "无法回答",
    "未找到",
    "没有找到",
    "没有相关",
    "未提及",
    "资料中无法确定",
    "不确定",
    "不知道",
]


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS)


# ------------------------------------------------------------ 确定性指标
def retrieval_hit(docs: list[Document], gold_evidence: str | None) -> bool:
    """召回的 chunk 中是否包含标准答案所在的原文片段。"""
    if not gold_evidence:
        return False
    return any(gold_evidence in d.page_content for d in docs)


def reciprocal_rank(docs: list[Document], gold_evidence: str | None) -> float:
    if not gold_evidence:
        return 0.0
    for i, d in enumerate(docs, 1):
        if gold_evidence in d.page_content:
            return 1.0 / i
    return 0.0


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """关键点覆盖率：答案里出现了多少个必须出现的数字/术语。"""
    if not keywords:
        return float("nan")
    hit = sum(1 for k in keywords if k.lower() in answer.lower())
    return hit / len(keywords)


# ------------------------------------------------------------ LLM-as-Judge
def _parse_json_obj(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class JudgeScore:
    score: float  # 0 ~ 1
    reason: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.score >= 0.99


def _judge(chain_prompt, llm: BaseChatModel, **kwargs) -> JudgeScore:
    """调用 LLM 判分；任何异常都安全降级到 0 分，不影响主流程。"""
    try:
        chain = chain_prompt | llm | StrOutputParser()
        raw = chain.invoke(kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge 调用失败，降级为 0 分：%s", exc)
        return JudgeScore(score=0.0, reason=f"judge-error: {type(exc).__name__}", raw="")
    obj = _parse_json_obj(raw) or {}
    try:
        s = int(obj.get("score", 0))
    except (TypeError, ValueError):
        s = 0
    s = max(0, min(2, s))
    return JudgeScore(score=s / 2.0, reason=str(obj.get("reason", "")), raw=raw)


def judge_correctness(
    llm: BaseChatModel, question: str, reference: str, answer: str
) -> JudgeScore:
    return _judge(correctness_prompt(), llm, question=question, reference=reference, answer=answer)


def judge_faithfulness(llm: BaseChatModel, context: str, answer: str) -> JudgeScore:
    return _judge(faithfulness_prompt(), llm, context=context, answer=answer)


# ------------------------------------------------------------ 聚合
def mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]  # 过滤 NaN
    return sum(vals) / len(vals) if vals else float("nan")


def pct(x: float) -> str:
    return "-" if x != x else f"{x * 100:.1f}%"
