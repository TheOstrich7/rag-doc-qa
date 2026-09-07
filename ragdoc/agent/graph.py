"""ReAct Agent：由大模型自己决定是否调用工具、调用哪个工具。

与固定 RAG 链路的区别：
  RAG 链路  = 无论什么问题，都先检索再回答
  Agent     = 先判断"这个问题需不需要查"，再决定查知识库 / 算数 / 查时间 / 直接回答
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from .tools import build_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是「星海科技」的智能助理，可以调用工具来帮助用户。

决策原则：
1. 问题涉及公司**具体制度、金额、天数、时限、故障等级、审批权限**等事实性内容时，
   必须先调用 search_knowledge_base 检索知识库，并**只依据检索结果**作答，不得编造。
2. 检索结果里没有答案时，直接回答"根据现有资料无法回答该问题。"，不要用常识补。
3. 需要算数时调用 calculator；需要当前日期/时间时调用 get_current_datetime。
4. 纯粹的闲聊、概念解释、对已有信息的改写，**直接回答**，不要调用任何工具。

回答时说明你依据了什么；如果调用了工具，用一句话说明用了哪个工具。"""


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[str] = field(default_factory=list)
    used_knowledge_base: bool = False
    steps: list[str] = field(default_factory=list)
    error: str | None = None


def _create_react_agent(llm: BaseChatModel, tools: list[BaseTool]) -> Any:
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415

    sig = inspect.signature(create_react_agent)
    if "prompt" in sig.parameters:
        return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    if "state_modifier" in sig.parameters:
        return create_react_agent(llm, tools, state_modifier=SYSTEM_PROMPT)
    return create_react_agent(llm, tools)


def build_agent(llm: BaseChatModel, retriever: Any | None = None, top_k: int = 4) -> Any:
    tools = build_tools(retriever, top_k=top_k)
    return _create_react_agent(llm, tools)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content) if content is not None else ""


def run_agent(agent: Any, question: str) -> AgentResult:
    """执行 Agent 并抽取「用了哪些工具」的可解释轨迹。"""
    result = AgentResult(question=question, answer="")
    try:
        state = agent.invoke({"messages": [("user", question)]}, config={"recursion_limit": 20})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent 执行失败")
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    messages: list[BaseMessage] = state.get("messages", [])
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", []) or []:
                name = call.get("name", "")
                args = call.get("args", {})
                result.tool_calls.append(name)
                if name == "search_knowledge_base":
                    result.used_knowledge_base = True
                result.steps.append(f"调用工具 {name}({_short(str(args), 80)})")
            text = _extract_text(msg.content)
            if text.strip():
                result.answer = text.strip()
        elif isinstance(msg, ToolMessage):
            result.steps.append(f"  -> 返回 {_short(_extract_text(msg.content), 100)}")
    return result


def _short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


__all__ = ["build_agent", "run_agent", "AgentResult", "SYSTEM_PROMPT"]
