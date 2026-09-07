"""Agent 可用工具集。Agent 自己决定：查知识库 / 算数 / 查时间 / 直接回答。"""

from __future__ import annotations

import ast
import datetime as _dt
import operator
from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from ..vectorstore import format_docs


# ---------------------------------------------------------------- 计算器
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """只做算术，绝不使用 eval()，避免任意代码执行。"""
    try:
        value = _safe_eval(ast.parse(expression.strip(), mode="eval"))
    except Exception as exc:  # noqa: BLE001
        return f"计算失败：{exc}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


@tool
def calculator(expression: str) -> str:
    """执行加减乘除、乘方等算术运算。输入是纯数学表达式，例如 "120 * 3.5" 或 "(600-400)*2"。
    仅当问题需要计算结果，且知识库里没有现成答案时才使用。"""
    return safe_calculate(expression)


@tool
def get_current_datetime(_unused: str = "") -> str:
    """获取当前日期与时间，用于回答与"今天/现在/还剩多少天"相关的问题。不需要参数。"""
    now = _dt.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f" 星期{'一二三四五六日'[now.weekday()]}"


def make_knowledge_tool(retriever: BaseRetriever, top_k: int = 4) -> BaseTool:
    """工厂函数：把检索器包装成 Agent 可用的工具（必须放在函数里，因为要绑定运行时对象）。"""

    @tool
    def search_knowledge_base(query: str) -> str:
        """在《星海科技内部知识库手册》中检索与问题相关的原文片段。
        当问题涉及公司制度、报销、请假、服务器申请、安全合规、故障等级、绩效考核等**具体规定**时，
        必须先调用本工具，再基于返回的原文作答。输入应为面向检索的关键词或短句。"""
        docs = retriever.invoke(query)
        if not docs:
            return "（未检索到任何相关内容）"
        return "检索到以下内容：\n\n" + format_docs(docs[:top_k])

    return search_knowledge_base


def build_tools(retriever: BaseRetriever | None = None, top_k: int = 4) -> list[BaseTool]:
    tools: list[BaseTool] = []
    if retriever is not None:
        tools.append(make_knowledge_tool(retriever, top_k))
    tools.extend([calculator, get_current_datetime])
    return tools


__all__ = ["build_tools", "make_knowledge_tool", "calculator", "get_current_datetime", "safe_calculate"]
