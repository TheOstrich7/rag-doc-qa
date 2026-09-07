"""命令行入口。子命令：

  ingest  从 PDF 建索引
  ask     单轮 RAG 问答
  search  只做检索（调试用）
  eval    跑「纯大模型 vs RAG」评测，生成报告
  agent   用 ReAct Agent 回答，演示工具调用

示例：
  python -m ragdoc.cli ingest --pdf data/sample_knowledge_base.pdf
  python -m ragdoc.cli ask "去深圳出差，住宿标准是多少？"
  python -m ragdoc.cli search "差旅补贴怎么算"
  python -m ragdoc.cli eval --limit 10
  python -m ragdoc.cli agent "公司附近有什么好吃的" --show-trace
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import agent as agent_mod
from . import evals as evals_mod
from .chains.rag import rag_answer
from .chains.naive import naive_answer
from .config import settings
from .embeddings import get_embeddings, describe_backend
from .llm import get_chat_llm
from .loaders import SplitConfig, load_and_split
from .vectorstore import build_index, build_retriever, load_index, retrieve_with_score


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    pdf_arg = (args.pdf or "").strip()
    if not pdf_arg:
        candidates = sorted(settings.data_dir.glob("*.pdf"))
        if not candidates:
            print(f"[error] 未指定 PDF，且 {settings.data_dir} 下没有 PDF 文件。")
            print("        可指定 --pdf <路径>，或把 PDF 放到 data/ 目录下。")
            return 1
        pdf_path = candidates[0]
        print(f"[info] 未指定 PDF，使用 {pdf_path}")
    else:
        pdf_path = Path(pdf_arg)
        if not pdf_path.exists():
            print(f"[error] 找不到 PDF：{pdf_path}")
            return 1
    docs = load_and_split(pdf_path, SplitConfig(settings.chunk_size, settings.chunk_overlap))
    embeddings = get_embeddings(settings)
    print(f"[info] Embedding 后端：{describe_backend(embeddings)} ({settings.embedding_model})")
    vs = build_index(docs, embeddings)
    from .vectorstore import save_index
    save_index(vs, settings.index_dir)
    print(f"[ok] 已建立索引：{len(docs)} 个 chunk -> {settings.index_dir}")
    return 0


def _load_vs():
    embeddings = get_embeddings(settings)
    vs = load_index(settings.index_dir, embeddings)
    return vs, embeddings


def cmd_ask(args: argparse.Namespace) -> int:
    settings.require_llm_key()
    vs, _ = _load_vs()
    retriever = build_retriever(
        vs, top_k=args.k or settings.top_k, search_type=settings.search_type,
        fetch_k=settings.fetch_k, lambda_mult=settings.lambda_mult,
    )
    llm = get_chat_llm(settings)
    res = rag_answer(retriever, llm, args.question)
    print(res.answer)
    if args.show_context:
        print("\n--- 召回的参考资料 ---")
        for i, d in enumerate(res.contexts, 1):
            sec = d.metadata.get("section", "")
            page = d.metadata.get("page", "")
            tag = f"[{i}] {sec}" + (f" p{int(page) + 1}" if page != "" else "")
            print(f"\n{tag}\n{d.page_content}")
    if args.show_scores:
        print(f"\n(latency: {res.latency_ms:.0f}ms, refused: {res.refused})")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    vs, _ = _load_vs()
    hits = retrieve_with_score(vs, args.query, k=args.k or settings.top_k)
    for i, (doc, score) in enumerate(hits, 1):
        sec = doc.metadata.get("section", "")
        page = doc.metadata.get("page", "")
        tag = f"[{i}] score={score:.4f} {sec}" + (f" p{int(page) + 1}" if page != "" else "")
        print(f"{tag}\n{doc.page_content}\n")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings.require_llm_key()
    report = evals_mod.runner.run_evaluation(
        settings,
        limit=args.limit,
        run_baseline=not args.no_baseline,
        categories=[c.strip() for c in args.category.split(",")] if args.category else None,
        verbose=args.verbose,
    )
    paths = evals_mod.runner.write_outputs(report, settings.results_dir)
    print(f"\n[ok] 评测完成，共 {report.summary['n']} 题（{report.summary['elapsed_sec']}s）")
    for k, v in paths.items():
        print(f"  - {k}: {v}")
    print("\n--- 摘要 ---")
    for k, v in report.summary.items():
        if isinstance(v, float):
            print(f"  {k:38s} = {v:.4f}")
        else:
            print(f"  {k:38s} = {v}")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    settings.require_llm_key()
    vs, _ = _load_vs()
    retriever = build_retriever(
        vs, top_k=settings.top_k, search_type=settings.search_type,
        fetch_k=settings.fetch_k, lambda_mult=settings.lambda_mult,
    )
    llm = get_chat_llm(settings)
    from .agent import graph as agent_graph, tools as agent_tools
    agent = agent_graph.build_agent(llm, retriever, top_k=settings.top_k)
    res = agent_graph.run_agent(agent, args.question)
    print(res.answer)
    if args.show_trace:
        print("\n--- Agent 决策轨迹 ---")
        for step in res.steps or ["(无工具调用)"]:
            print(f"  · {step}")
        if res.used_knowledge_base:
            print("  · ✓ 决定检索知识库")
        if not res.tool_calls:
            print("  · ✗ 没调用任何工具，直接回答")
        if res.error:
            print(f"  ! {res.error}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ragdoc", description="RAG 文档问答系统（LangChain + FAISS + DeepSeek）")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="从 PDF 建立索引")
    sp.add_argument("--pdf", default="", help="PDF 路径（默认取 data/ 下第一个）")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("ask", help="单轮 RAG 问答")
    sp.add_argument("question")
    sp.add_argument("-k", type=int, default=None)
    sp.add_argument("--show-context", action="store_true")
    sp.add_argument("--show-scores", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("search", help="只看检索结果（不调大模型）")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=None)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("eval", help="跑「纯大模型 vs RAG」评测")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--no-baseline", action="store_true", help="只跑 RAG，节省一半 API 调用")
    sp.add_argument("--category", default="", help="按类别过滤（逗号分隔）")
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("agent", help="ReAct Agent：让模型自己决定查不查知识库")
    sp.add_argument("question")
    sp.add_argument("--show-trace", action="store_true")
    sp.set_defaults(func=cmd_agent)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
