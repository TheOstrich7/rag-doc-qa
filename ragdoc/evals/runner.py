"""评测执行器：跑「纯大模型 vs RAG」两套链路，算指标，产出报告。

用法：
    python -m ragdoc.cli eval                # 跑全量
    python -m ragdoc.cli eval --limit 10     # 快速试跑
    python -m ragdoc.cli eval --no-baseline  # 只跑 RAG（省一半 token）
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from ..chains.naive import naive_answer
from ..chains.rag import NO_ANSWER, rag_answer
from ..config import Settings, settings as default_settings
from ..embeddings import get_embeddings
from ..llm import get_chat_llm
from ..vectorstore import build_retriever, load_index
from . import metrics as M
from .dataset import EvalItem, load_dataset, stats

logger = logging.getLogger(__name__)


@dataclass
class Row:
    id: str
    category: str
    answerable: bool
    question: str
    reference: str
    rag_answer: str
    rag_latency_ms: float
    rag_refused: bool
    retrieval_hit: bool
    reciprocal_rank: float
    rag_keyword_coverage: float
    rag_correctness: float
    rag_faithfulness: float
    baseline_answer: str = ""
    baseline_latency_ms: float = float("nan")
    baseline_refused: bool = False
    baseline_keyword_coverage: float = float("nan")
    baseline_correctness: float = float("nan")
    judge_notes: str = ""


@dataclass
class Report:
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    config: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)
    rows: list[Row] = field(default_factory=list)


def _correctness_of(item: EvalItem, answer: str, judge_llm: BaseChatModel) -> tuple[float, str]:
    """可回答题用 LLM 打分；越界题用"是否明确拒答"判定。"""
    if not item.answerable:
        refused = M.is_refusal(answer)
        return (1.0 if refused else 0.0), ("已拒答" if refused else "未拒答（编造）")
    score = M.judge_correctness(judge_llm, item.question, item.reference, answer)
    return score.score, score.reason


def run_evaluation(
    cfg: Settings | None = None,
    *,
    limit: int | None = None,
    run_baseline: bool = True,
    categories: list[str] | None = None,
    verbose: bool = False,
) -> Report:
    cfg = cfg or default_settings
    cfg.require_llm_key()

    embeddings = get_embeddings(cfg)
    vs = load_index(cfg.index_dir, embeddings)
    retriever = build_retriever(
        vs, top_k=cfg.top_k, search_type=cfg.search_type,
        fetch_k=cfg.fetch_k, lambda_mult=cfg.lambda_mult,
    )
    llm = get_chat_llm(cfg)
    # Judge 用 JSON 模式，且限制输出 token（DeepSeek 偶尔会失控长生成）
    judge_llm = get_chat_llm(cfg, json_mode=True, max_tokens=256)

    dataset = load_dataset(cfg.eval_file)
    if categories:
        dataset = [d for d in dataset if d.category in categories]
    if limit:
        dataset = dataset[:limit]
    if not dataset:
        raise ValueError("评测集为空")

    report = Report(
        config={
            "llm_model": cfg.llm_model,
            "embedding_backend": cfg.embedding_backend,
            "embedding_model": cfg.embedding_model,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "top_k": cfg.top_k,
            "search_type": cfg.search_type,
            "index_dir": str(cfg.index_dir),
        },
        dataset=stats(dataset),
    )

    rows: list[Row] = []
    t_start = time.perf_counter()
    for i, item in enumerate(dataset, 1):
        logger.info("[%d/%d] %s %s", i, len(dataset), item.id, item.question)
        docs = retriever.invoke(item.question)
        rag = rag_answer(None, llm, item.question, docs=docs)

        rag_corr, rag_note = _correctness_of(item, rag.answer, judge_llm)
        if rag.refused:
            rag_faith = 1.0  # 明确拒答 = 没有编造
        else:
            rag_faith = M.judge_faithfulness(judge_llm, rag.context_text, rag.answer).score

        row = Row(
            id=item.id,
            category=item.category,
            answerable=item.answerable,
            question=item.question,
            reference=item.reference,
            rag_answer=rag.answer,
            rag_latency_ms=round(rag.latency_ms, 1),
            rag_refused=rag.refused,
            retrieval_hit=M.retrieval_hit(docs, item.gold_evidence),
            reciprocal_rank=M.reciprocal_rank(docs, item.gold_evidence),
            rag_keyword_coverage=M.keyword_coverage(rag.answer, item.keywords),
            rag_correctness=rag_corr,
            rag_faithfulness=rag_faith,
            judge_notes=rag_note,
        )

        if run_baseline:
            base = naive_answer(llm, item.question)
            base_corr, base_note = _correctness_of(item, base.answer, judge_llm)
            row.baseline_answer = base.answer
            row.baseline_latency_ms = round(base.latency_ms, 1)
            row.baseline_refused = M.is_refusal(base.answer)
            row.baseline_keyword_coverage = M.keyword_coverage(base.answer, item.keywords)
            row.baseline_correctness = base_corr
            row.judge_notes = f"RAG: {rag_note} | 基线: {base_note}"
            if verbose:
                print(f"  基线：{base.answer[:80]}")

        if verbose:
            print(f"  RAG ：{rag.answer[:80]}")
        rows.append(row)

    report.rows = rows
    report.summary = _summarize(rows, elapsed=time.perf_counter() - t_start)
    return report


def _summarize(rows: list[Row], elapsed: float) -> dict[str, float]:
    ans = [r for r in rows if r.answerable]
    out = {
        "n": len(rows),
        "elapsed_sec": round(elapsed, 1),
        # 检索质量
        "retrieval_recall@k": M.mean([1.0 if r.retrieval_hit else 0.0 for r in ans]),
        "retrieval_mrr": M.mean([r.reciprocal_rank for r in ans]),
        # 答案质量
        "rag_correctness": M.mean([r.rag_correctness for r in rows]),
        "rag_correctness_answerable": M.mean([r.rag_correctness for r in ans]),
        "rag_keyword_coverage": M.mean([r.rag_keyword_coverage for r in ans]),
        "rag_faithfulness": M.mean([r.rag_faithfulness for r in rows]),
        "rag_hallucination_rate": 1 - M.mean([r.rag_faithfulness for r in rows]),
        "rag_refusal_rate_answerable": M.mean([1.0 if r.rag_refused else 0.0 for r in ans]),
        "rag_latency_ms": M.mean([r.rag_latency_ms for r in rows]),
    }
    unans = [r for r in rows if not r.answerable]
    if unans:
        out["rag_refusal_accuracy_unanswerable"] = M.mean(
            [1.0 if r.rag_refused else 0.0 for r in unans]
        )

    if any(r.baseline_answer for r in rows):
        out.update(
            {
                "baseline_correctness": M.mean([r.baseline_correctness for r in rows]),
                "baseline_correctness_answerable": M.mean(
                    [r.baseline_correctness for r in ans]
                ),
                "baseline_keyword_coverage": M.mean([r.baseline_keyword_coverage for r in ans]),
                "baseline_latency_ms": M.mean([r.baseline_latency_ms for r in rows]),
            }
        )
        if unans:
            out["baseline_refusal_accuracy_unanswerable"] = M.mean(
                [1.0 if _baseline_refused(r) else 0.0 for r in unans]
            )
    return out


def _baseline_refused(r: Row) -> bool:
    return r.baseline_refused


# ------------------------------------------------------------------ 输出
def _fmt(v: float, digits: int = 1) -> str:
    return "-" if v != v else f"{v * 100:.{digits}f}%"


def render_markdown(report: Report) -> str:
    s = report.summary
    has_base = "baseline_correctness" in s
    lines = [
        f"# RAG 评测报告",
        "",
        f"- 生成时间：{report.created_at}",
        f"- 模型：`{report.config.get('llm_model')}`",
        f"- Embedding：`{report.config.get('embedding_backend')}` / `{report.config.get('embedding_model')}`",
        f"- 切分：chunk_size={report.config.get('chunk_size')}, overlap={report.config.get('chunk_overlap')}",
        f"- 检索：top_k={report.config.get('top_k')}, search_type={report.config.get('search_type')}",
        f"- 样本：{report.dataset.get('total')} 题（可回答 {report.dataset.get('answerable')} / 越界 {report.dataset.get('unanswerable')}）",
        "",
        "## 总体对比",
        "",
        "| 指标 | 纯大模型 | RAG | 提升 |",
        "| --- | --- | --- | --- |",
    ]

    def row(name: str, base_key: str, rag_key: str, higher_better: bool = True) -> str:
        b = s.get(base_key, float("nan"))
        r = s.get(rag_key, float("nan"))
        if has_base and b == b and r == r:
            delta = r - b
            sign = "+" if delta >= 0 else ""
            return f"| {name} | {_fmt(b)} | {_fmt(r)} | {sign}{delta * 100:.1f}pp |"
        return f"| {name} | {_fmt(b) if has_base else '-'} | {_fmt(r)} | - |"

    if has_base:
        lines += [
            row("答案正确率（全量，LLM 判分）", "baseline_correctness", "rag_correctness"),
            row("答案正确率（可回答题）", "baseline_correctness_answerable", "rag_correctness_answerable"),
            row("关键数字覆盖率", "baseline_keyword_coverage", "rag_keyword_coverage"),
        ]
    else:
        lines += [
            f"| 答案正确率（全量） | - | {_fmt(s.get('rag_correctness', float('nan')))} | - |",
        ]
    if "baseline_refusal_accuracy_unanswerable" in s:
        lines.append(
            row("越界问题正确拒答率", "baseline_refusal_accuracy_unanswerable",
                "rag_refusal_accuracy_unanswerable")
        )
    lines += [
        f"| 有据可依率（忠实度，越低越爱编） | - | {_fmt(s.get('rag_faithfulness', float('nan')))} | - |",
        f"| 幻觉率 | - | {_fmt(s.get('rag_hallucination_rate', float('nan')))} | - |",
        f"| 检索召回率@k | - | {_fmt(s.get('retrieval_recall@k', float('nan')))} | - |",
        f"| 检索 MRR | - | {_fmt(s.get('retrieval_mrr', float('nan')))} | - |",
        f"| 可回答题误拒答率 | - | {_fmt(s.get('rag_refusal_rate_answerable', float('nan')))} | - |",
    ]
    if has_base:
        lines.append(
            f"| 平均延迟 | {s.get('baseline_latency_ms', 0):.0f} ms | "
            f"{s.get('rag_latency_ms', 0):.0f} ms | - |"
        )
    lines += ["", f"耗时 {s.get('elapsed_sec')} 秒。", "", "## 逐题明细", "",
              "| # | 问题 | 检索命中 | RAG 正确率 | 基线正确率 | RAG 答案（截断） |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in report.rows:
        q = r.question if len(r.question) <= 28 else r.question[:28] + "…"
        a = " ".join(r.rag_answer.split())
        a = a if len(a) <= 40 else a[:40] + "…"
        base = _fmt(r.baseline_correctness, 0) if r.baseline_answer else "-"
        lines.append(
            f"| {r.id} | {q} | {'✓' if r.retrieval_hit else '✗'} | "
            f"{_fmt(r.rag_correctness, 0)} | {base} | {a} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(report: Report, results_dir: Path) -> dict[str, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    md_path = results_dir / "report.md"
    json_path = results_dir / "report.json"
    raw_path = results_dir / "raw_records.jsonl"

    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "created_at": report.created_at,
                "config": report.config,
                "dataset": report.dataset,
                "summary": report.summary,
                "rows": [asdict(r) for r in report.rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with raw_path.open("w", encoding="utf-8") as f:
        for r in report.rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    out = {"markdown": md_path, "json": json_path, "raw": raw_path}
    png = _try_plot(report, results_dir / "compare.png")
    if png:
        out["chart"] = png
    return out


def _try_plot(report: Report, path: Path) -> Path | None:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001
        return None

    s = report.summary
    labels, base_vals, rag_vals = [], [], []
    if "baseline_correctness" in s:
        labels += ["答案正确率", "可回答题正确率", "关键数字覆盖率"]
        base_vals += [s["baseline_correctness"], s["baseline_correctness_answerable"],
                      s["baseline_keyword_coverage"]]
        rag_vals += [s["rag_correctness"], s["rag_correctness_answerable"],
                     s["rag_keyword_coverage"]]
    if "baseline_refusal_accuracy_unanswerable" in s:
        labels.append("越界题拒答率")
        base_vals.append(s["baseline_refusal_accuracy_unanswerable"])
        rag_vals.append(s["rag_refusal_accuracy_unanswerable"])
    labels.append("有据可依率")
    base_vals.append(float("nan"))
    rag_vals.append(s["rag_faithfulness"])

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    ax.bar([i - width / 2 for i in x], [max(v, 0) if v == v else 0 for v in base_vals],
           width, label="纯大模型", color="#9aa5b1")
    ax.bar([i + width / 2 for i in x], [max(v, 0) if v == v else 0 for v in rag_vals],
           width, label="RAG", color="#2b6cb0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例")
    ax.set_title(f"纯大模型 vs RAG（{report.dataset.get('total')} 题 · {report.config.get('llm_model')}）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (b, r) in enumerate(zip(base_vals, rag_vals)):
        if r == r:
            ax.text(i + width / 2, min(r + 0.03, 1.0), f"{r * 100:.0f}", ha="center", fontsize=9)
        if b == b:
            ax.text(i - width / 2, min(b + 0.03, 1.0), f"{b * 100:.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
