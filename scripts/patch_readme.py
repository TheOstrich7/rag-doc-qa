"""把 `results/report.md` 里的「总体对比」表格回填到 README 的评测占位区。

用法：
    python -m ragdoc.cli eval         # 跑评测，生成 results/report.md
    python scripts/patch_readme.py    # 把数字回填到 README
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "results" / "report.md"

START_MARK = "<!-- EVAL_TABLE_START -->"
END_MARK = "<!-- EVAL_TABLE_END -->"


def extract_table(report_text: str) -> str:
    m = re.search(r"## 总体对比(.*?)\n## 逐题明细", report_text, re.S)
    if not m:
        raise SystemExit("无法在 report.md 中定位'总体对比'段。请确认 eval 成功跑完。")
    body = m.group(1).rstrip()
    # 提取配置/样本说明（生成时间、Embedding、样本数等），去掉 H1
    meta = ""
    m_meta = re.search(
        r"^- 生成时间.*?(?=\n## )", report_text, re.S
    )
    if m_meta:
        meta = m_meta.group(0).rstrip() + "\n"
    warning = (
        "> ⚠️ 下面这张表是 `python -m ragdoc.cli eval` 跑出来的最新数字。"
        "如果显示全 0 / 100% 幻觉，**通常是因为还没在 `.env` 里填 DEEPSEEK_API_KEY**，"
        "框架跑通但大模型那边没真正接上。\n"
    )
    return f"{warning}{meta}\n## 总体对比\n\n{body}\n"


def main() -> None:
    if not REPORT.exists():
        raise SystemExit(f"未找到 {REPORT}。请先运行：python -m ragdoc.cli eval")
    if not README.exists():
        raise SystemExit(f"未找到 {README}")
    readme = README.read_text(encoding="utf-8")
    if START_MARK not in readme or END_MARK not in readme:
        raise SystemExit(
            f"README 缺少 {START_MARK} / {END_MARK} 占位标记，"
            "请确认 README 模板完整。"
        )
    block = extract_table(REPORT.read_text(encoding="utf-8"))
    new = re.sub(
        re.escape(START_MARK) + ".*?" + re.escape(END_MARK),
        f"{START_MARK}\n{block}\n{END_MARK}",
        readme,
        flags=re.S,
    )
    README.write_text(new, encoding="utf-8")
    print(f"[ok] 已把 {REPORT.name} 的总体对比表回填到 {README.name}")
    print(f"     打开 {README} 即可看到最新数字。")


if __name__ == "__main__":
    main()
