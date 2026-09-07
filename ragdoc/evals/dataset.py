"""评测数据集定义与加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalItem:
    id: str
    question: str
    reference: str
    category: str = ""
    answerable: bool = True
    gold_evidence: str | None = None
    keywords: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.keywords = self.keywords or []


def load_dataset(path: str | Path) -> list[EvalItem]:
    path = Path(path)
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
            items.append(
                EvalItem(
                    id=raw["id"],
                    question=raw["question"],
                    reference=raw.get("reference", ""),
                    category=raw.get("category", ""),
                    answerable=bool(raw.get("answerable", True)),
                    gold_evidence=raw.get("gold_evidence"),
                    keywords=raw.get("keywords", []),
                )
            )
    return items


def stats(dataset: list[EvalItem]) -> dict:
    cats: dict[str, int] = {}
    for it in dataset:
        cats[it.category] = cats.get(it.category, 0) + 1
    return {
        "total": len(dataset),
        "answerable": sum(1 for i in dataset if i.answerable),
        "unanswerable": sum(1 for i in dataset if not i.answerable),
        "by_category": cats,
    }
