"""文档加载与切分。

PDF 抽取出来的文本有个典型坑：报告/PDF 的**断行**会把一句完整的话切成两行，
直接切块会把语义截断，检索质量明显下降。所以这里先做「标题感知 + 断行合并」的归一化，
再按中文标点优先级递归切分。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# 句末标点：出现这些标点后的换行才被认为是真正的段落结束
_SENT_END = set("。！？；…”’）】》!?;:")
# 中文/英文标题行（"第 2 章 xxx"、"3.1 xxx"），用于在归一化时强制单独成段
_HEADING_RE = re.compile(
    r"^\s*(第\s*[0-9一二三四五六七八九十]+\s*[章节篇部]|[0-9]+(?:\.[0-9]+)*)\s*\S",
    re.MULTILINE,
)

_CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]


def normalize_pdf_text(text: str) -> str:
    """修掉 PDF 断行，并把标题行单独成段。

    实现要点：按行处理而不是整段正则替换，因为 PDF 抽取的断行可能是"句中换行"
    （需要拼回去）或"标题/段尾换行"（需要保留）。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)

    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if _HEADING_RE.match(line):
            # 标题：保证前后都有空行
            if out and out[-1] != "":
                out.append("")
            out.append(line)
            out.append("")
            continue
        # 普通行：若上一行不是句末标点结尾，就拼到上一行（修复 PDF 断行）
        if out and out[-1] != "" and (out[-1][-1] not in _SENT_END):
            out[-1] = out[-1] + line
        else:
            out.append(line)

    normalized = "\n".join(out)
    # 中文 PDF 的段落之间往往没有空行，只有"句末标点 + 换行"。
    # 把这种断行升级为真正的段落分隔（双换行），让切块器能按段落切。
    normalized = re.sub(rf"([{re.escape(''.join(_SENT_END))}])\n(?=\S)", r"\1\n\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


@dataclass
class SplitConfig:
    chunk_size: int = 400
    chunk_overlap: int = 80


def build_splitter(cfg: SplitConfig | None = None) -> RecursiveCharacterTextSplitter:
    cfg = cfg or SplitConfig()
    return RecursiveCharacterTextSplitter(
        separators=_CHINESE_SEPARATORS,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def load_pdf(path: str | Path) -> list[Document]:
    """按页加载 PDF，并做归一化。"""
    from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 PDF：{path}")
    pages = PyPDFLoader(str(path)).load()
    for doc in pages:
        doc.page_content = normalize_pdf_text(doc.page_content)
        doc.metadata["source"] = str(path)
    logger.info("加载 PDF：%s，共 %d 页", path.name, len(pages))
    return pages


def split_documents(
    docs: list[Document],
    cfg: SplitConfig | None = None,
) -> list[Document]:
    """切块，并把最近出现的章节标题写进 metadata['section']。"""
    splitter = build_splitter(cfg)
    chunks: list[Document] = []
    section = ""

    for doc in docs:
        for block in doc.page_content.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            first_line = block.split("\n", 1)[0].strip()
            # 短且像标题的行 -> 更新当前章节，不单独成块内容
            if len(first_line) <= 30 and _HEADING_RE.match(first_line) and "\n" not in block:
                section = first_line
                continue
            for piece in splitter.split_text(block):
                piece = piece.strip()
                if not piece:
                    continue
                meta = dict(doc.metadata)
                meta["section"] = section
                chunks.append(Document(page_content=piece, metadata=meta))

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    logger.info("切分完成：%d 个 chunk（size=%s, overlap=%s）", len(chunks),
                (cfg or SplitConfig()).chunk_size, (cfg or SplitConfig()).chunk_overlap)
    return chunks


def load_and_split(path: str | Path, cfg: SplitConfig | None = None) -> list[Document]:
    return split_documents(load_pdf(path), cfg)
