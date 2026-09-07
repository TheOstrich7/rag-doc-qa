"""PDF 抽取后归一化的关键场景：断行合并 / 标题分段 / 句末不合并。"""

from ragdoc.loaders import normalize_pdf_text


def test_merges_pdf_line_break_within_sentence():
    text = "国内出差住宿标准为：一线城市\n（北京、上海、广州、深圳）每晚 600 元。"
    out = normalize_pdf_text(text)
    assert "（北京、上海、广州、深圳）每晚 600 元" in out, out


def test_does_not_merge_after_sentence_end():
    text = "每晚 600 元。\n二线城市每晚 400 元。"
    out = normalize_pdf_text(text)
    # 句号后必须有换行，作为段落分隔
    assert "每晚 600 元。\n" in out, out


def test_heading_creates_paragraph_break():
    text = "上一章内容。每晚 600 元。\n第 2 章 差旅与费用报销\n员工应在..."
    out = normalize_pdf_text(text)
    assert "\n\n第 2 章 差旅与费用报销" in out, out


def test_collapses_spaces_and_blank_lines():
    text = "每晚  600  元。\n\n\n\n二线城市每晚 400 元。"
    out = normalize_pdf_text(text)
    assert "每晚 600 元。" in out
    assert "二线城市每晚 400 元。" in out
    assert "\n\n\n" not in out
