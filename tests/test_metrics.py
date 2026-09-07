"""确定性评测指标：拒答、关键词覆盖、检索命中、倒数排名。"""

import math

from langchain_core.documents import Document

from ragdoc.evals.metrics import (
    is_refusal,
    keyword_coverage,
    reciprocal_rank,
    retrieval_hit,
)


def test_keyword_coverage_full():
    assert keyword_coverage("住宿每晚 600 元", ["600"]) == 1.0
    assert keyword_coverage("包含 600 与 120", ["600", "120"]) == 1.0


def test_keyword_coverage_partial():
    assert keyword_coverage("只提到 600", ["600", "120"]) == 0.5


def test_keyword_coverage_case_insensitive():
    assert keyword_coverage("CHG-20240101-001 通过", ["chg-20240101"]) == 1.0


def test_keyword_coverage_empty_returns_nan():
    assert math.isnan(keyword_coverage("随便", []))


def test_retrieval_hit_when_evidence_in_chunk():
    docs = [Document(page_content="住宿每晚 600 元", metadata={})]
    assert retrieval_hit(docs, "每晚 600")


def test_retrieval_hit_false_when_absent():
    docs = [Document(page_content="绩效等级 S 档", metadata={})]
    assert not retrieval_hit(docs, "每晚 600")


def test_reciprocal_rank_first():
    docs = [Document(page_content="含 gold", metadata={})]
    assert reciprocal_rank(docs, "gold") == 1.0


def test_reciprocal_rank_second():
    docs = [
        Document(page_content="不含", metadata={}),
        Document(page_content="这里有 gold", metadata={}),
    ]
    assert reciprocal_rank(docs, "gold") == 0.5


def test_is_refusal_positive():
    for s in [
        "根据现有资料无法回答该问题。",
        "抱歉，我不确定",
        "没有找到相关信息",
    ]:
        assert is_refusal(s), s


def test_is_refusal_negative():
    assert not is_refusal("住宿标准是每晚 600 元")
    assert not is_refusal("可以办理")
