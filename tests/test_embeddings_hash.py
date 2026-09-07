"""HashingEmbeddings：纯依赖、向量化稳定、长度正确。"""

import math

from ragdoc.embeddings import HashingEmbeddings


def test_dim_matches_setting():
    h = HashingEmbeddings(dim=128)
    v = h.embed_query("深圳住宿每晚 600 元")
    assert len(v) == 128


def test_deterministic():
    h = HashingEmbeddings(dim=64)
    a = h.embed_query("去深圳出差 600 元一晚")
    b = h.embed_query("去深圳出差 600 元一晚")
    assert a == b


def test_l2_normalized():
    h = HashingEmbeddings(dim=64)
    v = h.embed_query("国内出差住宿标准为每晚 600 元")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_different_texts_not_identical():
    h = HashingEmbeddings(dim=128)
    a = h.embed_query("出差住宿每晚 600 元")
    b = h.embed_query("考核绩效等级 S 档系数 3.0")
    sim = sum(x * y for x, y in zip(a, b))
    assert sim < 0.95  # 不同主题应区分开
