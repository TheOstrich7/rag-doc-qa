"""Embedding 层：可插拔的三种后端。

- fastembed  : ONNX 推理，体积小（bge-small-zh 约 90MB），**默认**，不依赖 torch
- huggingface: sentence-transformers，效果最好，但会拉 torch（约 2GB）
- hash       : 纯 Python 字符 n-gram 哈希向量，零依赖、零下载，用于离线冒烟测试与单测

auto 模式依次尝试 fastembed -> huggingface -> hash，保证任何环境都能跑通链路。
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Iterable

from langchain_core.embeddings import Embeddings

from .config import Settings

logger = logging.getLogger(__name__)

_LATIN_OR_NUM = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
_CJK = re.compile(r"[\u4e00-\u9fff]")


class HashingEmbeddings(Embeddings):
    """零依赖的确定性哈希向量（词袋 + 字符 n-gram），用于离线冒烟与单元测试。

    语义表达能力远弱于神经模型，但对"术语精确匹配 + 数字"这类检索有不错的表现，
    因此用它跑通链路是安全的；正式评测请务必切到 fastembed / huggingface。
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    # ---- 内部工具 ----
    @staticmethod
    def _idx_sign(token: str, dim: int) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        val = int.from_bytes(digest[:8], "big")
        return val % dim, 1.0 if (val >> 63) & 1 else -1.0

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        low = text.lower()
        # 英文单词 / 数字：权重高
        for tok in _LATIN_OR_NUM.findall(low):
            yield (f"w:{tok}", 2.0)
        # 中文字符：uni-gram + bi-gram
        chars = [c for c in low if _CJK.match(c)]
        for c in chars:
            yield (f"c:{c}", 1.0)
        for a, b in zip(chars, chars[1:]):
            yield (f"b:{a}{b}", 1.5)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token, weight in self._features(text):
            idx, sign = self._idx_sign(token, self.dim)
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    # ---- LangChain 接口 ----
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def _build_fastembed(cfg: Settings) -> Embeddings:
    from langchain_community.embeddings import FastEmbedEmbeddings  # noqa: PLC0415

    return FastEmbedEmbeddings(model_name=cfg.embedding_model)


def _build_huggingface(cfg: Settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415

    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings(cfg: Settings | None = None) -> Embeddings:
    """按配置构造 Embeddings；auto 模式下依次降级。"""
    cfg = cfg or Settings()
    backend = (cfg.embedding_backend or "auto").lower()

    if backend == "hash":
        logger.warning("使用 HashingEmbeddings：仅供离线冒烟测试，请勿用于正式评测。")
        return HashingEmbeddings(dim=cfg.embedding_dim)

    if backend == "fastembed":
        return _build_fastembed(cfg)
    if backend == "huggingface":
        return _build_huggingface(cfg)

    # auto
    for name, builder in (("fastembed", _build_fastembed), ("huggingface", _build_huggingface)):
        try:
            emb = builder(cfg)
            logger.info("Embedding 后端：%s (%s)", name, cfg.embedding_model)
            return emb
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding 后端 %s 初始化失败：%s", name, exc)

    logger.warning("所有神经 Embedding 后端均不可用，降级为 HashingEmbeddings（仅供冒烟测试）。")
    return HashingEmbeddings(dim=cfg.embedding_dim)


def describe_backend(emb: Embeddings) -> str:
    return type(emb).__name__
