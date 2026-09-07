"""LLM 工厂。DeepSeek 官方接口是 OpenAI 兼容协议，直接用 ChatOpenAI 指向其 base_url。"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import Settings


def get_chat_llm(
    cfg: Settings | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> ChatOpenAI:
    cfg = cfg or Settings()
    cfg.require_llm_key()
    kwargs: dict = {}
    if json_mode:
        # DeepSeek 支持 JSON Output
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        temperature=cfg.llm_temperature if temperature is None else temperature,
        max_tokens=max_tokens or cfg.llm_max_tokens,
        timeout=cfg.request_timeout,
        max_retries=2,
        **kwargs,
    )


def get_fake_llm(responses: list[str] | None = None) -> BaseChatModel:
    """离线测试用的假 LLM，不需要 API Key。"""
    from langchain_core.language_models.fake_chat_models import (  # noqa: PLC0415
        FakeListChatModel,
    )

    return FakeListChatModel(responses=responses or ["这是离线假模型的固定回答。"])


__all__ = ["get_chat_llm", "get_fake_llm", "BaseChatModel"]
