"""全局配置。所有可调参数集中在这里，避免散落在业务代码里。

优先级：环境变量 / .env 文件 > 默认值。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（ragdoc/ 的上一级）
BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---------------- LLM（DeepSeek，OpenAI 兼容协议）----------------
    deepseek_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "RAGDOC_DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "RAGDOC_DEEPSEEK_BASE_URL"),
    )
    llm_model: str = Field(default="deepseek-chat", validation_alias=AliasChoices("LLM_MODEL"))
    llm_temperature: float = Field(default=0.0, validation_alias=AliasChoices("LLM_TEMPERATURE"))
    llm_max_tokens: int = Field(default=1024, validation_alias=AliasChoices("LLM_MAX_TOKENS"))
    request_timeout: int = Field(default=60, validation_alias=AliasChoices("REQUEST_TIMEOUT"))

    # ---------------- Embedding ----------------
    # auto | fastembed | huggingface | hash
    embedding_backend: str = Field(default="auto", validation_alias=AliasChoices("EMBEDDING_BACKEND"))
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5", validation_alias=AliasChoices("EMBEDDING_MODEL")
    )
    embedding_dim: int = Field(default=384, validation_alias=AliasChoices("EMBEDDING_DIM"))

    # ---------------- 切分 ----------------
    chunk_size: int = Field(default=400, validation_alias=AliasChoices("CHUNK_SIZE"))
    chunk_overlap: int = Field(default=80, validation_alias=AliasChoices("CHUNK_OVERLAP"))

    # ---------------- 检索 ----------------
    top_k: int = Field(default=4, validation_alias=AliasChoices("TOP_K"))
    # similarity | mmr
    search_type: str = Field(default="mmr", validation_alias=AliasChoices("SEARCH_TYPE"))
    fetch_k: int = Field(default=20, validation_alias=AliasChoices("FETCH_K"))
    lambda_mult: float = Field(default=0.5, validation_alias=AliasChoices("LAMBDA_MULT"))

    # ---------------- 路径 ----------------
    data_dir: Path = Field(default=BASE_DIR / "data")
    # 注意：faiss-cpu 在 Windows 下的 C++ I/O 不能处理含非 ASCII 字符的路径，
    # 所以索引默认放到用户家目录。如果你的项目根目录是纯 ASCII，可以用 INDEX_DIR 覆盖。
    index_dir: Path = Field(
        default_factory=lambda: Path.home() / ".ragdoc_index",
        validation_alias=AliasChoices("INDEX_DIR", "RAGDOC_INDEX_DIR"),
    )
    results_dir: Path = Field(default=BASE_DIR / "results")
    eval_file: Path = Field(default=BASE_DIR / "eval" / "questions.jsonl")

    @property
    def has_llm_key(self) -> bool:
        return bool(self.deepseek_api_key and self.deepseek_api_key.strip())

    def require_llm_key(self) -> None:
        if not self.has_llm_key:
            raise RuntimeError(
                "未检测到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入你的 Key：\n"
                "  cp .env.example .env\n"
                "然后重新运行命令。"
            )


settings = Settings()
