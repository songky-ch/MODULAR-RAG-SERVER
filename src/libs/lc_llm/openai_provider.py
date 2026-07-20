"""OpenAI 兼容提供商配置映射器。

Covers the standard OpenAI API as well as any OpenAI-compatible
endpoint such as Alibaba DashScope, DeepSeek, etc.  The ``base_url``
parameter is the key to making one provider class work for all of them.

Supports all three model types: LLM, Vision LLM, and Embedding.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from src.libs.lc_llm.base_provider import BaseProviderConfig

if TYPE_CHECKING:
    from src.core.settings import Settings


class OpenAIProviderConfig(BaseProviderConfig):
    """将 ``settings.yaml`` 转换为 OpenAI 兼容模型的关键字参数。"""

    def get_model_provider(self) -> str:
        return "openai"

    def get_required_package(self) -> str:
        return "langchain-openai"

    # ------------------------------------------------------------------
    # 文本 LLM（读取 settings.llm）
    # ------------------------------------------------------------------

    def build_llm_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        cfg = settings.llm

        kwargs: dict[str, Any] = {
            "model": overrides.get("model", cfg.model),
            "temperature": overrides.get("temperature", cfg.temperature),
            "max_tokens": overrides.get("max_tokens", cfg.max_tokens),
        }

        api_key = (
            overrides.get("api_key")
            or cfg.api_key
            or os.environ.get("OPENAI_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key

        base_url = overrides.get("base_url") or cfg.base_url
        if base_url:
            kwargs["base_url"] = base_url

        return kwargs

    # ------------------------------------------------------------------
    # 视觉 LLM（读取 settings.vision_llm）
    # ------------------------------------------------------------------

    # 视觉请求的默认超时时间（图片加 Prompt 比纯文本更慢）
    VISION_DEFAULT_TIMEOUT = 120

    def build_vision_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        cfg = settings.vision_llm

        kwargs: dict[str, Any] = {
            "model": overrides.get("model", cfg.model),
        }

        api_key = (
            overrides.get("api_key")
            or cfg.api_key
            or os.environ.get("OPENAI_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key

        base_url = overrides.get("base_url") or getattr(cfg, "base_url", None) or getattr(cfg, "endpoint", None)
        if base_url:
            kwargs["base_url"] = base_url

        timeout = overrides.get("timeout") or getattr(cfg, "timeout", None)
        kwargs["timeout"] = int(timeout) if timeout is not None else self.VISION_DEFAULT_TIMEOUT

        return kwargs

    # ------------------------------------------------------------------
    # 向量模型（读取 settings.embedding）
    # ------------------------------------------------------------------

    def build_embedding_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        cfg = settings.embedding

        kwargs: dict[str, Any] = {
            "model": overrides.get("model", cfg.model),
        }

        api_key = (
            overrides.get("api_key")
            or cfg.api_key
            or os.environ.get("OPENAI_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key

        base_url = overrides.get("base_url") or cfg.base_url
        if base_url:
            kwargs["base_url"] = base_url
            # OpenAI 兼容端点（DashScope 等）不接受分词后的输入，
            # 因此关闭客户端的 tiktoken 切分。
            kwargs.setdefault("check_embedding_ctx_length", False)

        dimensions = overrides.get("dimensions") or cfg.dimensions
        if dimensions:
            kwargs["dimensions"] = dimensions

        return kwargs
