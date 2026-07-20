"""适配 OpenAI 兼容对话与视觉模型的小米 MiMo 提供商。"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from src.libs.lc_llm.openai_provider import OpenAIProviderConfig

if TYPE_CHECKING:
    from src.core.settings import Settings


class MiMoProviderConfig(OpenAIProviderConfig):
    """使用 ``MIMO_API_KEY`` 构建 MiMo 对话/视觉客户端。"""

    DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"

    def build_llm_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        kwargs = super().build_llm_kwargs(settings, **overrides)
        api_key = (
            overrides.get("api_key")
            or settings.llm.api_key
            or os.environ.get("MIMO_LLM_API_KEY")
            or os.environ.get("MIMO_API_KEY")
        )
        if not api_key:
            raise ValueError("MiMo API key not provided. Set MIMO_API_KEY.")
        kwargs["api_key"] = api_key
        kwargs["base_url"] = (
            overrides.get("base_url")
            or settings.llm.base_url
            or os.environ.get("MIMO_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return kwargs

    def build_vision_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        kwargs = super().build_vision_kwargs(settings, **overrides)
        cfg = settings.vision_llm
        api_key = (
            overrides.get("api_key")
            or cfg.api_key
            or os.environ.get("MIMO_VISION_API_KEY")
            or os.environ.get("MIMO_API_KEY")
        )
        if not api_key:
            raise ValueError("MiMo API key not provided. Set MIMO_API_KEY.")
        kwargs["api_key"] = api_key
        kwargs["base_url"] = (
            overrides.get("base_url")
            or cfg.base_url
            or os.environ.get("MIMO_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        kwargs["max_completion_tokens"] = settings.llm.max_tokens
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return kwargs

    def build_embedding_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Xiaomi MiMo does not provide an embedding model for this pipeline. "
            "Use provider='ollama' for local embeddings."
        )
