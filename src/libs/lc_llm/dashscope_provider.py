"""DashScope (Qwen) provider configuration mapper.

Uses the native dashscope SDK via langchain_community's ChatTongyi and
DashScopeEmbeddings. Supports LLM, Vision LLM, and Embedding.

Requires: dashscope, langchain-community
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.libs.lc_llm.base_provider import BaseProviderConfig

if TYPE_CHECKING:
    from src.core.settings import Settings


class DashScopeProviderConfig(BaseProviderConfig):
    """Translate ``settings.yaml`` into kwargs for DashScope (Qwen) models.

    Uses ChatTongyi for LLM/Vision and DashScopeEmbeddings for embeddings.
    API key can be set via settings or DASHSCOPE_API_KEY env var.
    """

    USES_DIRECT_INSTANTIATION = True

    def get_model_provider(self) -> str:
        return "dashscope"

    def get_required_package(self) -> str:
        return "dashscope"

    def check_dependency(self) -> None:
        """Verify dashscope and langchain_community are installed."""
        super().check_dependency()
        try:
            __import__("langchain_community.chat_models.tongyi")
            __import__("langchain_community.embeddings.dashscope")
        except ImportError as exc:
            raise ImportError(
                "Provider 'dashscope' requires langchain-community. "
                "Install it with:\n\n    pip install langchain-community\n"
            ) from exc

    # ------------------------------------------------------------------
    # Text LLM  (reads settings.llm)
    # ------------------------------------------------------------------

    def build_llm_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        cfg = settings.llm

        kwargs: dict[str, Any] = {
            "model": overrides.get("model", cfg.model),
            "top_p": 0.8,
            "model_kwargs": {
                "result_format": "message",
                "max_tokens": overrides.get("max_tokens", cfg.max_tokens),
                "temperature": overrides.get("temperature", cfg.temperature),
            },
        }

        api_key = (
            overrides.get("api_key")
            or cfg.api_key
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key

        return kwargs

    def create_llm_directly(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> BaseChatModel:
        """Create ChatTongyi directly (init_chat_model does not support tongyi)."""
        from langchain_community.chat_models.tongyi import ChatTongyi

        kwargs = self.build_llm_kwargs(settings, **overrides)
        model = kwargs.pop("model")
        return ChatTongyi(model=model, **kwargs)

    # ------------------------------------------------------------------
    # Vision LLM  (reads settings.vision_llm)
    # ------------------------------------------------------------------

    VISION_DEFAULT_TIMEOUT = 120

    def build_vision_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        cfg = settings.vision_llm

        kwargs: dict[str, Any] = {
            "model": overrides.get("model", cfg.model),
            "model_kwargs": {},
        }

        api_key = (
            overrides.get("api_key")
            or cfg.api_key
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key

        timeout = overrides.get("timeout") or getattr(cfg, "timeout", None)
        if timeout is not None:
            kwargs["model_kwargs"]["request_timeout"] = int(timeout)
        else:
            kwargs["model_kwargs"]["request_timeout"] = self.VISION_DEFAULT_TIMEOUT

        return kwargs

    def create_vision_llm_directly(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> BaseChatModel:
        """Create ChatTongyi for vision (qwen-vl-* models)."""
        from langchain_community.chat_models.tongyi import ChatTongyi

        kwargs = self.build_vision_kwargs(settings, **overrides)
        model = kwargs.pop("model")
        model_kwargs = kwargs.pop("model_kwargs", {})
        return ChatTongyi(model=model, model_kwargs=model_kwargs, **kwargs)

    # ------------------------------------------------------------------
    # Embedding  (reads settings.embedding)
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
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if api_key:
            kwargs["dashscope_api_key"] = api_key

        return kwargs

    def create_embedding_directly(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> Embeddings:
        """Create DashScopeEmbeddings directly (init_embeddings does not support dashscope)."""
        from langchain_community.embeddings.dashscope import DashScopeEmbeddings

        kwargs = self.build_embedding_kwargs(settings, **overrides)
        model = kwargs.pop("model")
        return DashScopeEmbeddings(model=model, **kwargs)
