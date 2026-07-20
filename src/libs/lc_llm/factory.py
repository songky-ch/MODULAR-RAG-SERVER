"""LangChain 模型工厂。

Provides ``LCLLMFactory`` — a single configuration-driven factory for
creating three kinds of LangChain model instances:

* **Text LLM** — ``BaseChatModel`` via ``init_chat_model``
* **Vision LLM** — ``BaseChatModel`` via ``init_chat_model``
  (same type; vision is a message-level feature in LangChain)
* **Embedding** — ``Embeddings`` via ``init_embeddings``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.chat_models.base import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.libs.lc_llm.base_provider import BaseProviderConfig

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class LCLLMFactory:
    """根据项目配置创建 LangChain 模型实例。

    A **single registry** holds provider configurations.  Each
    registered :class:`BaseProviderConfig` subclass can supply kwargs
    for LLM, Vision LLM, and/or Embedding — so one registration covers
    all model types a provider supports.

    Usage::

        from src.libs.lc_llm import LCLLMFactory
        from src.core.settings import load_settings

        settings = load_settings()

        llm        = LCLLMFactory.create(settings)
        vision_llm = LCLLMFactory.create_vision_llm(settings)
        embeddings = LCLLMFactory.create_embedding(settings)
    """

    _PROVIDERS: dict[str, type[BaseProviderConfig]] = {}

    # ------------------------------------------------------------------
    # 注册表管理
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        config_class: type[BaseProviderConfig],
    ) -> None:
        """注册提供商配置映射器。

        Args:
            name: Provider identifier (e.g. ``"openai"``, ``"ollama"``).
            config_class: A :class:`BaseProviderConfig` subclass.
        """
        if not (isinstance(config_class, type) and issubclass(config_class, BaseProviderConfig)):
            raise TypeError(
                f"{config_class!r} is not a BaseProviderConfig subclass"
            )
        cls._PROVIDERS[name.lower()] = config_class
        logger.debug("Registered LangChain provider: %s", name)

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回排序后的已注册提供商名称列表。"""
        return sorted(cls._PROVIDERS.keys())

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_config(cls, provider_name: str) -> BaseProviderConfig:
        """查找并实例化提供商配置，同时检查依赖。"""
        config_class = cls._PROVIDERS.get(provider_name)
        if config_class is None:
            raise ValueError(
                f"Unknown LangChain provider: '{provider_name}'. "
                f"Registered providers: {cls.list_providers()}"
            )
        config = config_class()
        config.check_dependency()
        return config

    # ------------------------------------------------------------------
    # 文本 LLM
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        settings: Settings,
        **overrides: Any,
    ) -> BaseChatModel:
        """创建用于文本生成的 ``BaseChatModel``。

        Reads provider/model from ``settings.llm``.

        Args:
            settings: The loaded application settings.
            **overrides: Values that take precedence over *settings*.
                A special ``"provider"`` key overrides
                ``settings.llm.provider``.

        Returns:
            A ready-to-use LangChain ``BaseChatModel``.
        """
        provider_name = (
            overrides.pop("provider", None)
            or settings.llm.provider.lower()
        )
        config = cls._resolve_config(provider_name)

        if getattr(config, "USES_DIRECT_INSTANTIATION", False):
            logger.info("Creating LLM (direct): provider=%s", provider_name)
            return config.create_llm_directly(settings, **overrides)

        kwargs = config.build_llm_kwargs(settings, **overrides)
        model = kwargs.pop("model")
        logger.info("Creating LLM: provider=%s, model=%s", provider_name, model)
        return init_chat_model(
            model,
            model_provider=config.get_model_provider(),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 视觉 LLM
    # ------------------------------------------------------------------

    @classmethod
    def create_vision_llm(
        cls,
        settings: Settings,
        **overrides: Any,
    ) -> BaseChatModel:
        """创建用于视觉（多模态）任务的 ``BaseChatModel``。

        Reads provider/model from ``settings.vision_llm``.
        In LangChain, vision is handled by the same ``BaseChatModel``
        — you simply include image content in the messages.

        Args:
            settings: The loaded application settings.
            **overrides: Values that take precedence over *settings*.
                A special ``"provider"`` key overrides
                ``settings.vision_llm.provider``.

        Returns:
            A ready-to-use LangChain ``BaseChatModel`` that accepts
            image content in messages.

        Raises:
            ValueError: If ``settings.vision_llm`` is not configured.
        """
        if not getattr(settings, "vision_llm", None):
            raise ValueError(
                "Vision LLM is not configured. "
                "Add a 'vision_llm' section to settings.yaml."
            )

        provider_name = (
            overrides.pop("provider", None)
            or settings.vision_llm.provider.lower()
        )
        config = cls._resolve_config(provider_name)

        if getattr(config, "USES_DIRECT_INSTANTIATION", False):
            logger.info("Creating Vision LLM (direct): provider=%s", provider_name)
            return config.create_vision_llm_directly(settings, **overrides)

        kwargs = config.build_vision_kwargs(settings, **overrides)
        model = kwargs.pop("model")
        logger.info("Creating Vision LLM: provider=%s, model=%s", provider_name, model)
        return init_chat_model(
            model,
            model_provider=config.get_model_provider(),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 向量模型
    # ------------------------------------------------------------------

    @classmethod
    def create_embedding(
        cls,
        settings: Settings,
        **overrides: Any,
    ) -> Embeddings:
        """创建 ``Embeddings`` 模型。

        Reads provider/model from ``settings.embedding``.

        Args:
            settings: The loaded application settings.
            **overrides: Values that take precedence over *settings*.
                A special ``"provider"`` key overrides
                ``settings.embedding.provider``.

        Returns:
            A ready-to-use LangChain ``Embeddings`` instance.
        """
        provider_name = (
            overrides.pop("provider", None)
            or settings.embedding.provider.lower()
        )
        config = cls._resolve_config(provider_name)

        if getattr(config, "USES_DIRECT_INSTANTIATION", False):
            logger.info("Creating Embedding (direct): provider=%s", provider_name)
            return config.create_embedding_directly(settings, **overrides)

        kwargs = config.build_embedding_kwargs(settings, **overrides)
        model = kwargs.pop("model")
        logger.info("Creating Embedding: provider=%s, model=%s", provider_name, model)
        return init_embeddings(
            model,
            provider=config.get_model_provider(),
            **kwargs,
        )
