"""LangChain 提供商配置映射器的抽象基类。

Each provider implements this interface to translate settings.yaml
configuration into kwargs for LangChain's ``init_chat_model`` /
``init_embeddings`` factory functions.

A single provider class covers **all three model types** (LLM, Vision
LLM, Embedding) so that adding a new provider only requires one file.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.settings import Settings


class BaseProviderConfig(ABC):
    """将 ``settings.yaml`` 配置值映射为 LangChain 工厂参数。

    Subclasses **must** implement:

    * :meth:`get_model_provider`
    * :meth:`get_required_package`
    * :meth:`build_llm_kwargs`

    And **may** override:

    * :meth:`build_vision_kwargs`  (defaults to ``NotImplementedError``)
    * :meth:`build_embedding_kwargs`  (defaults to ``NotImplementedError``)

    To add a new provider, subclass this and register via
    ``LCLLMFactory.register()``.

    Providers that are not supported by ``init_chat_model`` / ``init_embeddings``
    (e.g. DashScope) can set ``USES_DIRECT_INSTANTIATION = True`` and implement
    ``create_llm_directly``, ``create_vision_llm_directly``, ``create_embedding_directly``.
    """

    USES_DIRECT_INSTANTIATION: bool = False
    """为 True 时，工厂调用 create_*_directly()，而不是 init_chat_model。"""

    # ------------------------------------------------------------------
    # 必须实现的方法
    # ------------------------------------------------------------------

    @abstractmethod
    def get_model_provider(self) -> str:
        """返回 ``model_provider`` / ``provider`` 字符串。

        Examples: ``"openai"``, ``"ollama"``, ``"anthropic"``.
        """

    @abstractmethod
    def get_required_package(self) -> str:
        """返回当前提供商所需的 pip 包。

        Example: ``"langchain-openai"``.
        """

    @abstractmethod
    def build_llm_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        """构建 ``init_chat_model``（文本 LLM）的关键字参数。

        Reads from ``settings.llm``.
        The returned dict **must** contain a ``"model"`` key.
        """

    # ------------------------------------------------------------------
    # 可选实现（视觉 LLM / Embedding）
    # ------------------------------------------------------------------

    def build_vision_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        """构建 ``init_chat_model``（视觉 LLM）的关键字参数。

        Reads from ``settings.vision_llm``.
        The returned dict **must** contain a ``"model"`` key.

        Override in subclasses that support vision.
        """
        raise NotImplementedError(
            f"Provider '{self.get_model_provider()}' does not support Vision LLM. "
            f"Override build_vision_kwargs() to add support."
        )

    def build_embedding_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        """构建 ``init_embeddings`` 的关键字参数。

        Reads from ``settings.embedding``.
        The returned dict **must** contain a ``"model"`` key.

        Override in subclasses that support embeddings.
        """
        raise NotImplementedError(
            f"Provider '{self.get_model_provider()}' does not support Embedding. "
            f"Override build_embedding_kwargs() to add support."
        )

    # ------------------------------------------------------------------
    # 公共依赖检查
    # ------------------------------------------------------------------

    def check_dependency(self) -> None:
        """校验是否已安装当前提供商的集成包。

        Raises:
            ImportError: with a ``pip install`` hint when missing.
        """
        pkg = self.get_required_package()
        module_name = pkg.replace("-", "_")
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Provider '{self.get_model_provider()}' requires the "
                f"'{pkg}' package.  Install it with:\n\n"
                f"    pip install {pkg}\n"
            ) from exc
