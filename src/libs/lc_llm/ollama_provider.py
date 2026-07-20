"""Ollama 提供商配置映射器。

Maps ``settings.yaml`` values into kwargs for ``ChatOllama`` /
``OllamaEmbeddings`` (local inference via the Ollama server).

Supports all three model types: LLM, Vision LLM, and Embedding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.lc_llm.base_provider import BaseProviderConfig

if TYPE_CHECKING:
    from src.core.settings import Settings


class OllamaProviderConfig(BaseProviderConfig):
    """将 ``settings.yaml`` 转换为 Ollama 模型的关键字参数。"""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def get_model_provider(self) -> str:
        return "ollama"

    def get_required_package(self) -> str:
        return "langchain-ollama"

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
            "num_predict": overrides.get("max_tokens", cfg.max_tokens),
        }

        base_url = (
            overrides.get("base_url")
            or cfg.base_url
            or self.DEFAULT_BASE_URL
        )
        kwargs["base_url"] = base_url

        return kwargs

    # ------------------------------------------------------------------
    # 视觉 LLM（读取 settings.vision_llm）
    # ------------------------------------------------------------------

    def build_vision_kwargs(
        self,
        settings: Settings,
        **overrides: Any,
    ) -> dict[str, Any]:
        cfg = settings.vision_llm

        kwargs: dict[str, Any] = {
            "model": overrides.get("model", cfg.model),
        }

        base_url = (
            overrides.get("base_url")
            or getattr(cfg, "base_url", None)
            or self.DEFAULT_BASE_URL
        )
        kwargs["base_url"] = base_url

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

        base_url = (
            overrides.get("base_url")
            or cfg.base_url
            or self.DEFAULT_BASE_URL
        )
        kwargs["base_url"] = base_url

        return kwargs
