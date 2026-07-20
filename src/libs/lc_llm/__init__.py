"""LangChain-based model module.

Provides a unified, configuration-driven factory for creating three
kinds of LangChain model instances:

* **Text LLM** — ``LCLLMFactory.create(settings)``
* **Vision LLM** — ``LCLLMFactory.create_vision_llm(settings)``
* **Embedding** — ``LCLLMFactory.create_embedding(settings)``

Coexists with the legacy ``src.libs.llm`` module — callers can choose
either.

Quick start::

    from src.libs.lc_llm import LCLLMFactory
    from src.core.settings import load_settings
    from langchain_core.messages import HumanMessage

    settings = load_settings()

    llm = LCLLMFactory.create(settings)
    response = llm.invoke([HumanMessage(content="Hello")])

    embeddings = LCLLMFactory.create_embedding(settings)
    vector = embeddings.embed_query("Hello")
"""

from src.libs.lc_llm.base_provider import BaseProviderConfig
from src.libs.lc_llm.dashscope_provider import DashScopeProviderConfig
from src.libs.lc_llm.factory import LCLLMFactory
from src.libs.lc_llm.mimo_provider import MiMoProviderConfig
from src.libs.lc_llm.ollama_provider import OllamaProviderConfig
from src.libs.lc_llm.openai_provider import OpenAIProviderConfig

LCLLMFactory.register("openai", OpenAIProviderConfig)
LCLLMFactory.register("ollama", OllamaProviderConfig)
LCLLMFactory.register("dashscope", DashScopeProviderConfig)
LCLLMFactory.register("qwen", DashScopeProviderConfig)
LCLLMFactory.register("mimo", MiMoProviderConfig)

__all__ = [
    "LCLLMFactory",
    "BaseProviderConfig",
    "DashScopeProviderConfig",
    "MiMoProviderConfig",
    "OpenAIProviderConfig",
    "OllamaProviderConfig",
]
