"""用于集成 TraceContext 的 LangChain 回调处理器。

连接 LangChain 回调系统与项目自建的 TraceContext，记录 LLM/Embedding
调用耗时和 Token 使用量。

Usage::

    from src.core.trace.langchain_callback import LangChainTraceCallback

    callback = LangChainTraceCallback(trace)
    llm.invoke(messages, config={"callbacks": [callback]})
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


class LangChainTraceCallback(BaseCallbackHandler):
    """将 LangChain LLM/Embedding 事件记录到 :class:`TraceContext`。

    每次 LLM 调用记录的数据:
      - 提供商和模型（来自 *serialized* 字典）
      - Prompt Token 数量（根据输入长度估算）
      - 补全 Token 数量和内容长度
      - 毫秒级耗时
      - 错误信息（如果存在）
    """

    def __init__(self, trace: TraceContext) -> None:
        super().__init__()
        self.trace = trace
        self._start_times: dict[UUID, float] = {}
        self._call_metadata: dict[UUID, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # LLM 生命周期
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.monotonic()
        self._call_metadata[run_id] = {
            "provider": serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown",
            "model": kwargs.get("invocation_params", {}).get("model", "unknown"),
            "prompt_chars": sum(len(p) for p in prompts),
        }

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.monotonic()
        total_chars = sum(
            len(m.content) if isinstance(m.content, str) else 0
            for batch in messages
            for m in batch
        )
        model_name = (
            kwargs.get("invocation_params", {}).get("model")
            or kwargs.get("invocation_params", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model", "unknown")
        )
        self._call_metadata[run_id] = {
            "provider": serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown",
            "model": model_name,
            "prompt_chars": total_chars,
            "message_count": sum(len(batch) for batch in messages),
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed_ms = self._elapsed(run_id)
        meta = self._call_metadata.pop(run_id, {})

        token_usage: Dict[str, Any] = {}
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage", {})
            if usage:
                token_usage = dict(usage)

        content_length = 0
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    content_length += len(gen.text) if gen.text else 0

        self.trace.record_stage("llm_call", {
            "provider": meta.get("provider", "unknown"),
            "model": meta.get("model", "unknown"),
            "prompt_chars": meta.get("prompt_chars", 0),
            "message_count": meta.get("message_count"),
            "completion_chars": content_length,
            "token_usage": token_usage or None,
        }, elapsed_ms=elapsed_ms)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed_ms = self._elapsed(run_id)
        meta = self._call_metadata.pop(run_id, {})

        self.trace.record_stage("llm_error", {
            "provider": meta.get("provider", "unknown"),
            "model": meta.get("model", "unknown"),
            "error": str(error),
        }, elapsed_ms=elapsed_ms)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _elapsed(self, run_id: UUID) -> float:
        start = self._start_times.pop(run_id, None)
        if start is None:
            return 0.0
        return (time.monotonic() - start) * 1000.0


def build_langchain_callbacks(
    trace: Optional[TraceContext] = None,
) -> List[BaseCallbackHandler]:
    """构建本地追踪回调和可选的自托管 Langfuse 回调。"""
    callbacks: List[BaseCallbackHandler] = []
    if trace is not None:
        callbacks.append(LangChainTraceCallback(trace))

    if os.environ.get("LANGFUSE_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return callbacks

    try:
        from langfuse.langchain import CallbackHandler

        callbacks.append(CallbackHandler())
    except Exception as exc:
        logging.getLogger(__name__).warning("Langfuse callback disabled: %s", exc)
    return callbacks
