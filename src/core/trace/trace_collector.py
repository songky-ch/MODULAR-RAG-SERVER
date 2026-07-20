"""追踪收集器——接收已完成的 TraceContext 并进行持久化。

收集器连接内存中的 TraceContext 对象和管理面板使用的磁盘 JSON Lines 日志。
它有意与日志模块解耦，使追踪持久化行为保持可预测、可测试。
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.core.settings import resolve_path
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# 追踪文件的默认绝对路径，不依赖当前工作目录
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceCollector:
    """收集已完成的追踪，并追加到 JSON Lines 文件。

    参数:
        traces_path: ``traces.jsonl`` 输出文件路径，父目录会自动创建。
    """

    def __init__(self, traces_path: str | Path = _DEFAULT_TRACES_PATH) -> None:
        self._path = Path(traces_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """将单条追踪持久化为一行 JSON。

        如果追踪尚未结束，则自动调用 ``finish()``，确保输出始终包含耗时数据。

        参数:
            trace: 已填充数据的 :class:`TraceContext`。
        """
        if trace.finished_at is None:
            trace.finish()

        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            logger.exception("Failed to write trace %s", trace.trace_id)

    @property
    def path(self) -> Path:
        """返回解析后的追踪文件路径。"""
        return self._path
