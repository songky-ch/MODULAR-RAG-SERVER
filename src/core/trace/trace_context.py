"""用于观测各流水线阶段的追踪上下文。

提供 trace_id、trace_type（查询/入库）、各阶段耗时、finish() 生命周期，
以及用于输出 JSON Lines 的 to_dict() 序列化方法。
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TraceContext:
    """记录流水线阶段和耗时的请求级追踪上下文。

    属性:
        trace_id: 当前追踪的唯一标识。
        trace_type: ``"query"`` 或 ``"ingestion"``。
        started_at: 创建追踪时的 ISO-8601 时间戳。
        finished_at: 调用 ``finish()`` 时的 ISO-8601 时间戳，未调用时为 None。
        stages: 按顺序记录的阶段字典列表。
        metadata: 附加到追踪上的任意键值对。
    """

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = field(default=None)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 使用内部单调时钟精确计算耗时
    _start_mono: float = field(default_factory=time.monotonic, repr=False)
    _finish_mono: Optional[float] = field(default=None, repr=False)
    _stage_timings: Dict[str, float] = field(default_factory=dict, repr=False)

    # ---- 记录 --------------------------------------------------------

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """记录流水线某个阶段的数据。

        参数:
            stage_name: 阶段名称（例如 ``"dense_retrieval"``）。
            data: 阶段专用数据（方法、提供商、详情等）。
            elapsed_ms: 预先计算的毫秒耗时。如果为 *None*，调用方应自行计时，
                或交给 ``stage_timer`` 上下文管理器处理。
        """
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    # ---- 生命周期 ----------------------------------------------------

    def finish(self) -> None:
        """将追踪标记为已完成，并记录结束时间。"""
        self._finish_mono = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    # ---- 计时辅助方法 ------------------------------------------------

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """返回毫秒级耗时。

        参数:
            stage_name: 如果指定，则返回该阶段记录的耗时。如果为 *None*，
                返回整个追踪的总耗时（开始到结束；尚未结束时为开始到当前时间）。

        返回:
            毫秒级耗时。

        异常:
            KeyError: 指定了 *stage_name* 但未找到对应阶段。
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"Stage '{stage_name}' has no recorded timing")
            return self._stage_timings[stage_name]

        end = self._finish_mono if self._finish_mono is not None else time.monotonic()
        return (end - self._start_mono) * 1000.0

    # ---- 序列化 ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """将追踪序列化为适合 ``json.dumps`` 的普通字典。

        返回:
            包含全部追踪数据的字典。
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round(self.elapsed_ms(), 2),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }

    # ---- C5/C6 使用的向后兼容辅助方法 --------------------------------

    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """获取指定阶段记录的数据。

        搜索阶段列表；存在重复名称时，以最后一次记录为准。

        参数:
            stage_name: 要获取的阶段名称。

        返回:
            匹配阶段的 ``data`` 字典，未找到时返回 *None*。
        """
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None
