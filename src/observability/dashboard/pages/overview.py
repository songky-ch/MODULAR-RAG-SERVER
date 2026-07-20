"""Overview page – system configuration and data statistics.

Displays:
- Component configuration cards (LLM, Embedding, VectorStore …)
- Collection statistics (document count, chunk count, image count)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


def _safe_collection_stats() -> Dict[str, Any]:
    """Attempt to load collection statistics from ChromaDB.

    Returns empty dict on failure so the page still renders.
    """
    try:
        from src.core.settings import load_settings, resolve_path
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = load_settings()
        persist_dir = str(
            resolve_path(settings.vector_store.persist_directory)
        )
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        stats: Dict[str, Any] = {}
        for col in client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            collection = client.get_collection(name)
            stats[name] = {"chunk_count": collection.count()}
        return stats
    except Exception:
        return {}


def render() -> None:
    """Render the Overview page."""
    st.header("📊 系统总览")

    # ── Component configuration cards ──────────────────────────────
    st.subheader("🔧 组件配置")

    try:
        config_service = ConfigService()
        cards = config_service.get_component_cards()
    except Exception as exc:
        st.error(f"加载配置失败: {exc}")
        return

    cols = st.columns(min(len(cards), 3))
    component_names = {
        "Embedding": "向量模型",
        "Vector Store": "向量数据库",
        "Retrieval": "混合检索",
        "Reranker": "重排模型",
        "Vision LLM": "视觉模型",
        "Ingestion": "文档处理",
    }
    detail_names = {
        "temperature": "温度",
        "max_tokens": "最大 Token 数",
        "dimensions": "向量维度",
        "persist_directory": "存储目录",
        "dense_top_k": "稠密检索 Top-K",
        "sparse_top_k": "稀疏检索 Top-K",
        "fusion_top_k": "融合 Top-K",
        "enabled": "是否启用",
        "top_k": "Top-K",
        "max_image_size": "最大图片尺寸",
        "chunk_size": "分块大小",
        "chunk_overlap": "分块重叠",
        "batch_size": "批处理大小",
    }
    for idx, card in enumerate(cards):
        with cols[idx % len(cols)]:
            st.markdown(f"**{component_names.get(card.name, card.name)}**")
            st.caption(f"提供商: `{card.provider}`  \n模型: `{card.model}`")
            with st.expander("详细信息"):
                for k, v in card.extra.items():
                    st.text(f"{detail_names.get(k, k)}: {v}")

    # ── Collection statistics ──────────────────────────────────────
    st.subheader("📁 知识库统计")

    stats = _safe_collection_stats()
    if stats:
        stat_cols = st.columns(min(len(stats), 4))
        for idx, (name, info) in enumerate(sorted(stats.items())):
            with stat_cols[idx % len(stat_cols)]:
                count = info.get("chunk_count", "?")
                st.metric(label=name, value=count)
                if count == 0 or count == "?":
                    st.caption("⚠️ 空")
    else:
        st.warning(
            "**未找到知识库或 ChromaDB 不可用。** "
            "请前往“文档入库”页面上传并处理文档。"
        )

    # ── Trace file statistics ──────────────────────────────────────
    st.subheader("📈 追踪统计")

    from src.core.settings import resolve_path
    traces_path = resolve_path("logs/traces.jsonl")
    if traces_path.exists():
        line_count = sum(1 for _ in traces_path.open(encoding="utf-8"))
        if line_count > 0:
            st.metric("追踪总数", line_count)
        else:
            st.info("尚无追踪记录，请先执行查询或文档入库。")
    else:
        st.info("尚无追踪记录，请先执行查询或文档入库。")
