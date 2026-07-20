"""Data Browser page – browse ingested documents, chunks, and images.

Layout:
1. Collection selector (sidebar)
2. Document list with chunk counts
3. Expandable document detail → chunk cards with text + metadata
4. Image preview gallery
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render() -> None:
    """Render the Data Browser page."""
    st.header("🔍 数据浏览")

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"初始化数据服务失败: {exc}")
        return

    # ── Collection selector ────────────────────────────────────────
    collections = svc.list_collections()
    if "default" not in collections:
        collections.insert(0, "default")
    collection = st.selectbox(
        "知识库",
        options=collections,
        index=0,
        key="db_collection_filter",
    )
    coll_arg = collection if collection else None

    # ── Danger zone: clear all data ────────────────────────────────
    st.divider()
    with st.expander("⚠️ 危险操作", expanded=False):
        st.warning(
            "此操作将**永久删除**所有数据，包括 ChromaDB 知识库、BM25 索引、"
            "图片、入库历史和追踪日志。"
        )
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            if st.button("🗑️ 清空所有数据", type="primary", key="btn_clear_all"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error("确定要继续吗？此操作无法撤销！")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("✅ 确认全部删除", key="btn_confirm_clear"):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(
                            f"清理完成，但出现 {len(result['errors'])} 个错误: "
                            + "; ".join(result["errors"])
                        )
                    else:
                        st.success(
                            f"所有数据已清空，共删除 "
                            f"{result['collections_deleted']} 个知识库。"
                        )
                    st.rerun()
            with c2:
                if st.button("❌ 取消", key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    # ── Document list ──────────────────────────────────────────────
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(f"加载文档失败: {exc}")
        return

    if not docs:
        st.info(
            "**当前知识库中没有文档。** "
            "请前往“文档入库”页面上传文件，或从上方选择其他知识库。"
        )
        return

    st.subheader(f"📄 文档 ({len(docs)})")

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = f"📑 {source_name}  —  {doc['chunk_count']} 个分块 · {doc['image_count']} 张图片"
        with st.expander(label, expanded=(len(docs) == 1)):
            # ── Document metadata ──────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("分块", doc["chunk_count"])
            col_b.metric("图片", doc["image_count"])
            col_c.metric("知识库", doc.get("collection", "—"))
            st.caption(
                f"**来源:** {doc['source_path']}  ·  "
                f"**哈希:** `{doc['source_hash'][:16]}…`  ·  "
                f"**处理时间:** {doc.get('processed_at', '—')}"
            )

            st.divider()

            # ── Chunk cards ────────────────────────────────────────
            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(f"### 📦 文本分块 ({len(chunks)})")
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]

                    # Title from metadata or first line
                    title = meta.get("title", "")
                    if not title:
                        title = text[:60].replace("\n", " ").strip()
                        if len(text) > 60:
                            title += "…"

                    with st.container(border=True):
                        st.markdown(
                            f"**分块 {cidx + 1}** · `{chunk_id[-16:]}` · "
                            f"{len(text)} 个字符"
                        )
                        # Show the actual chunk text (scrollable)
                        _height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            "内容",
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        # Expandable metadata
                        with st.expander("📋 元数据", expanded=False):
                            st.json(meta)
            else:
                st.caption("向量库中未找到此文档的分块。")

            # ── Image preview ──────────────────────────────────────
            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(f"### 🖼️ 图片 ({len(images)})")
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], width=200)
                        else:
                            st.caption(f"{img['image_id']}（文件缺失）")
