"""Unit tests for SectionTableMetadataTransform."""

from unittest.mock import MagicMock

from src.core.types import Chunk
from src.ingestion.transform.section_table_metadata_transform import (
    SectionTableMetadataTransform,
)


class TestSectionTableMetadataTransform:
    def test_table_chunk_metadata(self) -> None:
        transform = SectionTableMetadataTransform()
        table_text = (
            "表1 主要财务指标\n"
            "| 年度 | 收入 |\n"
            "| ---- | ---- |\n"
            "| 2021 | 10 |\n"
        )
        chunks = [
            Chunk(id="c1", text=table_text, metadata={}, source_ref="doc1"),
        ]

        enriched = transform.transform(chunks, trace=None)
        assert enriched[0].metadata.get("is_table_chunk") is True
        assert "table_title" in enriched[0].metadata

    def test_section_title_metadata(self) -> None:
        transform = SectionTableMetadataTransform()
        text = "一、公司概况\n这里是公司概况的详细介绍。"
        chunks = [
            Chunk(id="c2", text=text, metadata={}, source_ref="doc1"),
        ]

        enriched = transform.transform(chunks, trace=None)
        assert "section_title" in enriched[0].metadata
        assert enriched[0].metadata["section_title"].startswith("一、")

