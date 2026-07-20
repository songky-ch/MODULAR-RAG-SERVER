"""Tests for directly ingesting a pre-converted Markdown file.

This test bypasses the PDF → Markdown GLM OCR loader step and feeds an
existing Markdown report file into the ingestion pipeline.
"""

from pathlib import Path

from src.core.settings import load_settings
from src.core.types import Document
from src.ingestion.pipeline import IngestionPipeline, PipelineResult


MARKDOWN_PATH = Path(
    "data/markdown/3f59624725f0d79a1894d36dc727c9fedc7cfeb7cbcc11d257b1c08f6b42a00f/tmpbiafcspt.md"
)


class MarkdownFileLoader:
    """Simple loader that wraps an existing Markdown file as a Document."""

    def load(self, file_path: str) -> Document:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        # Use first non-empty line as a best-effort title
        title = None
        for line in text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                title = stripped
                break

        metadata = {
            "source_path": str(path),
            "doc_type": "markdown",
        }
        if title:
            metadata["title"] = title

        return Document(
            id=f"doc_markdown_{path.stem}",
            text=text,
            metadata=metadata,
        )


class TestMarkdownDirectIngestion:
    """End-to-end ingestion of a Markdown financial report without GLM OCR."""

    def test_markdown_file_can_be_ingested_directly(self) -> None:
        assert MARKDOWN_PATH.exists(), f"Markdown file not found: {MARKDOWN_PATH}"

        # Load main settings (includes financial splitter etc.)
        settings = load_settings("config/settings.yaml")

        # Force re-processing to avoid integrity skip
        pipeline = IngestionPipeline(settings, collection="financial_markdown", force=True)

        # Replace PDF/GLM loader with our Markdown loader
        pipeline.loader = MarkdownFileLoader()
        pipeline.loader_backend = "markdown_direct"

        result: PipelineResult = pipeline.run(str(MARKDOWN_PATH))

        assert result.success is True
        assert result.chunk_count > 0
        # For a pure Markdown file we do not expect images
        assert result.image_count == 0
        # Vectors should have been written for all chunks
        assert len(result.vector_ids) == result.chunk_count

