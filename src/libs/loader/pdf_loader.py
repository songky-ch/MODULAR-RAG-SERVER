"""PDF Loader implementation using MarkItDown.

This module implements PDF parsing with image extraction support,
converting PDFs to standardized Markdown format with image placeholders.

Features:
- Text extraction and Markdown conversion via MarkItDown
- Image extraction and storage
- Image placeholder insertion with metadata tracking
- Graceful degradation if image extraction fails
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from PIL import Image
import io

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """PDF Loader using MarkItDown for text extraction and Markdown conversion.
    
    This loader:
    1. Extracts text from PDF and converts to Markdown
    2. Extracts images and saves to data/images/{doc_hash}/
    3. Inserts image placeholders in the format [IMAGE: {image_id}]
    4. Records image metadata in Document.metadata.images
    
    Configuration:
        extract_images: Enable/disable image extraction (default: True)
        image_storage_dir: Base directory for image storage (default: data/images)
    
    Graceful Degradation:
        If image extraction fails, logs warning and continues with text-only parsing.
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images"
    ):
        """Initialize PDF Loader.
        
        Args:
            extract_images: Whether to extract images from PDFs.
            image_storage_dir: Base directory for storing extracted images.
        """
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required for PdfLoader. "
                "Install with: pip install markitdown"
            )
        
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._markitdown = MarkItDown()
    
    def load(self, file_path: str | Path) -> Document:
        """Load and parse a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Document with Markdown text and metadata.
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If parsing fails critically.
        """
        # Validate file
        path = self._validate_file(file_path)
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {path}")
        
        # Compute document hash for unique ID and image directory
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"
        
        # Parse PDF with MarkItDown
        try:
            result = self._markitdown.convert(str(path))
            text_content = result.text_content if hasattr(result, 'text_content') else str(result)
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e
        
        # Initialize metadata
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
        }
        
        # Extract title from first heading if available
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title
        
        # Handle image extraction (with graceful degradation)
        if self.extract_images:
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    path, text_content, doc_hash
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(
                    f"Image extraction failed for {path}, continuing with text-only: {e}"
                )
        
        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Hex string of SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown heading or first non-empty line.
        
        Args:
            text: Markdown text content.
            
        Returns:
            Title string if found, None otherwise.
        """
        lines = text.split('\n')
        
        # First try to find a markdown heading
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fallback: use first non-empty line as title
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 0:
                return line
        
        return None
    
    def _estimate_page_boundaries(self, text_content: str, doc) -> List[int]:
        """Estimate where each page's content ends in the MarkItDown text.

        Uses proportional mapping: PyMuPDF extracts per-page character counts,
        then maps those proportions onto the MarkItDown output length to get
        approximate page-end positions.

        Returns:
            List of end-of-page character offsets (one per page).
        """
        page_count = len(doc)
        if page_count == 0:
            return []

        page_char_counts = [len(doc[i].get_text("text")) for i in range(page_count)]
        total = sum(page_char_counts) or 1
        text_len = len(text_content)

        boundaries: List[int] = []
        cumulative = 0
        for count in page_char_counts:
            cumulative += count
            boundaries.append(int((cumulative / total) * text_len))

        return boundaries

    def _extract_and_process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Extract images from PDF and insert placeholders near their source pages.

        Strategy:
        1. Estimate page boundary positions in the MarkItDown text via
           proportional per-page character counts from PyMuPDF.
        2. Extract all images with PyMuPDF and group them by page.
        3. Insert placeholder blocks at the estimated page boundaries in
           reverse page order so earlier offsets remain valid.

        Args:
            pdf_path: Path to PDF file.
            text_content: Extracted text content.
            doc_hash: Document hash for image directory.

        Returns:
            Tuple of (modified_text, images_metadata_list)
        """
        if not self.extract_images:
            logger.debug(f"Image extraction disabled for {pdf_path}")
            return text_content, []

        if not PYMUPDF_AVAILABLE:
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []

        images_metadata: List[Dict[str, Any]] = []
        modified_text = text_content

        try:
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)

            doc = fitz.open(pdf_path)

            page_boundaries = self._estimate_page_boundaries(modified_text, doc)

            # Phase 1: extract every image and group by page
            page_images_map: Dict[int, List[tuple]] = {}

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = image_dir / image_filename

                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            width, height = img.size
                        except Exception:
                            width, height = 0, 0

                        placeholder = f"[IMAGE: {image_id}]"

                        try:
                            relative_path = image_path.relative_to(Path.cwd())
                        except ValueError:
                            relative_path = image_path.absolute()

                        image_metadata: Dict[str, Any] = {
                            "id": image_id,
                            "path": str(relative_path),
                            "page": page_num + 1,
                            "text_length": len(placeholder),
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num + 1,
                                "index": img_index,
                            },
                        }

                        page_images_map.setdefault(page_num, []).append(
                            (placeholder, image_metadata)
                        )
                        images_metadata.append(image_metadata)

                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_index} from page {page_num + 1}: {e}"
                        )
                        continue

            doc.close()

            # Phase 2: insert placeholders at page boundaries (reverse order
            # so that earlier offsets are not invalidated by later insertions)
            for page_num in sorted(page_images_map.keys(), reverse=True):
                boundary = (
                    page_boundaries[page_num]
                    if page_num < len(page_boundaries)
                    else len(modified_text)
                )

                # Snap forward to the next line boundary for clean insertion
                newline_pos = modified_text.find("\n", boundary)
                insert_pos = (newline_pos + 1) if newline_pos != -1 else len(modified_text)

                placeholder_block = ""
                for placeholder_str, _ in page_images_map[page_num]:
                    placeholder_block += f"\n{placeholder_str}\n"

                modified_text = (
                    modified_text[:insert_pos]
                    + placeholder_block
                    + modified_text[insert_pos:]
                )

            # Phase 3: back-fill text_offset in metadata now that all
            # placeholders have their final positions
            for img_meta in images_metadata:
                placeholder = f"[IMAGE: {img_meta['id']}]"
                pos = modified_text.find(placeholder)
                if pos != -1:
                    img_meta["text_offset"] = pos

            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images from {pdf_path}")
            else:
                logger.debug(f"No images found in {pdf_path}")

            return modified_text, images_metadata

        except Exception as e:
            logger.warning(f"Image extraction failed for {pdf_path}: {e}")
            return text_content, []
    
    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """Generate unique image ID.
        
        Args:
            doc_hash: Document hash.
            page: Page number (0-based).
            sequence: Image sequence on page (0-based).
            
        Returns:
            Unique image ID string.
        """
        return f"{doc_hash[:8]}_{page}_{sequence}"
