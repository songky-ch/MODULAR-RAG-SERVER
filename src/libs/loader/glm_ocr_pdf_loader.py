"""PDF Loader using Zhipu online APIs for layout-aware parsing.

Uses the ``file_parser`` API (multipart upload) by default, which avoids
base64 encoding issues that can cause 400 errors with layout_parsing.
Falls back to ``layout_parsing`` (base64) if file_parser is unavailable.

Both return Markdown text that preserves tables, formulas and images.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from zai import ZaiClient, ZhipuAiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# API limits
PDF_MAX_SIZE_LAYOUT_PARSING = 50 * 1024 * 1024  # 50 MB (layout_parsing)
PDF_MAX_SIZE_FILE_PARSER = 100 * 1024 * 1024   # 100 MB (file_parser Prime)
PDF_MAX_PAGES = 100

from src.core.settings import resolve_path
from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class GlmOcrPdfLoader(BaseLoader):
    """PDF → Markdown loader backed by the Zhipu GLM-OCR layout parsing API.

    The returned :class:`Document` contains:

    * ``text`` – full Markdown with tables / formulas / image references.
    * ``metadata["engine"]`` – ``"glm-ocr"``
    * ``metadata["images"]`` – list of image metadata dicts (when images are
      downloaded to local storage).
    * ``metadata["layout_details"]`` – raw per-page layout detail list.
    * ``metadata["data_info"]`` – page count / dimensions.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        use_zhipu_domain: bool = True,
        download_images: bool = True,
        image_storage_dir: str | Path = "data/images",
        markdown_output_dir: Optional[str | Path] = None,
        api_mode: str = "file_parser",  # "file_parser" (multipart) or "layout_parsing" (base64)
        start_page_id: Optional[int] = None,
        end_page_id: Optional[int] = None,
        timeout: float = 300.0,
    ) -> None:
        if not ZAI_AVAILABLE:
            raise ImportError(
                "zai-sdk is required for GlmOcrPdfLoader. "
                "Install with: pip install zai-sdk"
            )

        if use_zhipu_domain:
            self._client = ZhipuAiClient(api_key=api_key)
        else:
            self._client = ZaiClient(api_key=api_key)

        self.download_images = download_images
        self.image_storage_dir = Path(image_storage_dir)
        self.markdown_output_dir = Path(markdown_output_dir) if markdown_output_dir else None
        self.api_mode = api_mode
        self.start_page_id = start_page_id
        self.end_page_id = end_page_id
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"
        file_size = path.stat().st_size

        # Pre-validate API limits
        max_size = (
            PDF_MAX_SIZE_FILE_PARSER
            if self.api_mode == "file_parser"
            else PDF_MAX_SIZE_LAYOUT_PARSING
        )
        if file_size > max_size:
            raise ValueError(
                f"PDF exceeds limit ({max_size // (1024*1024)}MB): {path.name} is "
                f"{file_size / (1024 * 1024):.1f} MB"
            )
        if PYMUPDF_AVAILABLE:
            with fitz.open(path) as doc:
                page_count = len(doc)
            if page_count > PDF_MAX_PAGES:
                raise ValueError(
                    f"PDF exceeds page limit (100): {path.name} has {page_count} pages"
                )

        md_text: str
        layout_details_serialised: Optional[List[List[Dict[str, Any]]]] = None
        data_info_serialised: Optional[Dict[str, Any]] = None
        request_id: Optional[str] = None
        engine = "glm-ocr"

        # Prefer file_parser (multipart upload) to avoid base64 400 errors
        parsing_result_url: Optional[str] = None
        if self.api_mode == "file_parser":
            md_text, layout_details_serialised, data_info_serialised, request_id, parsing_result_url = (
                self._load_via_file_parser(path)
            )
            engine = "file_parser"
        else:
            md_text, layout_details_serialised, data_info_serialised, request_id = (
                self._load_via_layout_parsing(path)
            )
            parsing_result_url = None

        logger.info(f"Parsed {path.name}: {len(md_text)} chars of Markdown")

        # Build metadata
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "engine": engine,
            "data_info": data_info_serialised,
            "request_id": request_id,
        }

        title = self._extract_title(md_text)
        if title:
            metadata["title"] = title

        # Download images: layout_parsing uses layout_details; file_parser uses ZIP from parsing_result_url
        images_metadata: List[Dict[str, Any]] = []
        md_output_dir: Optional[Path] = None
        if self.download_images:
            if layout_details_serialised:
                md_text, images_metadata = self._download_and_rewrite_images(
                    md_text, layout_details_serialised, doc_hash
                )
            elif parsing_result_url:
                md_text, images_metadata, md_output_dir = self._download_images_from_zip(
                    md_text, parsing_result_url, path, doc_hash
                )

        if images_metadata:
            metadata["images"] = images_metadata

        if layout_details_serialised:
            metadata["layout_details"] = layout_details_serialised

        md_path = self._save_markdown(path, md_text, doc_hash, md_output_dir)
        if md_path:
            metadata["markdown_path"] = str(md_path)

        return Document(id=doc_id, text=md_text, metadata=metadata)

    def _load_via_file_parser(
        self, path: Path
    ) -> Tuple[str, Optional[List[List[Dict[str, Any]]]], Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """Use file_parser API (multipart upload) - avoids base64 400 errors."""
        logger.info(f"Calling file_parser (prime-sync) for {path.name} …")
        with path.open("rb") as f:
            resp = self._client.file_parser.create_sync(
                file=f,
                file_type="pdf",
                tool_type="prime-sync",
                timeout=self.timeout,
            )
        md_text = getattr(resp, "content", None) or ""
        task_id = getattr(resp, "task_id", None)
        parsing_result_url = getattr(resp, "parsing_result_url", None) or None
        return md_text, None, None, task_id, parsing_result_url

    def _load_via_layout_parsing(
        self, path: Path
    ) -> Tuple[str, Optional[List[List[Dict[str, Any]]]], Optional[Dict[str, Any]], Optional[str]]:
        """Use layout_parsing API (base64 in JSON)."""
        pdf_bytes = path.read_bytes()
        b64_str = base64.b64encode(pdf_bytes).decode("utf-8")
        data_uri = f"data:application/pdf;base64,{b64_str}"

        logger.info(f"Calling layout_parsing (glm-ocr) for {path.name} …")
        kwargs: Dict[str, Any] = {
            "model": "glm-ocr",
            "file": data_uri,
            "timeout": self.timeout,
        }
        if self.start_page_id is not None:
            kwargs["start_page_id"] = self.start_page_id
        if self.end_page_id is not None:
            kwargs["end_page_id"] = self.end_page_id

        resp = self._client.layout_parsing.create(**kwargs)
        md_text = resp.md_results or ""

        layout_details_serialised: Optional[List[List[Dict[str, Any]]]] = None
        if resp.layout_details:
            layout_details_serialised = [
                [item.model_dump() if hasattr(item, "model_dump") else item for item in page]
                for page in resp.layout_details
            ]

        data_info_serialised: Optional[Dict[str, Any]] = None
        if resp.data_info:
            data_info_serialised = (
                resp.data_info.model_dump()
                if hasattr(resp.data_info, "model_dump")
                else resp.data_info
            )

        return md_text, layout_details_serialised, data_info_serialised, getattr(resp, "request_id", None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_markdown(
        self,
        pdf_path: Path,
        md_text: str,
        doc_hash: str,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Persist the converted Markdown.

        If *output_dir* is given (e.g. from ZIP extraction), save there.
        Else if *markdown_output_dir* was provided, use it.
        Otherwise save next to the PDF or to data/markdown/{doc_hash}/ for persistence.
        """
        try:
            if output_dir is not None:
                out_dir = output_dir
            elif self.markdown_output_dir:
                out_dir = Path(self.markdown_output_dir)
            else:
                # Prefer persistent location; temp PDF paths get cleaned up
                out_dir = resolve_path("data/markdown") / doc_hash

            out_dir.mkdir(parents=True, exist_ok=True)
            md_path = out_dir / f"{pdf_path.stem}.md"
            md_path.write_text(md_text, encoding="utf-8")
            logger.info(f"Saved Markdown → {md_path}")
            return md_path
        except Exception as e:
            logger.warning(f"Failed to save Markdown file: {e}")
            return None

    def _download_images_from_zip(
        self,
        md_text: str,
        zip_url: str,
        pdf_path: Path,
        doc_hash: str,
    ) -> Tuple[str, List[Dict[str, Any]], Path]:
        """Download ZIP from parsing_result_url, extract images to output_dir/images/.

        The Markdown references images as images/xxx.png. We extract so that
        output_dir/images/xxx.png exists, making the relative path valid.

        Returns (md_text, images_metadata, output_dir).
        """
        # Output dir: persistent, so markdown + images survive temp PDF cleanup
        if self.markdown_output_dir:
            output_dir = Path(self.markdown_output_dir) / doc_hash
        else:
            output_dir = resolve_path("data/markdown") / doc_hash
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images_metadata: List[Dict[str, Any]] = []
        try:
            r = httpx.get(zip_url, timeout=60.0, follow_redirects=True)
            r.raise_for_status()
            zip_bytes = r.content
        except Exception as e:
            logger.warning(f"Failed to download parsing result ZIP: {e}")
            return md_text, images_metadata, output_dir

        try:
            with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    lower = name.lower()
                    if not (
                        lower.endswith(".png")
                        or lower.endswith(".jpg")
                        or lower.endswith(".jpeg")
                        or lower.endswith(".gif")
                        or lower.endswith(".webp")
                        or lower.endswith(".bmp")
                    ):
                        continue
                    try:
                        data = zf.read(name)
                        base_name = Path(name).name
                        dest = images_dir / base_name
                        dest.write_bytes(data)

                        try:
                            rel_path = dest.relative_to(Path.cwd())
                        except ValueError:
                            rel_path = dest.absolute()

                        img_id = base_name.rsplit(".", 1)[0].replace(".", "_")
                        images_metadata.append({
                            "id": img_id,
                            "path": str(rel_path),
                            "page": 0,
                        })
                        logger.debug(f"Extracted image → {dest}")
                    except Exception as e:
                        logger.warning(f"Failed to extract {name}: {e}")

            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images to {images_dir}")
        except zipfile.BadZipFile as e:
            logger.warning(f"Invalid ZIP from parsing_result_url: {e}")
        except Exception as e:
            logger.warning(f"Failed to extract parsing result ZIP: {e}")

        return md_text, images_metadata, output_dir

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        for line in text.split("\n")[:20]:
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        for line in text.split("\n")[:10]:
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    def _download_and_rewrite_images(
        self,
        md_text: str,
        layout_details: List[List[Dict[str, Any]]],
        doc_hash: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Download remote images referenced in *layout_details* and rewrite
        the Markdown so it points to local file paths instead of URLs.

        Returns the rewritten Markdown and a list of image metadata dicts
        compatible with the existing pipeline (keys: ``id``, ``path``,
        ``page``).
        """
        image_dir = self.image_storage_dir / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)

        images_meta: List[Dict[str, Any]] = []
        url_to_local: Dict[str, str] = {}
        img_seq = 0

        for page_idx, page_items in enumerate(layout_details):
            for item in page_items:
                if item.get("label") != "image":
                    continue
                url = item.get("content")
                if not url or url in url_to_local:
                    continue

                img_seq += 1
                image_id = f"{doc_hash[:8]}_{page_idx + 1}_{img_seq}"
                ext = self._guess_extension(url)
                local_name = f"{image_id}.{ext}"
                local_path = image_dir / local_name

                try:
                    r = httpx.get(url, timeout=30.0, follow_redirects=True)
                    r.raise_for_status()
                    local_path.write_bytes(r.content)

                    try:
                        relative = local_path.relative_to(Path.cwd())
                    except ValueError:
                        relative = local_path.absolute()

                    url_to_local[url] = str(relative)
                    images_meta.append({
                        "id": image_id,
                        "path": str(relative),
                        "page": page_idx + 1,
                        "bbox_2d": item.get("bbox_2d"),
                    })
                    logger.debug(f"Downloaded image → {local_path}")
                except Exception as e:
                    logger.warning(f"Failed to download image {url}: {e}")

        # Rewrite Markdown: replace remote URLs with local paths
        for url, local in url_to_local.items():
            md_text = md_text.replace(url, local)

        if images_meta:
            logger.info(f"Downloaded {len(images_meta)} images to {image_dir}")

        return md_text, images_meta

    @staticmethod
    def _guess_extension(url: str) -> str:
        """Best-effort extension from URL path."""
        lower = url.lower().split("?")[0]
        for ext in ("png", "jpg", "jpeg", "gif", "bmp", "webp", "svg"):
            if lower.endswith(f".{ext}"):
                return ext
        return "png"
