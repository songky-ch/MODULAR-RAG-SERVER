"""Modular RAG MCP Server 的配置加载与校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 仓库根目录与路径解析
# ---------------------------------------------------------------------------
# 以当前文件位置为基准: <repo>/src/core/settings.py → parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")

# settings.yaml 的默认绝对路径
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """将仓库相对路径解析为绝对路径。

    如果 *relative* 已经是绝对路径，则直接返回；否则相对于
    :data:`REPO_ROOT` 进行解析。

    >>> resolve_path("config/settings.yaml")  # doctest: +SKIP
    PosixPath('/home/user/Modular-RAG-MCP-Server/config/settings.yaml')
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class SettingsError(ValueError):
    """配置校验失败时抛出。"""


def _require_mapping(data: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {path}.{key}")
    return value


def _require_value(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    return data[key]


def _require_str(data: Dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: Dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: Dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_list(data: Dict[str, Any], key: str, path: str) -> List[Any]:
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    # Azure/OpenAI 专用可选字段
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama 专用可选字段
    base_url: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    # Azure 专用可选字段
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama 专用可选字段
    base_url: Optional[str] = None


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    provider: str
    metrics: List[str]


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(frozen=True)
class VisionLLMSettings:
    enabled: bool
    provider: str
    model: str
    max_image_size: int
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[int] = None  # Request timeout in seconds (default used by client if not set)


@dataclass(frozen=True)
class GlmOcrSettings:
    api_key: Optional[str] = None
    use_zhipu_domain: bool = True
    download_images: bool = True
    api_mode: str = "file_parser"  # "file_parser" (multipart) or "layout_parsing" (base64)
    timeout: int = 300


@dataclass(frozen=True)
class PdfLoaderSettings:
    backend: str = "markitdown"  # "markitdown" | "glm_ocr"
    glm_ocr: Optional[GlmOcrSettings] = None


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int
    chunk_refiner: Optional[Dict[str, Any]] = None
    metadata_enricher: Optional[Dict[str, Any]] = None
    financial_splitter: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: Optional[IngestionSettings] = None
    vision_llm: Optional[VisionLLMSettings] = None
    pdf_loader: Optional[PdfLoaderSettings] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")

        ingestion_settings = None
        if "ingestion" in data:
            ingestion = _require_mapping(data, "ingestion", "settings")
            ingestion_settings = IngestionSettings(
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
                chunk_refiner=ingestion.get("chunk_refiner"),
                metadata_enricher=ingestion.get("metadata_enricher"),
                financial_splitter=ingestion.get("financial_splitter"),
            )

        vision_llm_settings = None
        if "vision_llm" in data:
            vision_llm = _require_mapping(data, "vision_llm", "settings")
            vision_llm_settings = VisionLLMSettings(
                enabled=_require_bool(vision_llm, "enabled", "vision_llm"),
                provider=_require_str(vision_llm, "provider", "vision_llm"),
                model=_require_str(vision_llm, "model", "vision_llm"),
                max_image_size=_require_int(vision_llm, "max_image_size", "vision_llm"),
                api_key=vision_llm.get("api_key"),
                api_version=vision_llm.get("api_version"),
                azure_endpoint=vision_llm.get("azure_endpoint"),
                deployment_name=vision_llm.get("deployment_name"),
                base_url=vision_llm.get("base_url"),
                timeout=vision_llm.get("timeout"),
            )

        pdf_loader_settings = None
        if "pdf_loader" in data:
            pdf_loader = _require_mapping(data, "pdf_loader", "settings")
            glm_ocr_settings = None
            if "glm_ocr" in pdf_loader:
                glm_ocr_raw = pdf_loader["glm_ocr"]
                if isinstance(glm_ocr_raw, dict):
                    glm_ocr_settings = GlmOcrSettings(
                        api_key=glm_ocr_raw.get("api_key") or None,
                        use_zhipu_domain=glm_ocr_raw.get("use_zhipu_domain", True),
                        download_images=glm_ocr_raw.get("download_images", True),
                        api_mode=glm_ocr_raw.get("api_mode", "file_parser"),
                        timeout=glm_ocr_raw.get("timeout", 300),
                    )
            pdf_loader_settings = PdfLoaderSettings(
                backend=pdf_loader.get("backend", "markitdown"),
                glm_ocr=glm_ocr_settings,
            )

        settings = cls(
            llm=LLMSettings(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                api_key=llm.get("api_key"),
                api_version=llm.get("api_version"),
                azure_endpoint=llm.get("azure_endpoint"),
                deployment_name=llm.get("deployment_name"),
                base_url=llm.get("base_url"),
            ),
            embedding=EmbeddingSettings(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                api_key=embedding.get("api_key"),
                api_version=embedding.get("api_version"),
                azure_endpoint=embedding.get("azure_endpoint"),
                deployment_name=embedding.get("deployment_name"),
                base_url=embedding.get("base_url"),
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(vector_store, "persist_directory", "vector_store"),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
            ),
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=_require_str(rerank, "provider", "rerank"),
                model=_require_str(rerank, "model", "rerank"),
                top_k=_require_int(rerank, "top_k", "rerank"),
                api_key=rerank.get("api_key"),
                base_url=rerank.get("base_url"),
            ),
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
            ),
            observability=ObservabilitySettings(
                log_level=_require_str(observability, "log_level", "observability"),
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                trace_file=_require_str(observability, "trace_file", "observability"),
                structured_logging=_require_bool(observability, "structured_logging", "observability"),
            ),
            ingestion=ingestion_settings,
            vision_llm=vision_llm_settings,
            pdf_loader=pdf_loader_settings,
        )

        return settings


def validate_settings(settings: Settings) -> None:
    """校验配置，配置无效时抛出 SettingsError。"""

    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    if not settings.retrieval.rrf_k:
        raise SettingsError("Missing required field: retrieval.rrf_k")
    if not settings.rerank.provider:
        raise SettingsError("Missing required field: rerank.provider")
    if not settings.evaluation.provider:
        raise SettingsError("Missing required field: evaluation.provider")
    if not settings.observability.log_level:
        raise SettingsError("Missing required field: observability.log_level")


def load_settings(path: str | Path | None = None) -> Settings:
    """从 YAML 文件加载配置并校验必填字段。

    Args:
        path: Path to settings YAML.  Defaults to
            ``<repo>/config/settings.yaml`` (absolute, CWD-independent).
    """
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    return settings
