"""
Document ingestion endpoints.

  POST /ingest/text   — submit raw text
  POST /ingest/file   — upload a .txt / .md / .pdf file (generic)
  POST /ingest/pdf    — upload a PDF specifically; reports page count
                        and extracted character count for the UI
  GET  /ingest/stats  — collection size + embedding info
  DELETE /ingest      — wipe the collection (dev only)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.api.dependencies import (
    embedder_dep,
    ingestion_pipeline_dep,
    settings_dep,
    vector_store_dep,
)
from app.config import Settings
from app.models.schemas import (
    CollectionStatsResponse,
    IngestResponse,
    IngestTextRequest,
)
from app.rag.embeddings import BaseEmbedder
from app.rag.ingestion_pipeline import (
    SUPPORTED_EXTENSIONS,
    IngestionPipeline,
    IngestionReport,
)
from app.rag.vector_store import BaseVectorStore
from app.services.logging_service import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


# Hard cap on uploaded PDF size so the API can't be DOSed by a 1 GB upload.
_MAX_PDF_BYTES = 25 * 1024 * 1024  # 25 MB


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _to_response(
    report: IngestionReport,
    vector_store: BaseVectorStore,
    *,
    n_pages: Optional[int] = None,
    n_chars_extracted: Optional[int] = None,
) -> IngestResponse:
    return IngestResponse(
        success=report.success,
        source=report.source,
        n_chunks=report.n_chunks,
        embedding_dim=report.embedding_dim,
        embedding_model=report.embedding_model,
        stored_ids=report.stored_ids,
        skipped_reason=report.skipped_reason,
        collection_total=vector_store.count(),
        n_pages=n_pages,
        n_chars_extracted=n_chars_extracted,
    )


def _parse_metadata(raw: Optional[str]) -> Dict[str, Any]:
    """Parse the optional metadata_str form field. Empty -> {}."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"metadata_str is not valid JSON: {exc.msg}",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata_str must encode a JSON object.",
        )
    return parsed


def _extract_pdf_text(path: Path) -> tuple[str, int]:
    """Return (extracted_text, n_pages) using pypdf. Page-level extract
    failures are logged and skipped, never crash the request."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read PDF: {exc}",
        ) from exc

    if reader.is_encrypted:
        # Try empty password (some PDFs are encrypted with no password).
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Encrypted PDFs are not supported. Please remove the "
                    "password and try again."
                ),
            )

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdf_page_extract_failed", page=i, error=str(exc))
            pages.append("")
    return "\n\n".join(pages), len(reader.pages)


# --------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------- #
@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(
    payload: IngestTextRequest,
    pipeline: IngestionPipeline = Depends(ingestion_pipeline_dep),
    vector_store: BaseVectorStore = Depends(vector_store_dep),
) -> IngestResponse:
    """Ingest a raw text blob."""
    report = pipeline.ingest_text(
        payload.text,
        source=payload.source,
        extra_metadata=payload.metadata,
    )
    return _to_response(report, vector_store)


@router.post("/file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(
    file: UploadFile = File(...),
    source: str | None = Form(default=None),
    pipeline: IngestionPipeline = Depends(ingestion_pipeline_dep),
    vector_store: BaseVectorStore = Depends(vector_store_dep),
) -> IngestResponse:
    """Ingest an uploaded file (.txt, .md, .pdf) - generic path."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is missing a filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. "
                   f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        report = pipeline.ingest_file(tmp_path)
        report.source = source or file.filename
        return _to_response(report, vector_store)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("temp_file_cleanup_failed", path=str(tmp_path))


@router.post("/pdf", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(
    file: UploadFile = File(...),
    metadata_str: Optional[str] = Form(default=None),
    source: Optional[str] = Form(default=None),
    pipeline: IngestionPipeline = Depends(ingestion_pipeline_dep),
    vector_store: BaseVectorStore = Depends(vector_store_dep),
) -> IngestResponse:
    """
    Upload a PDF, extract its text, and feed it through the same
    chunking + embedding + vector-store path as plain text. The
    response includes ``n_pages`` and ``n_chars_extracted`` for UI
    feedback.

    Form fields:
      * ``file``         (required) - the PDF upload.
      * ``metadata_str`` (optional) - JSON object to attach to every
                          chunk's metadata.
      * ``source``       (optional) - override the auto-detected
                          source label (defaults to the filename).
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is missing a filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"This endpoint only accepts .pdf files (got '{ext}'). "
                "Use /ingest/file for other formats."
            ),
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF is empty.",
        )
    if len(contents) > _MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"PDF is {len(contents) / 1_000_000:.1f} MB; "
                f"max is {_MAX_PDF_BYTES // 1_000_000} MB."
            ),
        )

    metadata = _parse_metadata(metadata_str)
    metadata.setdefault("file_size_bytes", len(contents))
    metadata.setdefault("content_type", file.content_type or "application/pdf")
    metadata.setdefault("origin", "pdf_upload")

    label = source or file.filename

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        text, n_pages = _extract_pdf_text(tmp_path)
        n_chars = len(text)
        metadata.setdefault("n_pages", n_pages)

        if not text.strip():
            logger.warning(
                "pdf_no_text_extracted",
                source=label,
                n_pages=n_pages,
                file_size_bytes=len(contents),
            )
            empty_report = IngestionReport(
                source=label,
                n_chunks=0,
                embedding_dim=pipeline.embedder.dimension,
                embedding_model=pipeline.embedder.model_name,
                skipped_reason=(
                    "no_text_extracted (PDF may be scanned images; OCR is "
                    "not enabled)"
                ),
            )
            return _to_response(
                empty_report,
                vector_store,
                n_pages=n_pages,
                n_chars_extracted=0,
            )

        report = pipeline.ingest_text(
            text,
            source=label,
            extra_metadata=metadata,
        )
        logger.info(
            "pdf_ingested",
            source=label,
            n_pages=n_pages,
            n_chars=n_chars,
            n_chunks=report.n_chunks,
        )
        return _to_response(
            report,
            vector_store,
            n_pages=n_pages,
            n_chars_extracted=n_chars,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("temp_file_cleanup_failed", path=str(tmp_path))


@router.get("/stats", response_model=CollectionStatsResponse)
async def collection_stats(
    settings: Settings = Depends(settings_dep),
    vector_store: BaseVectorStore = Depends(vector_store_dep),
    embedder: BaseEmbedder = Depends(embedder_dep),
) -> CollectionStatsResponse:
    return CollectionStatsResponse(
        collection_name=settings.chroma_collection_name,
        total_chunks=vector_store.count(),
        embedding_model=embedder.model_name,
        embedding_dim=embedder.dimension,
    )


@router.delete("")
async def reset_collection(
    settings: Settings = Depends(settings_dep),
    vector_store: BaseVectorStore = Depends(vector_store_dep),
):
    """Wipe the entire collection. Dev/staging only."""
    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Collection reset is disabled in production.",
        )
    vector_store.reset()
    return {"reset": True, "collection_total": vector_store.count()}
