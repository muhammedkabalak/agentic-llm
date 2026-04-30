"""
End-to-end document ingestion pipeline.

Pipeline stages:
    load (file or raw text)  →  chunk  →  embed  →  store

Supported file types: .txt, .md, .pdf
PDF parsing uses pypdf (lightweight, pure-Python).

Usage:
    pipeline = IngestionPipeline(chunker, embedder, vector_store)
    report = pipeline.ingest_text("...", source="manual_upload")
    report = pipeline.ingest_file(Path("docs/paper.pdf"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from app.models.domain import DocumentChunk
from app.rag.chunking import ChunkingConfig, TextChunker
from app.rag.embeddings import BaseEmbedder
from app.rag.vector_store import BaseVectorStore
from app.services.logging_service import get_logger

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class IngestionReport:
    """Per-ingestion summary returned to the API caller."""
    source: str
    n_chunks: int
    embedding_dim: int
    embedding_model: str
    stored_ids: List[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.n_chunks > 0 and self.skipped_reason is None


class IngestionPipeline:
    """Composes a chunker, embedder, and vector store into one pipeline."""

    def __init__(
        self,
        chunker: TextChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    @classmethod
    def from_components(
        cls,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        chunking_config: Optional[ChunkingConfig] = None,
    ) -> "IngestionPipeline":
        return cls(
            chunker=TextChunker(chunking_config),
            embedder=embedder,
            vector_store=vector_store,
        )

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_file(path: Path) -> str:
        """Read a file and return its raw text."""
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".pdf":
            return IngestionPipeline._load_pdf(path)
        # .txt, .md
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _load_pdf(path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("pdf_page_extract_failed", page=i, error=str(exc))
                pages.append("")
        return "\n\n".join(pages)

    # ------------------------------------------------------------------ #
    # Public ingestion methods
    # ------------------------------------------------------------------ #
    def ingest_text(
        self,
        text: str,
        *,
        source: str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> IngestionReport:
        """Chunk + embed + store raw text."""
        if not text or not text.strip():
            return IngestionReport(
                source=source,
                n_chunks=0,
                embedding_dim=self.embedder.dimension,
                embedding_model=self.embedder.model_name,
                skipped_reason="empty_text",
            )

        chunks = self.chunker.chunk_text(
            text, source=source, extra_metadata=extra_metadata
        )
        return self._embed_and_store(chunks, source=source)

    def ingest_file(
        self,
        path: Path,
        *,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> IngestionReport:
        """Load + chunk + embed + store a file from disk."""
        text = self.load_file(path)
        return self.ingest_text(
            text, source=str(path.name), extra_metadata=extra_metadata
        )

    def ingest_batch(
        self,
        documents: List[tuple[str, str]],
        *,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> List[IngestionReport]:
        """Ingest many `(source, text)` pairs."""
        return [
            self.ingest_text(text, source=src, extra_metadata=extra_metadata)
            for src, text in documents
        ]

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _embed_and_store(
        self, chunks: List[DocumentChunk], *, source: str
    ) -> IngestionReport:
        if not chunks:
            return IngestionReport(
                source=source,
                n_chunks=0,
                embedding_dim=self.embedder.dimension,
                embedding_model=self.embedder.model_name,
                skipped_reason="no_chunks_produced",
            )

        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        ids = self.vector_store.add(chunks, embeddings)

        report = IngestionReport(
            source=source,
            n_chunks=len(chunks),
            embedding_dim=self.embedder.dimension,
            embedding_model=self.embedder.model_name,
            stored_ids=ids,
        )
        logger.info(
            "ingestion_done",
            source=source,
            n_chunks=report.n_chunks,
            embedding_model=report.embedding_model,
            embedding_dim=report.embedding_dim,
        )
        return report
