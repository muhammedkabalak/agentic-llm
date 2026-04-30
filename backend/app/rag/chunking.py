"""
Document chunking strategies.

Splits raw text into smaller, overlapping chunks suitable for
embedding and retrieval. Uses LangChain's `RecursiveCharacterTextSplitter`
under the hood, which respects natural boundaries (paragraphs, sentences,
words) before falling back to character-level splits.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

from app.models.domain import DocumentChunk
from app.services.logging_service import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    """Tunable chunking parameters."""
    chunk_size: int = 800            # characters per chunk
    chunk_overlap: int = 100         # characters of overlap between chunks
    separators: Optional[List[str]] = None  # split priority

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


class TextChunker:
    """Splits text into `DocumentChunk` objects."""

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
        # Lazy import — keep startup light
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )

    def chunk_text(
        self,
        text: str,
        *,
        source: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split a single document into chunks."""
        if not text or not text.strip():
            return []

        raw_chunks = self._splitter.split_text(text)
        # Deterministic document ID — same text + source = same id
        doc_id = hashlib.sha256(
            f"{source or ''}::{text}".encode("utf-8")
        ).hexdigest()[:16]

        chunks: List[DocumentChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc_id}-{idx}"
            metadata = {
                "doc_id": doc_id,
                "chunk_index": idx,
                "chunk_count": len(raw_chunks),
                **(extra_metadata or {}),
            }
            if source:
                metadata["source"] = source

            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    metadata=metadata,
                )
            )

        logger.info(
            "text_chunked",
            source=source,
            doc_id=doc_id,
            n_chunks=len(chunks),
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return chunks

    def chunk_batch(
        self,
        documents: List[tuple[str, str]],
        *,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk many documents at once.

        Each document is a `(source, text)` tuple.
        """
        all_chunks: List[DocumentChunk] = []
        for source, text in documents:
            all_chunks.extend(
                self.chunk_text(text, source=source, extra_metadata=extra_metadata)
            )
        return all_chunks


def make_chunk_id() -> str:
    """Convenience helper for non-deterministic chunk ids (e.g. transient text)."""
    return uuid.uuid4().hex[:16]
