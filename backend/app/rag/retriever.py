"""
Semantic retriever — composes an embedder and a vector store.

  retriever = Retriever(embedder, vector_store)
  chunks = retriever.retrieve("What is RAG?", top_k=5)

This is the single entry-point that agents call when they need
context. Agents should NOT touch the embedder or the vector store
directly — keep the seam here so we can later add hybrid search,
re-ranking, or query expansion behind the same interface.
"""

from __future__ import annotations

from typing import Any, List, Optional

from app.models.domain import DocumentChunk
from app.rag.embeddings import BaseEmbedder
from app.rag.vector_store import BaseVectorStore
from app.services.logging_service import get_logger

logger = get_logger(__name__)


class Retriever:
    """Embedder + vector store façade."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        *,
        default_top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        where: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[DocumentChunk]:
        """Return the most semantically relevant chunks for `query`."""
        if not query or not query.strip():
            return []

        k = top_k or self.default_top_k
        threshold = self.min_score if min_score is None else min_score

        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.query(
            embedding=query_embedding,
            top_k=k,
            where=where,
        )

        if threshold > 0.0:
            results = [c for c in results if c.score >= threshold]

        logger.info(
            "retrieval_done",
            query_preview=query[:80],
            requested_k=k,
            returned=len(results),
            min_score=threshold,
        )
        return results
