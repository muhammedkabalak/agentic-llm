"""
ChromaDB vector store wrapper.

Encapsulates all interactions with ChromaDB so the rest of the system
talks to a single, well-defined interface. Swapping ChromaDB out for
another vector DB later (Qdrant, Weaviate, …) becomes a matter of
implementing this same interface.

Key design choice: this wrapper expects pre-computed embeddings. The
embedding step is done by `BaseEmbedder` (separation of concerns) — we
do NOT delegate to Chroma's built-in embedding functions, because that
would couple the vector store to a specific embedding provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from app.config import Settings, get_settings
from app.models.domain import DocumentChunk
from app.services.logging_service import get_logger

logger = get_logger(__name__)


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> List[str]: ...

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> List[DocumentChunk]: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> int: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB-backed vector store."""

    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str,
    ) -> None:
        # Lazy import — heavy dependency
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        # `metadata={"hnsw:space": "cosine"}` ensures cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store_initialized",
            backend="chromadb",
            persist_dir=str(persist_dir),
            collection=collection_name,
            existing_items=self._collection.count(),
        )

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #
    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        if not chunks:
            return []
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must be the same length"
            )

        ids = [c.chunk_id or f"auto-{i}" for i, c in enumerate(chunks)]
        documents = [c.content for c in chunks]
        # Chroma metadata values must be primitive — sanitize.
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "vector_store_add",
            collection=self.collection_name,
            n_added=len(ids),
            total=self._collection.count(),
        )
        return ids

    # ------------------------------------------------------------------ #
    # Read operations
    # ------------------------------------------------------------------ #
    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )

        # Chroma returns lists-of-lists (one per query); we sent one query.
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: List[DocumentChunk] = []
        for cid, content, meta, dist in zip(ids, documents, metadatas, distances):
            # Chroma cosine "distance" is in [0, 2]; convert to similarity in [0, 1]
            similarity = max(0.0, 1.0 - float(dist) / 2.0)
            meta = meta or {}
            chunks.append(
                DocumentChunk(
                    content=content,
                    source=meta.get("source"),
                    chunk_id=cid,
                    score=similarity,
                    metadata=meta,
                )
            )
        return chunks

    def delete(self, ids: List[str]) -> int:
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        logger.info(
            "vector_store_delete",
            collection=self.collection_name,
            n_deleted=len(ids),
        )
        return len(ids)

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("vector_store_reset", collection=self.collection_name)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """Chroma only accepts str | int | float | bool metadata values."""
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif v is None:
                continue
            else:
                out[k] = str(v)
        return out


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #
def get_vector_store(settings: Optional[Settings] = None) -> BaseVectorStore:
    settings = settings or get_settings()
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
