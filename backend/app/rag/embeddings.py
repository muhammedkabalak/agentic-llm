"""
Embedding model abstraction.

A `BaseEmbedder` exposes a uniform interface that the rest of the RAG
stack (vector store, retriever, ingestion pipeline) calls. Concrete
implementations:

  * SentenceTransformerEmbedder — local model, no API key required (default).
  * OpenAIEmbedder — uses OpenAI's embedding API.

Pattern: Strategy + Factory (mirrors `llm_provider.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.config import EmbeddingProvider, Settings, get_settings
from app.services.logging_service import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedding model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.embedding_model

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents (used during ingestion)."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (used at retrieval time)."""


# --------------------------------------------------------------------- #
# Sentence-Transformers (local)
# --------------------------------------------------------------------- #
class SentenceTransformerEmbedder(BaseEmbedder):
    @property
    def provider_name(self) -> str:
        return "sentence-transformers"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        # Lazy import — heavy dependency
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            "embedder_initialized",
            provider=self.provider_name,
            model=self.model_name,
            dimension=self._dim,
        )

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()


# --------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------- #
class OpenAIEmbedder(BaseEmbedder):
    # Common OpenAI embedding model dimensions
    _OPENAI_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    @property
    def provider_name(self) -> str:
        return "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder")

        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(
            model=self.model_name,
            api_key=settings.openai_api_key,
        )
        self._dim = self._OPENAI_DIMS.get(self.model_name, 1536)
        logger.info(
            "embedder_initialized",
            provider=self.provider_name,
            model=self.model_name,
            dimension=self._dim,
        )

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #
_EMBEDDER_REGISTRY: dict[EmbeddingProvider, type[BaseEmbedder]] = {
    EmbeddingProvider.SENTENCE_TRANSFORMERS: SentenceTransformerEmbedder,
    EmbeddingProvider.OPENAI: OpenAIEmbedder,
}


def get_embedder(settings: Optional[Settings] = None) -> BaseEmbedder:
    """Factory that returns the configured embedder."""
    settings = settings or get_settings()
    embedder_cls = _EMBEDDER_REGISTRY.get(settings.embedding_provider)
    if embedder_cls is None:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
    return embedder_cls(settings)
