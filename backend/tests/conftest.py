"""
Shared test fixtures.

Provides a deterministic, dependency-free embedder, an in-memory
vector store, and a mock LLM provider so RAG/agent tests run fast
and don't require any model downloads, API keys, or network access.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, List, Optional

import pytest

from app.config import Settings
from app.models.domain import DocumentChunk
from app.rag.embeddings import BaseEmbedder
from app.rag.vector_store import BaseVectorStore
from app.services.llm_provider import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
)


# --------------------------------------------------------------------- #
# Hash-based deterministic embedder (no model download required)
# --------------------------------------------------------------------- #
class HashEmbedder(BaseEmbedder):
    """
    Maps text -> fixed-size unit vector deterministically via SHA-256.
    Not semantically meaningful but stable and dependency-free.
    """

    def __init__(self, dim: int = 32) -> None:
        self.settings = Settings.model_construct(embedding_model="hash-test")
        self.model_name = "hash-test"
        self._dim = dim

    @property
    def provider_name(self) -> str:
        return "hash"

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = []
        i = 0
        while len(raw) < self._dim:
            raw.append((digest[i % len(digest)] / 255.0) * 2.0 - 1.0)
            i += 1
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# --------------------------------------------------------------------- #
# In-memory vector store (cosine similarity)
# --------------------------------------------------------------------- #
class InMemoryVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def add(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> List[str]:
        ids: List[str] = []
        for chunk, emb in zip(chunks, embeddings):
            cid = chunk.chunk_id or f"auto-{len(self._items)}"
            self._items[cid] = {
                "content": chunk.content,
                "embedding": emb,
                "metadata": dict(chunk.metadata or {}),
                "source": chunk.source,
            }
            ids.append(cid)
        return ids

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        def cosine(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (na * nb)

        candidates = self._items.items()
        if where:
            candidates = [
                (cid, item)
                for cid, item in candidates
                if all(item["metadata"].get(k) == v for k, v in where.items())
            ]

        scored = [
            (cid, item, cosine(embedding, item["embedding"])) for cid, item in candidates
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        results = scored[:top_k]

        return [
            DocumentChunk(
                content=item["content"],
                source=item["source"],
                chunk_id=cid,
                score=score,
                metadata=dict(item["metadata"]),
            )
            for cid, item, score in results
        ]

    def delete(self, ids: List[str]) -> int:
        n = 0
        for i in ids:
            if self._items.pop(i, None) is not None:
                n += 1
        return n

    def count(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items.clear()


# --------------------------------------------------------------------- #
# Mock LLM provider (no network, no API key required)
# --------------------------------------------------------------------- #
class MockLLM(BaseLLMProvider):
    """
    Deterministic LLM for tests.

    Records every call so tests can assert on the messages the agent
    actually sent. By default echoes a canned answer; pass
    ``response_fn`` for dynamic behaviour.
    """

    def __init__(
        self,
        *,
        canned_response: str = "MOCK_ANSWER",
        response_fn: Optional[Any] = None,
        prompt_tokens: int = 11,
        completion_tokens: int = 22,
    ) -> None:
        # Bypass the parent's settings-driven __init__.
        self.settings = Settings.model_construct(
            llm_model="mock-llm",
            llm_temperature=0.0,
            llm_max_tokens=512,
            llm_timeout_seconds=5,
        )
        self.model = "mock-llm"
        self.temperature = 0.0
        self.max_tokens = 512
        self.timeout = 5
        self._canned = canned_response
        self._response_fn = response_fn
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: List[List[LLMMessage]] = []
        self.last_kwargs: dict[str, Any] = {}

    @property
    def provider_name(self) -> str:
        return "mock"

    async def _generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.last_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if self._response_fn is not None:
            content = self._response_fn(messages)
        else:
            content = self._canned
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._prompt_tokens + self._completion_tokens,
            raw=None,
        )


# --------------------------------------------------------------------- #
# Pytest fixtures
# --------------------------------------------------------------------- #
@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder(dim=32)


@pytest.fixture
def memory_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM(canned_response="Mock grounded answer with citation [1].")
