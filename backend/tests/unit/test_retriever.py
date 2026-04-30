"""Unit tests for the Retriever (using HashEmbedder + InMemoryVectorStore)."""

from __future__ import annotations

from app.rag.retriever import Retriever
from tests.conftest import HashEmbedder, InMemoryVectorStore


def test_retriever_returns_empty_for_empty_query(
    hash_embedder: HashEmbedder, memory_vector_store: InMemoryVectorStore
) -> None:
    retriever = Retriever(hash_embedder, memory_vector_store)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retriever_returns_empty_when_store_is_empty(
    hash_embedder: HashEmbedder, memory_vector_store: InMemoryVectorStore
) -> None:
    retriever = Retriever(hash_embedder, memory_vector_store)
    assert retriever.retrieve("anything", top_k=3) == []


def test_retriever_returns_top_k_in_score_order(
    hash_embedder: HashEmbedder, memory_vector_store: InMemoryVectorStore
) -> None:
    from app.models.domain import DocumentChunk

    chunks = [
        DocumentChunk(content="alpha", chunk_id="a", source="s"),
        DocumentChunk(content="beta", chunk_id="b", source="s"),
        DocumentChunk(content="gamma", chunk_id="c", source="s"),
    ]
    embeddings = hash_embedder.embed_documents([c.content for c in chunks])
    memory_vector_store.add(chunks, embeddings)

    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=2)
    results = retriever.retrieve("beta")

    assert len(results) == 2
    # Exact match should be top result
    assert results[0].content == "beta"
    # Scores must be monotonically non-increasing
    assert results[0].score >= results[1].score


def test_retriever_min_score_filters_results(
    hash_embedder: HashEmbedder, memory_vector_store: InMemoryVectorStore
) -> None:
    from app.models.domain import DocumentChunk

    chunks = [DocumentChunk(content=f"doc-{i}", chunk_id=f"d{i}") for i in range(5)]
    embeddings = hash_embedder.embed_documents([c.content for c in chunks])
    memory_vector_store.add(chunks, embeddings)

    retriever = Retriever(hash_embedder, memory_vector_store)
    # Threshold of 1.0 should keep only the exact match (cosine == 1.0)
    results = retriever.retrieve("doc-2", top_k=5, min_score=0.9999)
    assert len(results) == 1
    assert results[0].content == "doc-2"
