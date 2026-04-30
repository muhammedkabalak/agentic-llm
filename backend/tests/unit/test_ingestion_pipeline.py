"""Unit tests for the IngestionPipeline (text + file)."""

from __future__ import annotations

from pathlib import Path

from app.rag.chunking import ChunkingConfig
from app.rag.ingestion_pipeline import IngestionPipeline
from app.rag.retriever import Retriever
from tests.conftest import HashEmbedder, InMemoryVectorStore


def test_ingest_text_chunks_embeds_and_stores(
    hash_embedder: HashEmbedder, memory_vector_store: InMemoryVectorStore
) -> None:
    pipeline = IngestionPipeline.from_components(
        embedder=hash_embedder,
        vector_store=memory_vector_store,
        chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=10),
    )

    text = "RAG stands for Retrieval Augmented Generation. " * 20
    report = pipeline.ingest_text(text, source="rag_intro.txt")

    assert report.success
    assert report.n_chunks > 0
    assert len(report.stored_ids) == report.n_chunks
    assert report.embedding_model == "hash-test"
    assert report.embedding_dim == 32
    assert memory_vector_store.count() == report.n_chunks


def test_ingest_empty_text_is_skipped(
    hash_embedder: HashEmbedder, memory_vector_store: InMemoryVectorStore
) -> None:
    pipeline = IngestionPipeline.from_components(
        embedder=hash_embedder, vector_store=memory_vector_store
    )
    report = pipeline.ingest_text("   \n  ", source="blank.txt")
    assert not report.success
    assert report.n_chunks == 0
    assert report.skipped_reason == "empty_text"
    assert memory_vector_store.count() == 0


def test_ingest_file_txt(
    tmp_path: Path,
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Line one.\n" * 50, encoding="utf-8")

    pipeline = IngestionPipeline.from_components(
        embedder=hash_embedder,
        vector_store=memory_vector_store,
        chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=10),
    )
    report = pipeline.ingest_file(file_path)

    assert report.success
    assert report.source == "doc.txt"
    assert memory_vector_store.count() == report.n_chunks


def test_full_loop_ingest_then_retrieve(
    hash_embedder: HashEmbedder, memory_vector_store: InMemoryVectorStore
) -> None:
    pipeline = IngestionPipeline.from_components(
        embedder=hash_embedder,
        vector_store=memory_vector_store,
        chunking_config=ChunkingConfig(chunk_size=80, chunk_overlap=10),
    )
    pipeline.ingest_text(
        "Quantum computing uses qubits instead of classical bits.",
        source="qc.txt",
    )
    pipeline.ingest_text(
        "Bananas are a popular yellow fruit grown in tropical regions.",
        source="fruit.txt",
    )

    retriever = Retriever(hash_embedder, memory_vector_store)
    # Retrieve a chunk that exactly matches one we stored
    results = retriever.retrieve(
        "Quantum computing uses qubits instead of classical bits.", top_k=1
    )
    assert len(results) == 1
    assert "qubits" in results[0].content
    assert results[0].source == "qc.txt"
