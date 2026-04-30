"""Unit tests for the TextChunker."""

from __future__ import annotations

import pytest

from app.rag.chunking import ChunkingConfig, TextChunker


def test_short_text_produces_single_chunk() -> None:
    chunker = TextChunker(ChunkingConfig(chunk_size=200, chunk_overlap=20))
    chunks = chunker.chunk_text("Short document.", source="t.txt")
    assert len(chunks) == 1
    assert chunks[0].source == "t.txt"
    assert chunks[0].chunk_id and chunks[0].chunk_id.endswith("-0")
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_count"] == 1


def test_long_text_is_split_with_overlap() -> None:
    text = ("Paragraph one. " * 60) + "\n\n" + ("Paragraph two. " * 60)
    chunker = TextChunker(ChunkingConfig(chunk_size=200, chunk_overlap=40))
    chunks = chunker.chunk_text(text, source="long.txt")
    assert len(chunks) >= 3
    # Every chunk respects max size (allow small slack from splitter heuristics)
    for c in chunks:
        assert len(c.content) <= 220
    # Sequential chunk indices
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # Same doc_id for all chunks of the same source
    doc_ids = {c.metadata["doc_id"] for c in chunks}
    assert len(doc_ids) == 1


def test_empty_text_returns_no_chunks() -> None:
    chunker = TextChunker()
    assert chunker.chunk_text("", source="x") == []
    assert chunker.chunk_text("   \n\n  ", source="x") == []


def test_extra_metadata_propagates() -> None:
    chunker = TextChunker(ChunkingConfig(chunk_size=200, chunk_overlap=20))
    chunks = chunker.chunk_text(
        "Some content here.", source="m.txt", extra_metadata={"author": "mo", "year": 2026}
    )
    assert chunks[0].metadata["author"] == "mo"
    assert chunks[0].metadata["year"] == 2026


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)
