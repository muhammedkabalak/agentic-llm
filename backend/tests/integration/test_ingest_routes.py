"""
Integration tests for /ingest routes.

We override FastAPI's vector_store / embedder / pipeline dependencies
with the in-memory test fixtures so we never touch ChromaDB or download
any embedding model during tests.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    embedder_dep,
    ingestion_pipeline_dep,
    vector_store_dep,
)
from app.main import app
from app.rag.ingestion_pipeline import IngestionPipeline
from tests.conftest import HashEmbedder, InMemoryVectorStore


@pytest.fixture
def client() -> TestClient:
    embedder = HashEmbedder(dim=32)
    store = InMemoryVectorStore()
    pipeline = IngestionPipeline.from_components(
        embedder=embedder, vector_store=store
    )

    app.dependency_overrides[embedder_dep] = lambda: embedder
    app.dependency_overrides[vector_store_dep] = lambda: store
    app.dependency_overrides[ingestion_pipeline_dep] = lambda: pipeline

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_ingest_text_returns_chunk_count(client: TestClient) -> None:
    payload = {
        "text": "The quick brown fox jumps over the lazy dog. " * 30,
        "source": "fox.txt",
        "metadata": {"author": "test"},
    }
    response = client.post("/ingest/text", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["source"] == "fox.txt"
    assert body["n_chunks"] >= 1
    assert body["embedding_model"] == "hash-test"
    assert body["embedding_dim"] == 32
    assert body["collection_total"] == body["n_chunks"]


def test_ingest_text_rejects_empty(client: TestClient) -> None:
    response = client.post("/ingest/text", json={"text": "", "source": "x"})
    assert response.status_code == 422


def test_ingest_file_txt(client: TestClient) -> None:
    file_content = b"Some short text content for ingestion.\n" * 5
    files = {"file": ("hello.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/ingest/file", files=files)
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["source"] == "hello.txt"
    assert body["n_chunks"] >= 1


def test_ingest_file_rejects_unsupported_extension(client: TestClient) -> None:
    files = {"file": ("evil.exe", io.BytesIO(b"binary"), "application/octet-stream")}
    response = client.post("/ingest/file", files=files)
    assert response.status_code == 415


def test_collection_stats_endpoint(client: TestClient) -> None:
    client.post(
        "/ingest/text",
        json={"text": "stats test " * 30, "source": "stats.txt"},
    )
    response = client.get("/ingest/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["embedding_model"] == "hash-test"
    assert body["embedding_dim"] == 32
    assert body["total_chunks"] >= 1


def test_delete_collection_resets(client: TestClient) -> None:
    client.post(
        "/ingest/text",
        json={"text": "to be deleted " * 30, "source": "del.txt"},
    )
    pre = client.get("/ingest/stats").json()["total_chunks"]
    assert pre >= 1

    response = client.delete("/ingest")
    assert response.status_code == 200
    assert response.json().get("reset") is True

    post = client.get("/ingest/stats").json()["total_chunks"]
    assert post == 0


# --------------------------------------------------------------------- #
# /ingest/pdf
# --------------------------------------------------------------------- #
def _make_pdf_bytes(text: str) -> bytes:
    """Build a minimal one-page PDF with `text` rendered on it."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    # Wrap long text across multiple lines so the PDF actually has
    # text content even for paragraphs.
    y = 720
    for line in text.splitlines() or [text]:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()
    return buf.getvalue()


def test_ingest_pdf_extracts_and_indexes(client: TestClient) -> None:
    pdf_bytes = _make_pdf_bytes(
        "RAG combines retrieval with generation.\n"
        "Vector databases store dense embeddings."
    )
    response = client.post(
        "/ingest/pdf",
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
        data={"metadata_str": '{"author": "test"}'},
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["success"] is True
    assert body["source"] == "paper.pdf"
    assert body["n_chunks"] >= 1
    assert body["n_pages"] == 1
    assert body["n_chars_extracted"] is not None
    assert body["n_chars_extracted"] > 0
    assert body["embedding_model"] == "hash-test"
    assert body["collection_total"] == body["n_chunks"]


def test_ingest_pdf_respects_source_override(client: TestClient) -> None:
    pdf_bytes = _make_pdf_bytes("hello world content goes here")
    response = client.post(
        "/ingest/pdf",
        files={"file": ("ignored.pdf", pdf_bytes, "application/pdf")},
        data={"source": "handbook/section-2.pdf"},
    )
    assert response.status_code == 201
    assert response.json()["source"] == "handbook/section-2.pdf"


def test_ingest_pdf_rejects_non_pdf_extension(client: TestClient) -> None:
    response = client.post(
        "/ingest/pdf",
        files={"file": ("notes.txt", b"plain text not pdf", "text/plain")},
    )
    assert response.status_code == 415


def test_ingest_pdf_rejects_empty_upload(client: TestClient) -> None:
    response = client.post(
        "/ingest/pdf",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_ingest_pdf_rejects_invalid_metadata_json(client: TestClient) -> None:
    pdf_bytes = _make_pdf_bytes("hello")
    response = client.post(
        "/ingest/pdf",
        files={"file": ("x.pdf", pdf_bytes, "application/pdf")},
        data={"metadata_str": "not-json"},
    )
    assert response.status_code == 400


def test_ingest_pdf_skips_when_no_text_extracted(client: TestClient) -> None:
    """A blank PDF (one empty page) yields zero chunks but still 201s
    with a `skipped_reason` so the UI can show why nothing was added."""
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    pdf_bytes = buf.getvalue()

    response = client.post(
        "/ingest/pdf",
        files={"file": ("blank.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is False
    assert body["n_chunks"] == 0
    assert body["n_pages"] == 1
    assert body["n_chars_extracted"] == 0
    assert body["skipped_reason"] is not None
    assert "no_text_extracted" in body["skipped_reason"]


def test_ingest_pdf_rejects_corrupt_file(client: TestClient) -> None:
    response = client.post(
        "/ingest/pdf",
        files={"file": ("bad.pdf", b"not actually a pdf", "application/pdf")},
    )
    assert response.status_code == 400
