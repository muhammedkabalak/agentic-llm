"""
Integration tests for the /chat route (single-agent mode).

Strategy: override BOTH `single_agent_pipeline_dep` and
`multi_agent_orchestrator_dep` with stubs so FastAPI can resolve the
route without instantiating the real (network-bound) ones. Tests
explicitly request mode="single" and assert on Researcher-only
behaviour. Multi-mode is covered in test_chat_multi_route.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.orchestrator import SingleAgentPipeline
from app.agents.researcher_agent import ResearcherAgent
from app.api.dependencies import (
    multi_agent_orchestrator_dep,
    single_agent_pipeline_dep,
)
from app.main import app
from app.models.domain import DocumentChunk
from app.rag.retriever import Retriever
from app.services.llm_provider import LLMProviderError

from tests.conftest import HashEmbedder, InMemoryVectorStore, MockLLM


def _seed(embedder: HashEmbedder, store: InMemoryVectorStore) -> None:
    docs = [
        ("RAG combines retrieval with generation.", "doc-a.md"),
        ("Vector databases store embeddings for similarity search.", "doc-b.md"),
        ("FastAPI is a modern Python web framework.", "doc-c.md"),
    ]
    chunks = [
        DocumentChunk(content=c, source=s, chunk_id=f"c-{i}")
        for i, (c, s) in enumerate(docs)
    ]
    embeddings = embedder.embed_documents([c.content for c in chunks])
    store.add(chunks, embeddings)


@pytest.fixture
def chat_client() -> TestClient:
    embedder = HashEmbedder(dim=32)
    store = InMemoryVectorStore()
    _seed(embedder, store)

    retriever = Retriever(embedder, store, default_top_k=3)
    llm = MockLLM(canned_response="RAG = retrieval-augmented generation [1].")
    agent = ResearcherAgent(llm)
    pipeline = SingleAgentPipeline(retriever=retriever, agent=agent, default_top_k=3)

    # Override BOTH deps so FastAPI's resolver does not try to build
    # the real multi-orchestrator (which needs a real LLM provider).
    app.dependency_overrides[single_agent_pipeline_dep] = lambda: pipeline
    app.dependency_overrides[multi_agent_orchestrator_dep] = lambda: None

    client = TestClient(app)
    client._llm = llm  # type: ignore[attr-defined]
    yield client

    app.dependency_overrides.clear()


def test_chat_returns_grounded_answer(chat_client: TestClient) -> None:
    response = chat_client.post(
        "/chat",
        json={"query": "What is RAG?", "top_k": 3, "mode": "single"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "RAG = retrieval-augmented generation [1]."
    assert len(body["sources"]) == 3
    assert any("RAG" in s["content"] for s in body["sources"])

    assert len(body["traces"]) == 1
    trace = body["traces"][0]
    assert trace["agent_role"] == "researcher"
    assert trace["output"] == body["answer"]
    assert trace["input"] == "What is RAG?"
    assert len(trace["retrieved_chunks"]) == 3

    assert body["total_tokens"] >= 0
    assert body["total_latency_ms"] >= 0
    assert "request_id" in body


def test_chat_validates_request_body(chat_client: TestClient) -> None:
    # Empty query rejected by Pydantic
    response = chat_client.post("/chat", json={"query": "", "mode": "single"})
    assert response.status_code == 422

    # top_k out of range rejected
    response = chat_client.post(
        "/chat", json={"query": "hi", "top_k": 0, "mode": "single"}
    )
    assert response.status_code == 422

    # invalid mode rejected
    response = chat_client.post(
        "/chat", json={"query": "hi", "mode": "nope"}
    )
    assert response.status_code == 422


def test_chat_top_k_override(chat_client: TestClient) -> None:
    response = chat_client.post(
        "/chat", json={"query": "RAG?", "top_k": 1, "mode": "single"}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 1
    assert len(body["traces"][0]["retrieved_chunks"]) == 1


def test_chat_with_history_round_trip(chat_client: TestClient) -> None:
    payload = {
        "query": "And what does it solve?",
        "history": [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "Retrieval-augmented generation."},
        ],
        "top_k": 2,
        "mode": "single",
    }
    response = chat_client.post("/chat", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]

    sent = chat_client._llm.calls[-1]  # type: ignore[attr-defined]
    assert len(sent) == 4
    assert sent[0].role == "system"
    assert sent[-1].role == "user"
    assert "And what does it solve?" in sent[-1].content


def test_chat_returns_502_on_llm_failure() -> None:
    embedder = HashEmbedder(dim=32)
    store = InMemoryVectorStore()
    _seed(embedder, store)
    retriever = Retriever(embedder, store)

    class BoomLLM(MockLLM):
        async def _generate(self, messages, *, temperature=None, max_tokens=None, **kwargs):
            raise LLMProviderError("upstream exploded")

    pipeline = SingleAgentPipeline(
        retriever=retriever,
        agent=ResearcherAgent(BoomLLM()),
    )
    app.dependency_overrides[single_agent_pipeline_dep] = lambda: pipeline
    app.dependency_overrides[multi_agent_orchestrator_dep] = lambda: None

    try:
        client = TestClient(app)
        response = client.post("/chat", json={"query": "anything", "mode": "single"})
        assert response.status_code == 502
        assert "Upstream LLM" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
