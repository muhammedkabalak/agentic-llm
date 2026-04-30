"""
Integration tests for the /chat route in multi-agent mode.

Overrides `multi_agent_orchestrator_dep` (and stubs the single one) so
the full HTTP path runs end-to-end with HashEmbedder + InMemoryVectorStore
+ MockLLM. The single-agent route is covered in test_chat_route.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.analyst_agent import AnalystAgent
from app.agents.critic_agent import CriticAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.researcher_agent import ResearcherAgent
from app.api.dependencies import (
    multi_agent_orchestrator_dep,
    single_agent_pipeline_dep,
)
from app.main import app
from app.models.domain import DocumentChunk
from app.rag.retriever import Retriever

from tests.conftest import HashEmbedder, InMemoryVectorStore, MockLLM


def _seed(embedder: HashEmbedder, store: InMemoryVectorStore) -> None:
    docs = [
        ("RAG combines retrieval with generation.", "doc-a.md"),
        ("Vector databases store embeddings.", "doc-b.md"),
        ("FastAPI is a Python web framework.", "doc-c.md"),
    ]
    chunks = [
        DocumentChunk(content=c, source=s, chunk_id=f"c-{i}")
        for i, (c, s) in enumerate(docs)
    ]
    embeddings = embedder.embed_documents([c.content for c in chunks])
    store.add(chunks, embeddings)


def _role_router(researcher_text, analyst_text, critic_text):
    def fn(messages):
        sys = messages[0].content if messages and messages[0].role == "system" else ""
        if "You are the Researcher Agent" in sys:
            return researcher_text
        if "You are the Analyst Agent" in sys:
            return analyst_text
        if "You are the Critic Agent" in sys:
            return critic_text
        return ""
    return fn


@pytest.fixture
def multi_chat_client() -> TestClient:
    embedder = HashEmbedder(dim=32)
    store = InMemoryVectorStore()
    _seed(embedder, store)

    critic_text = (
        "VERDICT: pass\n"
        "FLAGS: none\n"
        "NOTES: All grounded.\n"
        "REVISED_ANSWER:\nUNCHANGED\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft answer [1].",
            "## Summary\nMulti-agent answer [1].",
            critic_text,
        )
    )

    retriever = Retriever(embedder, store, default_top_k=3)
    orch = MultiAgentOrchestrator(
        retriever=retriever,
        researcher=ResearcherAgent(llm),
        analyst=AnalystAgent(llm),
        critic=CriticAgent(llm),
        default_top_k=3,
    )

    app.dependency_overrides[multi_agent_orchestrator_dep] = lambda: orch
    app.dependency_overrides[single_agent_pipeline_dep] = lambda: None

    client = TestClient(app)
    client._llm = llm  # type: ignore[attr-defined]
    yield client

    app.dependency_overrides.clear()


def test_chat_multi_mode_runs_three_agents(multi_chat_client: TestClient) -> None:
    response = multi_chat_client.post(
        "/chat", json={"query": "What is RAG?", "mode": "multi", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()

    # Final answer is the analyst's (since critic verdict was "pass")
    assert "Multi-agent answer" in body["answer"]

    # Three traces, ordered Researcher -> Analyst -> Critic
    roles = [t["agent_role"] for t in body["traces"]]
    assert roles == ["researcher", "analyst", "critic"]

    # Guardrail report attached to every trace (researcher carries the
    # monitor-only report; analyst + critic carry the merged report).
    assert body["traces"][0]["guardrail_report"]["verdict"] == "pass"
    assert body["traces"][1]["guardrail_report"]["verdict"] == "pass"
    assert body["traces"][2]["guardrail_report"]["verdict"] == "pass"

    # Sources populated
    assert len(body["sources"]) == 3


def test_chat_multi_mode_uses_revised_answer_on_warn(
    multi_chat_client: TestClient,
) -> None:
    # Re-route the LLM to issue a warn verdict
    multi_chat_client._llm._response_fn = _role_router(  # type: ignore[attr-defined]
        "Researcher draft.",
        "## Summary\nOriginal analyst.",
        (
            "VERDICT: warn\n"
            "FLAGS: missing_citation\n"
            "NOTES: Add citation.\n"
            "REVISED_ANSWER:\n"
            "## Summary\nCorrected analyst answer [1].\n"
        ),
    )

    response = multi_chat_client.post(
        "/chat", json={"query": "What is RAG?", "mode": "multi"}
    )
    assert response.status_code == 200
    body = response.json()

    assert "Corrected analyst answer" in body["answer"]
    assert "Original analyst" not in body["answer"]

    report = body["traces"][1]["guardrail_report"]
    assert report["verdict"] == "warn"
    assert "missing_citation" in report["flags"]


def test_chat_multi_default_mode_is_multi(multi_chat_client: TestClient) -> None:
    # Omit `mode` entirely - default is "multi"
    response = multi_chat_client.post("/chat", json={"query": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()

    # Three traces means the multi pipeline ran (single would yield one)
    assert len(body["traces"]) == 3
    roles = [t["agent_role"] for t in body["traces"]]
    assert roles == ["researcher", "analyst", "critic"]
