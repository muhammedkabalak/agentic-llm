"""
Integration tests for the /eval route.

Overrides both the single and multi pipeline deps with in-memory
test doubles (HashEmbedder + InMemoryVectorStore + MockLLM) so the
full HTTP path runs offline.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.analyst_agent import AnalystAgent
from app.agents.critic_agent import CriticAgent
from app.agents.orchestrator import MultiAgentOrchestrator, SingleAgentPipeline
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
def eval_client() -> TestClient:
    embedder = HashEmbedder(dim=32)
    store = InMemoryVectorStore()
    _seed(embedder, store)

    retriever = Retriever(embedder, store, default_top_k=3)
    single_llm = MockLLM(
        canned_response="RAG combines retrieval with generation [1]."
    )
    single_pipeline = SingleAgentPipeline(
        retriever=retriever,
        agent=ResearcherAgent(single_llm),
        default_top_k=3,
    )

    multi_llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft [1].",
            "## Summary\nRAG combines retrieval with generation [1].",
            "VERDICT: pass\nFLAGS: none\nNOTES: ok\nREVISED_ANSWER:\nUNCHANGED\n",
        )
    )
    multi_orch = MultiAgentOrchestrator(
        retriever=retriever,
        researcher=ResearcherAgent(multi_llm),
        analyst=AnalystAgent(multi_llm),
        critic=CriticAgent(multi_llm),
        default_top_k=3,
    )

    app.dependency_overrides[single_agent_pipeline_dep] = lambda: single_pipeline
    app.dependency_overrides[multi_agent_orchestrator_dep] = lambda: multi_orch

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


def test_eval_run_single_mode_returns_report(eval_client: TestClient) -> None:
    payload = {
        "mode": "single",
        "dataset_name": "demo",
        "cases": [
            {
                "case_id": "rag-1",
                "query": "What is RAG?",
                "expected_answer": "RAG combines retrieval with generation.",
                "expected_keywords": ["retrieval", "generation"],
                "expected_sources": ["doc-a.md"],
            }
        ],
    }
    response = eval_client.post("/eval/run", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert body["dataset_name"] == "demo"
    assert body["n_cases"] == 1
    assert body["mode"] == "single"
    assert len(body["cases"]) == 1

    case = body["cases"][0]
    assert case["case_id"] == "rag-1"
    assert "bleu_like" in case["metrics"]
    assert case["metrics"]["task_completion"] == 1.0
    assert case["metrics"]["retrieval_at_k"] == 1.0
    assert case["guardrail_verdict"] == "pass"

    agg = body["aggregate"]
    assert agg["n_cases"] == 1
    assert agg["n_errors"] == 0
    assert agg["guardrail_pass_rate"] == 1.0


def test_eval_run_multi_mode(eval_client: TestClient) -> None:
    payload = {
        "mode": "multi",
        "cases": [
            {
                "query": "What is RAG?",
                "expected_keywords": ["retrieval", "generation"],
                "expected_sources": ["doc-a.md"],
            }
        ],
    }
    response = eval_client.post("/eval/run", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "multi"
    case = body["cases"][0]
    assert case["metrics"]["task_completion"] == 1.0
    # Multi-agent uses three LLM calls -> tokens_used > single-mode default.
    assert case["tokens_used"] > 33


def test_eval_run_validates_request(eval_client: TestClient) -> None:
    # Empty cases list rejected
    response = eval_client.post(
        "/eval/run", json={"mode": "single", "cases": []}
    )
    assert response.status_code == 422

    # Empty query rejected
    response = eval_client.post(
        "/eval/run",
        json={"mode": "single", "cases": [{"query": ""}]},
    )
    assert response.status_code == 422

    # Bogus mode rejected
    response = eval_client.post(
        "/eval/run",
        json={"mode": "weird", "cases": [{"query": "ok"}]},
    )
    assert response.status_code == 422


def test_eval_sample_dataset_endpoint(eval_client: TestClient) -> None:
    response = eval_client.get("/eval/sample-dataset")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "sample"
    assert len(body["cases"]) >= 2
    assert all(c["query"] for c in body["cases"])
