"""
Unit tests for SingleAgentPipeline.

Covers the full Step-3 flow without any network or model dependency:
    HashEmbedder + InMemoryVectorStore + ResearcherAgent + MockLLM

Verifies:
  * Retrieval happens before the agent runs and feeds context in.
  * The returned ChatResponse contains the LLM answer, traces, and
    sources, matching the public schema.
  * `top_k` overrides the pipeline default.
  * No retrieval results are gracefully handled (sources empty).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.agents.orchestrator import SingleAgentPipeline
from app.agents.researcher_agent import ResearcherAgent
from app.models.domain import DocumentChunk
from app.models.schemas import AgentRole, ChatMessage, MessageRole
from app.rag.retriever import Retriever

from tests.conftest import HashEmbedder, InMemoryVectorStore, MockLLM


def _seed(
    embedder: HashEmbedder,
    store: InMemoryVectorStore,
    docs: list[tuple[str, str]],
) -> None:
    """Helper: embed and store (content, source) pairs."""
    chunks = [
        DocumentChunk(content=content, source=source, chunk_id=f"c-{i}")
        for i, (content, source) in enumerate(docs)
    ]
    embeddings = embedder.embed_documents([c.content for c in chunks])
    store.add(chunks, embeddings)


@pytest.mark.asyncio
async def test_pipeline_retrieves_then_runs_agent(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(
        hash_embedder,
        memory_vector_store,
        [
            ("RAG combines retrieval with generation.", "doc-a.md"),
            ("Vector databases store embeddings.", "doc-b.md"),
            ("FastAPI is a Python web framework.", "doc-c.md"),
        ],
    )

    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    llm = MockLLM(canned_response="RAG combines retrieval and generation [1].")
    agent = ResearcherAgent(llm)

    pipeline = SingleAgentPipeline(retriever=retriever, agent=agent, default_top_k=3)

    response = await pipeline.run(query="What is RAG?")

    # Top-level answer comes from the LLM
    assert response.answer == "RAG combines retrieval and generation [1]."

    # Sources populated from the retriever
    assert len(response.sources) == 3
    sources_text = " ".join(s.content for s in response.sources)
    assert "RAG" in sources_text or "Vector" in sources_text

    # One agent trace, with the agent's role and output
    assert len(response.traces) == 1
    trace = response.traces[0]
    assert trace.agent_role == AgentRole.RESEARCHER
    assert trace.output == response.answer
    assert trace.input == "What is RAG?"
    assert len(trace.retrieved_chunks) == 3

    # Token + latency aggregation present
    assert response.total_tokens == 33  # MockLLM default
    assert response.total_latency_ms >= 0

    # The agent received the retrieved context in its user prompt
    assert len(llm.calls) == 1
    user_msg = llm.calls[0][-1].content
    assert "What is RAG?" in user_msg
    assert "[1]" in user_msg


@pytest.mark.asyncio
async def test_pipeline_respects_top_k_override(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(
        hash_embedder,
        memory_vector_store,
        [(f"chunk number {i}", f"src-{i}") for i in range(8)],
    )

    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=5)
    pipeline = SingleAgentPipeline(
        retriever=retriever,
        agent=ResearcherAgent(MockLLM()),
        default_top_k=5,
    )

    response = await pipeline.run(query="anything", top_k=2)

    assert len(response.sources) == 2
    assert len(response.traces[0].retrieved_chunks) == 2


@pytest.mark.asyncio
async def test_pipeline_with_empty_store_returns_no_sources(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    retriever = Retriever(hash_embedder, memory_vector_store)
    llm = MockLLM(canned_response="I cannot answer this from the provided sources.")
    pipeline = SingleAgentPipeline(
        retriever=retriever,
        agent=ResearcherAgent(llm),
    )

    response = await pipeline.run(query="What is RAG?")

    assert response.sources == []
    assert response.traces[0].retrieved_chunks == []
    assert "cannot answer" in response.answer.lower()
    # The agent saw the empty-context placeholder
    user_msg = llm.calls[0][-1].content
    assert "no retrieved context" in user_msg


@pytest.mark.asyncio
async def test_pipeline_passes_history_and_session_id(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store, [("ctx", "s")])
    retriever = Retriever(hash_embedder, memory_vector_store)
    llm = MockLLM(canned_response="ok")
    pipeline = SingleAgentPipeline(retriever=retriever, agent=ResearcherAgent(llm))

    sid = uuid4()
    history = [
        ChatMessage(role=MessageRole.USER, content="prior question"),
        ChatMessage(role=MessageRole.ASSISTANT, content="prior answer"),
    ]

    response = await pipeline.run(query="follow-up", history=history, session_id=sid)

    assert response.session_id == sid
    # Agent received history between system and current user message
    sent = llm.calls[0]
    roles = [m.role for m in sent]
    assert roles[0] == "system"
    assert "user" in roles and "assistant" in roles
    assert sent[-1].role == "user"
    assert "follow-up" in sent[-1].content
