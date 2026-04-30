"""Unit tests for MultiAgentOrchestrator (Researcher -> Analyst -> Critic)."""

from __future__ import annotations

import pytest

from app.agents.analyst_agent import AnalystAgent
from app.agents.critic_agent import CriticAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.researcher_agent import ResearcherAgent
from app.models.domain import DocumentChunk
from app.models.schemas import AgentRole, GuardrailVerdict
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


def _role_router(researcher_text: str, analyst_text: str, critic_text: str):
    """Return a response_fn that picks output based on the system prompt."""
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


@pytest.mark.asyncio
async def test_orchestrator_runs_three_agents_in_order(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)

    critic_text = (
        "VERDICT: pass\n"
        "FLAGS: none\n"
        "NOTES: Solid.\n"
        "REVISED_ANSWER:\nUNCHANGED\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft [1].",
            "## Summary\nAnalyst answer [1].",
            critic_text,
        )
    )

    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    orch = MultiAgentOrchestrator(
        retriever=retriever,
        researcher=ResearcherAgent(llm),
        analyst=AnalystAgent(llm),
        critic=CriticAgent(llm),
        default_top_k=3,
    )

    response = await orch.run(query="What is RAG?")

    # Three agent traces, in order
    assert [t.agent_role for t in response.traces] == [
        AgentRole.RESEARCHER,
        AgentRole.ANALYST,
        AgentRole.CRITIC,
    ]

    # On a "pass" verdict the Analyst's output is what the user sees
    assert response.answer.startswith("## Summary")
    assert "Analyst answer" in response.answer

    # Token aggregation across all three agents
    assert response.total_tokens == 33 * 3

    # Sources populated and identical across traces
    assert len(response.sources) == 3
    for trace in response.traces:
        assert len(trace.retrieved_chunks) == 3

    # Guardrail report attached to every trace now (Step 5: monitor
    # inspects every agent output; researcher is monitor-only).
    researcher_trace = response.traces[0]
    analyst_trace = response.traces[1]
    critic_trace = response.traces[2]
    assert researcher_trace.guardrail_report is not None
    assert researcher_trace.guardrail_report.verdict == GuardrailVerdict.PASS
    assert analyst_trace.guardrail_report is not None
    assert analyst_trace.guardrail_report.verdict == GuardrailVerdict.PASS
    assert critic_trace.guardrail_report is not None
    assert critic_trace.guardrail_report.verdict == GuardrailVerdict.PASS

    # Verify the Analyst actually saw the Researcher's output upstream.
    # Look at the second LLM call (the Analyst's): its user message must
    # include the researcher draft.
    analyst_messages = llm.calls[1]
    assert "Analyst Agent" in analyst_messages[0].content
    assert "Researcher draft [1]." in analyst_messages[-1].content

    # Verify the Critic saw the Analyst's output.
    critic_messages = llm.calls[2]
    assert "Critic Agent" in critic_messages[0].content
    assert "Analyst answer" in critic_messages[-1].content


@pytest.mark.asyncio
async def test_orchestrator_uses_revised_answer_on_warn(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)

    critic_text = (
        "VERDICT: warn\n"
        "FLAGS: missing_citation\n"
        "NOTES: One claim lacked a citation.\n"
        "REVISED_ANSWER:\n"
        "## Summary\nRevised analyst answer with citation [1].\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft [1].",
            "## Summary\nOriginal analyst answer.",
            critic_text,
        )
    )

    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    orch = MultiAgentOrchestrator(
        retriever=retriever,
        researcher=ResearcherAgent(llm),
        analyst=AnalystAgent(llm),
        critic=CriticAgent(llm),
        default_top_k=3,
    )

    response = await orch.run(query="What is RAG?")

    # On warn, the Critic's revised answer is surfaced
    assert "Revised analyst answer" in response.answer
    assert "Original analyst answer" not in response.answer

    report = response.traces[1].guardrail_report
    assert report is not None
    assert report.verdict == GuardrailVerdict.WARN
    assert "missing_citation" in report.flags


@pytest.mark.asyncio
async def test_orchestrator_skips_critic_when_disabled(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)

    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft.",
            "Analyst final.",
            "SHOULD NOT BE CALLED",
        )
    )

    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    orch = MultiAgentOrchestrator(
        retriever=retriever,
        researcher=ResearcherAgent(llm),
        analyst=AnalystAgent(llm),
        critic=CriticAgent(llm),
        default_top_k=3,
        enable_critic=False,
    )

    response = await orch.run(query="What is RAG?")

    # Only two traces, no critic
    assert [t.agent_role for t in response.traces] == [
        AgentRole.RESEARCHER,
        AgentRole.ANALYST,
    ]
    assert response.answer == "Analyst final."

    # Exactly two LLM calls (no critic invocation)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_orchestrator_handles_critic_none(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)

    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft.",
            "Analyst final answer.",
            "",
        )
    )
    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=2)
    orch = MultiAgentOrchestrator(
        retriever=retriever,
        researcher=ResearcherAgent(llm),
        analyst=AnalystAgent(llm),
        critic=None,
        default_top_k=2,
    )

    response = await orch.run(query="hello")

    assert response.answer == "Analyst final answer."
    assert len(response.traces) == 2
    # Critic disabled, but the GuardrailMonitor still attaches a clean
    # report to the analyst trace (Step 5).
    analyst_report = response.traces[1].guardrail_report
    assert analyst_report is not None
    assert analyst_report.verdict == GuardrailVerdict.PASS
