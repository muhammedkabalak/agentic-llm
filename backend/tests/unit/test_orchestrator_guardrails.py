"""
Integration tests: GuardrailMonitor wired into the orchestrators.

These verify the *behavioural* contract the project document asks for:
the guardrail layer must intercept every agent output and prevent
unsafe content from reaching the user.
"""

from __future__ import annotations

import pytest

from app.agents.analyst_agent import AnalystAgent
from app.agents.critic_agent import CriticAgent
from app.agents.orchestrator import (
    BLOCKED_ANSWER_FALLBACK,
    MultiAgentOrchestrator,
    SingleAgentPipeline,
)
from app.agents.researcher_agent import ResearcherAgent
from app.guardrails.monitor import GuardrailMonitor
from app.models.domain import DocumentChunk
from app.models.schemas import GuardrailCategory, GuardrailVerdict
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


# --------------------------------------------------------------------- #
# Single-agent pipeline
# --------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_single_pipeline_redacts_pii_in_answer(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    leaky = "RAG combines retrieval with generation. Contact me at leak@example.com."
    llm = MockLLM(canned_response=leaky)
    pipeline = SingleAgentPipeline(
        retriever=retriever, agent=ResearcherAgent(llm), default_top_k=3
    )

    response = await pipeline.run(query="What is RAG?")

    assert "[REDACTED:EMAIL]" in response.answer
    assert "leak@example.com" not in response.answer
    assert response.traces[0].output == leaky

    report = response.traces[0].guardrail_report
    assert report is not None
    assert report.verdict == GuardrailVerdict.BLOCK
    assert "pii" in report.flags
    assert any(f.category == GuardrailCategory.PII for f in report.findings)


@pytest.mark.asyncio
async def test_single_pipeline_blocks_radicalization(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    bad = "RAG combines retrieval. You should kill them all tomorrow."
    llm = MockLLM(canned_response=bad)
    pipeline = SingleAgentPipeline(
        retriever=retriever, agent=ResearcherAgent(llm), default_top_k=3
    )

    response = await pipeline.run(query="What is RAG?")

    assert response.answer == BLOCKED_ANSWER_FALLBACK
    report = response.traces[0].guardrail_report
    assert report is not None
    assert report.verdict == GuardrailVerdict.BLOCK
    assert "radicalization" in report.flags


@pytest.mark.asyncio
async def test_single_pipeline_passes_clean_answer(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    clean = "RAG combines retrieval with generation [1]."
    llm = MockLLM(canned_response=clean)
    pipeline = SingleAgentPipeline(
        retriever=retriever, agent=ResearcherAgent(llm), default_top_k=3
    )

    response = await pipeline.run(query="What is RAG?")

    assert response.answer == clean
    report = response.traces[0].guardrail_report
    assert report is not None
    assert report.verdict == GuardrailVerdict.PASS
    assert report.findings == []


# --------------------------------------------------------------------- #
# Multi-agent orchestrator
# --------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_multi_orchestrator_redacts_pii_when_analyst_leaks(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    leaky_analyst = (
        "## Summary\nReach me at leak@example.com.\n"
        "## Key Points\n- nothing\n"
        "## Gaps\n- none\n"
        "## Recommendation\nfollow up.\n"
    )
    critic_pass = (
        "VERDICT: pass\n"
        "FLAGS: none\n"
        "NOTES: looks fine.\n"
        "REVISED_ANSWER:\nUNCHANGED\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft [1].",
            leaky_analyst,
            critic_pass,
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

    # Critic said pass, but the monitor saw PII in the analyst output.
    # Policy: sanitise PII rather than blank out the answer.
    assert "[REDACTED:EMAIL]" in response.answer
    assert "leak@example.com" not in response.answer
    assert response.answer != BLOCKED_ANSWER_FALLBACK

    analyst_report = response.traces[1].guardrail_report
    assert analyst_report is not None
    assert analyst_report.verdict == GuardrailVerdict.BLOCK
    assert "pii" in analyst_report.flags
    # Critic verdict was pass; the monitor verdict promoted it.
    assert analyst_report.monitor_verdict == GuardrailVerdict.BLOCK


@pytest.mark.asyncio
async def test_multi_orchestrator_blocks_when_analyst_incites_violence(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    """A non-PII BLOCK upstream must short-circuit to the safe fallback."""
    _seed(hash_embedder, memory_vector_store)
    bad_analyst = (
        "## Summary\nYou should kill them all tomorrow.\n"
        "## Key Points\n- one\n"
        "## Gaps\n- none\n"
        "## Recommendation\nact now.\n"
    )
    critic_pass = (
        "VERDICT: pass\nFLAGS: none\nNOTES: ok\nREVISED_ANSWER:\nUNCHANGED\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft.",
            bad_analyst,
            critic_pass,
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
    assert response.answer == BLOCKED_ANSWER_FALLBACK
    analyst_report = response.traces[1].guardrail_report
    assert analyst_report is not None
    assert analyst_report.verdict == GuardrailVerdict.BLOCK
    assert "radicalization" in analyst_report.flags


@pytest.mark.asyncio
async def test_multi_orchestrator_uses_critic_revision_when_warn(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    critic_warn = (
        "VERDICT: warn\n"
        "FLAGS: missing_citation\n"
        "NOTES: needs citation\n"
        "REVISED_ANSWER:\n## Summary\nClean revised answer [1].\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft.",
            "## Summary\nOriginal analyst.",
            critic_warn,
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

    assert "Clean revised answer" in response.answer
    analyst_report = response.traces[1].guardrail_report
    assert analyst_report is not None
    assert analyst_report.verdict == GuardrailVerdict.WARN
    assert "missing_citation" in analyst_report.flags


@pytest.mark.asyncio
async def test_multi_orchestrator_redacts_pii_in_critic_revision(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    # Critic 'fixes' the analyst answer but smuggles in an email address.
    critic_warn = (
        "VERDICT: warn\n"
        "FLAGS: tone\n"
        "NOTES: tone\n"
        "REVISED_ANSWER:\n## Summary\nReach out at fixed@example.com about RAG.\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft.",
            "## Summary\nOriginal analyst (clean).",
            critic_warn,
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

    assert "[REDACTED:EMAIL]" in response.answer
    assert "fixed@example.com" not in response.answer
    critic_report = response.traces[2].guardrail_report
    assert critic_report is not None
    assert critic_report.verdict == GuardrailVerdict.BLOCK
    assert "pii" in critic_report.flags


@pytest.mark.asyncio
async def test_multi_orchestrator_attaches_monitor_report_to_researcher(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    critic_pass = (
        "VERDICT: pass\nFLAGS: none\nNOTES: ok\nREVISED_ANSWER:\nUNCHANGED\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft mentioning RAG and embeddings [1].",
            "## Summary\nAnalyst text.",
            critic_pass,
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

    researcher_trace = response.traces[0]
    assert researcher_trace.guardrail_report is not None
    assert researcher_trace.guardrail_report.verdict == GuardrailVerdict.PASS
    assert researcher_trace.guardrail_report.monitor_verdict == GuardrailVerdict.PASS


@pytest.mark.asyncio
async def test_multi_orchestrator_custom_monitor_is_used(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)

    seen_roles: list = []

    class TrackingMonitor(GuardrailMonitor):
        def inspect(self, text, *, agent_role, context=None):
            seen_roles.append(agent_role)
            return super().inspect(text, agent_role=agent_role, context=context)

    critic_pass = (
        "VERDICT: pass\nFLAGS: none\nNOTES: ok\nREVISED_ANSWER:\nUNCHANGED\n"
    )
    llm = MockLLM(
        response_fn=_role_router(
            "Researcher draft.",
            "## Summary\nAnalyst.",
            critic_pass,
        )
    )
    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    orch = MultiAgentOrchestrator(
        retriever=retriever,
        researcher=ResearcherAgent(llm),
        analyst=AnalystAgent(llm),
        critic=CriticAgent(llm),
        default_top_k=3,
        guardrail_monitor=TrackingMonitor(),
    )

    await orch.run(query="What is RAG?")

    # Researcher + analyst + final-answer (after critic) -> 3 inspections
    assert len(seen_roles) == 3
