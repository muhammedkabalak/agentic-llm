"""
Unit tests for ResearcherAgent.

These tests verify:
  * Agent identity (role, system prompt content).
  * User-prompt assembly: query + retrieved context (with [N] indices).
  * The empty-context fallback path is reachable (the system prompt
    instructs the model to refuse — we verify the prompt structure
    is correct so the LLM has the cue it needs).
  * The agent forwards the LLM response into AgentResult correctly.
"""

from __future__ import annotations

import pytest

from app.agents.researcher_agent import ResearcherAgent
from app.models.domain import AgentContext, DocumentChunk
from app.models.schemas import AgentRole

from tests.conftest import MockLLM


@pytest.mark.asyncio
async def test_researcher_role_and_system_prompt() -> None:
    agent = ResearcherAgent(MockLLM())

    assert agent.role == AgentRole.RESEARCHER

    sp = agent.system_prompt
    assert "Researcher Agent" in sp
    # Must forbid hallucination
    assert "Do NOT invent" in sp or "do not invent" in sp.lower()
    # Must require bracketed citations
    assert "[1]" in sp
    # Must contain the explicit refusal string
    assert "I cannot answer this from the provided sources." in sp


@pytest.mark.asyncio
async def test_researcher_user_prompt_includes_query_and_chunks() -> None:
    agent = ResearcherAgent(MockLLM())

    ctx = AgentContext(
        query="What is RAG?",
        retrieved_chunks=[
            DocumentChunk(
                content="RAG combines retrieval with generation.",
                source="doc-a.md",
            ),
            DocumentChunk(
                content="It reduces hallucination by grounding answers.",
                source="doc-b.md",
            ),
        ],
    )

    user_prompt = agent.build_user_prompt(ctx)

    # Question echoed verbatim
    assert "What is RAG?" in user_prompt
    # Both chunks rendered with bracketed indices
    assert "[1]" in user_prompt and "[2]" in user_prompt
    # Source filenames present so the model can cite them
    assert "doc-a.md" in user_prompt
    assert "doc-b.md" in user_prompt
    # Chunk content present
    assert "combines retrieval with generation" in user_prompt
    assert "reduces hallucination" in user_prompt


@pytest.mark.asyncio
async def test_researcher_user_prompt_when_no_chunks() -> None:
    agent = ResearcherAgent(MockLLM())

    ctx = AgentContext(query="Anything?", retrieved_chunks=[])
    user_prompt = agent.build_user_prompt(ctx)

    assert "Anything?" in user_prompt
    # Empty-context placeholder must be present so the LLM sees it
    assert "no retrieved context" in user_prompt


@pytest.mark.asyncio
async def test_researcher_run_returns_llm_output() -> None:
    llm = MockLLM(canned_response="RAG = retrieval-augmented generation [1].")
    agent = ResearcherAgent(llm)

    ctx = AgentContext(
        query="Define RAG",
        retrieved_chunks=[
            DocumentChunk(content="RAG = retrieval-augmented generation.", source="x"),
        ],
    )

    result = await agent.run(ctx)

    assert result.agent_role == AgentRole.RESEARCHER.value
    assert result.output == "RAG = retrieval-augmented generation [1]."
    assert result.tokens_used == 33  # 11 + 22 (defaults from MockLLM)
    # Recorded for downstream agents
    assert ctx.intermediate_outputs[AgentRole.RESEARCHER.value] == result.output

    # The LLM saw a system + user message in that order
    assert len(llm.calls) == 1
    sent = llm.calls[0]
    assert sent[0].role == "system"
    assert "Researcher Agent" in sent[0].content
    assert sent[-1].role == "user"
    assert "Define RAG" in sent[-1].content


@pytest.mark.asyncio
async def test_researcher_run_with_history_passes_through() -> None:
    llm = MockLLM(canned_response="ok")
    agent = ResearcherAgent(llm)

    ctx = AgentContext(
        query="Follow-up question?",
        history=[
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ],
        retrieved_chunks=[DocumentChunk(content="ctx", source="s")],
    )

    await agent.run(ctx)

    sent = llm.calls[0]
    # system + 2 history + 1 user = 4
    assert len(sent) == 4
    assert sent[0].role == "system"
    assert sent[1].role == "user" and sent[1].content == "first question"
    assert sent[2].role == "assistant" and sent[2].content == "first answer"
    assert sent[3].role == "user"
    assert "Follow-up question?" in sent[3].content
