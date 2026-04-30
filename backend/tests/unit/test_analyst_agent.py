"""Unit tests for AnalystAgent."""

from __future__ import annotations

import pytest

from app.agents.analyst_agent import AnalystAgent
from app.models.domain import AgentContext, DocumentChunk
from app.models.schemas import AgentRole

from tests.conftest import MockLLM


@pytest.mark.asyncio
async def test_analyst_role_and_system_prompt() -> None:
    agent = AnalystAgent(MockLLM())

    assert agent.role == AgentRole.ANALYST
    sp = agent.system_prompt
    assert "Analyst Agent" in sp
    # Must reference the Researcher upstream
    assert "Researcher" in sp
    # Must enforce grounding
    assert "grounded" in sp.lower() or "do not introduce" in sp.lower()
    # Must define the four-section output structure
    for header in ("## Summary", "## Key Points", "## Gaps", "## Recommendation"):
        assert header in sp


def test_analyst_user_prompt_pulls_researcher_output() -> None:
    agent = AnalystAgent(MockLLM())
    ctx = AgentContext(
        query="What is RAG?",
        retrieved_chunks=[DocumentChunk(content="RAG = retrieval+gen.", source="x")],
    )
    ctx.intermediate_outputs[AgentRole.RESEARCHER.value] = (
        "RAG combines retrieval with generation [1]."
    )

    user_prompt = agent.build_user_prompt(ctx)
    assert "What is RAG?" in user_prompt
    assert "RAG combines retrieval with generation [1]." in user_prompt
    assert "[1]" in user_prompt
    assert "RAG = retrieval+gen." in user_prompt


def test_analyst_user_prompt_falls_back_when_researcher_absent() -> None:
    agent = AnalystAgent(MockLLM())
    ctx = AgentContext(
        query="Anything?",
        retrieved_chunks=[DocumentChunk(content="ctx", source="s")],
    )
    user_prompt = agent.build_user_prompt(ctx)
    assert "Researcher did not produce" in user_prompt


@pytest.mark.asyncio
async def test_analyst_run_records_output_for_critic() -> None:
    llm = MockLLM(canned_response="## Summary\nDone.\n")
    agent = AnalystAgent(llm)

    ctx = AgentContext(
        query="q",
        retrieved_chunks=[DocumentChunk(content="c", source="s")],
    )
    ctx.intermediate_outputs[AgentRole.RESEARCHER.value] = "researcher draft"

    result = await agent.run(ctx)

    assert result.agent_role == AgentRole.ANALYST.value
    assert result.output == "## Summary\nDone.\n"
    assert ctx.intermediate_outputs[AgentRole.ANALYST.value] == result.output

    # Sanity: the LLM saw the system prompt + user prompt referencing the researcher
    sent = llm.calls[0]
    assert sent[0].role == "system"
    assert "Analyst Agent" in sent[0].content
    assert "researcher draft" in sent[-1].content
