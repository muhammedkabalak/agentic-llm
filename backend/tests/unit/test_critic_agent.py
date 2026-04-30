"""Unit tests for CriticAgent and its output parser."""

from __future__ import annotations

import pytest

from app.agents.critic_agent import CriticAgent, parse_critic_output
from app.models.domain import AgentContext, DocumentChunk
from app.models.schemas import AgentRole, GuardrailVerdict

from tests.conftest import MockLLM


# --------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------- #
def test_parser_pass_with_unchanged_revision() -> None:
    raw = (
        "VERDICT: pass\n"
        "FLAGS: none\n"
        "NOTES: Looks great.\n"
        "REVISED_ANSWER:\n"
        "UNCHANGED\n"
    )
    review = parse_critic_output(raw, fallback="ANALYST_OUTPUT")
    assert review.verdict == GuardrailVerdict.PASS
    assert review.flags == []
    assert "Looks great" in review.notes
    assert review.revised_answer == "ANALYST_OUTPUT"


def test_parser_warn_with_flags_and_revision() -> None:
    raw = (
        "VERDICT: warn\n"
        "FLAGS: missing_citation, vague_claim\n"
        "NOTES: Add citation in point 2.\n"
        "REVISED_ANSWER:\n"
        "## Summary\nFixed answer.\n"
    )
    review = parse_critic_output(raw, fallback="ORIG")
    assert review.verdict == GuardrailVerdict.WARN
    assert review.flags == ["missing_citation", "vague_claim"]
    assert review.revised_answer.startswith("## Summary")
    assert review.revised_answer.strip().endswith("Fixed answer.")


def test_parser_block_uppercase_value() -> None:
    raw = (
        "VERDICT: BLOCK\n"
        "FLAGS: hallucination\n"
        "NOTES: Unsupported claim.\n"
        "REVISED_ANSWER:\n"
        "I cannot answer this from the provided sources.\n"
    )
    review = parse_critic_output(raw, fallback="ORIG")
    assert review.verdict == GuardrailVerdict.BLOCK
    assert review.flags == ["hallucination"]


def test_parser_degrades_to_warn_on_garbage() -> None:
    raw = "this is not a verdict block"
    review = parse_critic_output(raw, fallback="ORIG")
    assert review.verdict == GuardrailVerdict.WARN
    assert review.flags == []
    # Empty revised block falls back
    assert review.revised_answer == "ORIG"


# --------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_critic_role_and_system_prompt() -> None:
    agent = CriticAgent(MockLLM())
    assert agent.role == AgentRole.CRITIC

    sp = agent.system_prompt
    assert "Critic Agent" in sp
    assert "VERDICT:" in sp
    assert "FLAGS:" in sp
    assert "REVISED_ANSWER:" in sp


def test_critic_user_prompt_includes_analyst_and_context() -> None:
    agent = CriticAgent(MockLLM())
    ctx = AgentContext(
        query="What is RAG?",
        retrieved_chunks=[DocumentChunk(content="grounded fact", source="g")],
    )
    ctx.intermediate_outputs[AgentRole.ANALYST.value] = "## Summary\nA.\n"

    user_prompt = agent.build_user_prompt(ctx)
    assert "What is RAG?" in user_prompt
    assert "## Summary" in user_prompt
    assert "grounded fact" in user_prompt


@pytest.mark.asyncio
async def test_critic_run_emits_structured_text() -> None:
    canned = (
        "VERDICT: pass\n"
        "FLAGS: none\n"
        "NOTES: Solid.\n"
        "REVISED_ANSWER:\nUNCHANGED\n"
    )
    llm = MockLLM(canned_response=canned)
    agent = CriticAgent(llm)

    ctx = AgentContext(
        query="q",
        retrieved_chunks=[DocumentChunk(content="c", source="s")],
    )
    ctx.intermediate_outputs[AgentRole.ANALYST.value] = "analyst body"

    result = await agent.run(ctx)
    assert result.agent_role == AgentRole.CRITIC.value
    assert result.output == canned
    # Round-trip through parser
    review = parse_critic_output(result.output, fallback="analyst body")
    assert review.verdict == GuardrailVerdict.PASS
    assert review.revised_answer == "analyst body"
