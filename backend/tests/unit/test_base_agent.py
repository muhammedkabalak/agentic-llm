"""Unit tests for the BaseAgent abstraction (uses a mock LLM provider)."""

from __future__ import annotations

import pytest

from app.agents.base_agent import BaseAgent
from app.config import Settings
from app.models.domain import AgentContext, DocumentChunk
from app.models.schemas import AgentRole
from app.services.llm_provider import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
)


class _MockLLM(BaseLLMProvider):
    """Deterministic provider used only in tests."""

    def __init__(self) -> None:
        # Build a minimal Settings instance without touching env / disk
        self.settings = Settings.model_construct(
            llm_model="mock-model",
            llm_temperature=0.0,
            llm_max_tokens=128,
            llm_timeout_seconds=10,
        )
        self.model = "mock-model"
        self.temperature = 0.0
        self.max_tokens = 128
        self.timeout = 10
        self.last_messages: list[LLMMessage] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def _generate(self, messages, *, temperature=None, max_tokens=None, **kwargs):
        self.last_messages = list(messages)
        return LLMResponse(
            content="MOCK_OUTPUT",
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


class _DummyAgent(BaseAgent):
    @property
    def role(self) -> AgentRole:
        return AgentRole.RESEARCHER

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."

    def build_user_prompt(self, context: AgentContext) -> str:
        return f"QUERY: {context.query}\nCTX: {self.format_chunks(context.retrieved_chunks)}"


@pytest.mark.asyncio
async def test_base_agent_runs_and_records_output() -> None:
    llm = _MockLLM()
    agent = _DummyAgent(llm)

    ctx = AgentContext(
        query="What is RAG?",
        retrieved_chunks=[
            DocumentChunk(content="RAG = Retrieval Augmented Generation.", source="doc1"),
        ],
    )

    result = await agent.run(ctx)

    assert result.agent_role == AgentRole.RESEARCHER.value
    assert result.output == "MOCK_OUTPUT"
    assert result.tokens_used == 15
    assert ctx.intermediate_outputs[AgentRole.RESEARCHER.value] == "MOCK_OUTPUT"

    # System prompt + user prompt were built correctly
    assert llm.last_messages[0].role == "system"
    assert "test agent" in llm.last_messages[0].content
    assert llm.last_messages[-1].role == "user"
    assert "What is RAG?" in llm.last_messages[-1].content
    assert "RAG = Retrieval" in llm.last_messages[-1].content
