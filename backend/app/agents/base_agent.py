"""
BaseAgent — abstract foundation for every agent in the system.

Every concrete agent (Researcher, Analyst, Critic, …) inherits from
`BaseAgent` to guarantee a uniform interface:

    result = await agent.run(context)

Responsibilities of the base class:
  * Hold a reference to the configured LLM provider.
  * Build chat-style prompts from a system prompt + user query + RAG context.
  * Provide lifecycle hooks (`before_run`, `after_run`) subclasses can override.
  * Measure latency and token usage automatically.
  * Emit structured logs for observability.

Concrete agents only need to implement:
  * `role`            — agent identity (e.g. AgentRole.RESEARCHER)
  * `system_prompt`   — the agent's persona / instructions
  * `build_user_prompt(context)` — how to assemble the user message
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional

from app.models.domain import AgentContext, AgentResult, DocumentChunk
from app.models.schemas import AgentRole
from app.services.llm_provider import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
)
from app.services.logging_service import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        *,
        name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.llm = llm_provider
        self.name = name or self.__class__.__name__
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """The semantic role this agent plays in the crew."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's persona/instructions; injected as the system message."""

    @abstractmethod
    def build_user_prompt(self, context: AgentContext) -> str:
        """Assemble the user message from the shared context."""

    # ------------------------------------------------------------------ #
    # Lifecycle hooks (override as needed)
    # ------------------------------------------------------------------ #
    async def before_run(self, context: AgentContext) -> None:
        """Hook called before the LLM invocation."""
        return None

    async def after_run(
        self, context: AgentContext, response: LLMResponse
    ) -> None:
        """Hook called after the LLM invocation, before returning."""
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_chunks(chunks: List[DocumentChunk]) -> str:
        """Render retrieved chunks as a readable context block."""
        if not chunks:
            return "(no retrieved context)"
        lines = []
        for i, c in enumerate(chunks, start=1):
            src = f" — source: {c.source}" if c.source else ""
            lines.append(f"[{i}]{src}\n{c.content.strip()}")
        return "\n\n".join(lines)

    def _build_messages(self, context: AgentContext) -> list[LLMMessage]:
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self.system_prompt)
        ]
        # Optional conversation history
        for h in context.history:
            role = h.get("role", "user")
            content = h.get("content", "")
            if content:
                messages.append(LLMMessage(role=role, content=content))
        # The current task
        messages.append(
            LLMMessage(role="user", content=self.build_user_prompt(context))
        )
        return messages

    # ------------------------------------------------------------------ #
    # Public entry-point
    # ------------------------------------------------------------------ #
    async def run(self, context: AgentContext) -> AgentResult:
        """Execute the agent against the shared context."""
        await self.before_run(context)

        messages = self._build_messages(context)
        started = time.perf_counter()

        logger.info(
            "agent_run_start",
            agent=self.name,
            role=self.role.value,
            request_id=str(context.request_id),
            chunks_in_context=len(context.retrieved_chunks),
        )

        response = await self.llm.generate(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)

        await self.after_run(context, response)

        # Persist this agent's output for downstream agents
        context.intermediate_outputs[self.role.value] = response.content

        logger.info(
            "agent_run_end",
            agent=self.name,
            role=self.role.value,
            request_id=str(context.request_id),
            latency_ms=latency_ms,
            tokens=response.total_tokens,
        )

        return AgentResult(
            agent_role=self.role.value,
            output=response.content,
            tokens_used=response.total_tokens,
            latency_ms=latency_ms,
            metadata={
                "model": response.model,
                "provider": response.provider,
            },
        )
