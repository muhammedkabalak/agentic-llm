"""
Orchestration layer.

Two pipelines live here:

* `SingleAgentPipeline` (Step 3) - one retrieval + one agent, the
  simplest end-to-end RAG flow.

* `MultiAgentOrchestrator` (Step 4) - the project's multi-agent
  framework. Sequentially runs Researcher -> Analyst -> (optional)
  Critic over a SHARED `AgentContext` so each agent reads the previous
  agent's output via `context.intermediate_outputs`.

Both pipelines also run a **GuardrailMonitor** (Step 5) on every
agent's output. The monitor is independent of the Critic agent: it is
a deterministic, LLM-free auditor that flags PII, hallucinations,
toxicity, bias, radicalisation, prompt-injection echoes, and
refusal-overreach. Its findings are attached to each `AgentTrace` via
`GuardrailReport`. The orchestrator merges the Critic's verdict with
the monitor's verdict into one final `GuardrailReport` for the
Analyst/Critic traces and lets the monitor's BLOCK verdict short-circuit
the final answer.

Final-answer policy (applied by both pipelines):
  1. PII detected      -> serve the redacted variant (preserves the
                          message, replaces PII with `[REDACTED:KIND]`).
  2. Non-PII BLOCK     -> serve `BLOCKED_ANSWER_FALLBACK` (radicalisation,
                          prompt-injection, etc).
  3. Critic non-PASS   -> serve the Critic's revised answer.
  4. Otherwise         -> serve the agent's raw output.
"""

from __future__ import annotations

import time
from typing import List, Optional
from uuid import UUID

from app.agents.base_agent import BaseAgent
from app.agents.critic_agent import CriticAgent, parse_critic_output
from app.guardrails.monitor import (
    GuardrailMonitor,
    GuardrailMonitorReport,
    merge_verdicts,
)
from app.models.domain import AgentContext, DocumentChunk
from app.models.schemas import (
    AgentRole,
    AgentTrace,
    ChatMessage,
    ChatResponse,
    GuardrailReport,
    GuardrailVerdict,
    RetrievedChunk,
)
from app.rag.retriever import Retriever
from app.services.logging_service import get_logger

logger = get_logger(__name__)

BLOCKED_ANSWER_FALLBACK = (
    "I can't share that response - it failed our safety review. "
    "Please rephrase your question or ask for a different angle."
)


def _to_history_dicts(history: Optional[List[ChatMessage]]) -> list[dict[str, str]]:
    return [
        {"role": m.role.value, "content": m.content} for m in (history or [])
    ]


def _to_retrieved_chunks(chunks: List[DocumentChunk]) -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            content=c.content,
            source=c.source,
            score=c.score,
            metadata=c.metadata,
        )
        for c in chunks
    ]


def _monitor_to_report(
    monitor_report: GuardrailMonitorReport,
) -> GuardrailReport:
    """Convert a monitor-only report into the wire-level GuardrailReport."""
    return GuardrailReport(
        verdict=monitor_report.verdict,
        flags=monitor_report.flags,
        notes=monitor_report.summary(),
        findings=list(monitor_report.findings),
        monitor_verdict=monitor_report.verdict,
        redacted_text=monitor_report.redacted_text,
    )


def _merge_monitor_with_critic(
    *,
    critic_verdict: GuardrailVerdict,
    critic_flags: List[str],
    critic_notes: Optional[str],
    monitor: GuardrailMonitorReport,
) -> GuardrailReport:
    """Combine Critic-agent review and GuardrailMonitor findings."""
    final_verdict = merge_verdicts(critic_verdict, monitor.verdict)

    seen: set[str] = set()
    merged_flags: List[str] = []
    for tag in (*critic_flags, *monitor.flags):
        if tag not in seen:
            seen.add(tag)
            merged_flags.append(tag)

    notes_parts: List[str] = []
    if critic_notes:
        notes_parts.append(f"Critic: {critic_notes}")
    if monitor.findings:
        notes_parts.append(monitor.summary())

    return GuardrailReport(
        verdict=final_verdict,
        flags=merged_flags,
        notes=" | ".join(notes_parts) if notes_parts else None,
        findings=list(monitor.findings),
        monitor_verdict=monitor.verdict,
        redacted_text=monitor.redacted_text,
    )


def _select_final_answer(
    *,
    candidate: str,
    primary_monitor: GuardrailMonitorReport,
    upstream_monitor: Optional[GuardrailMonitorReport] = None,
) -> str:
    """
    Apply the final-answer policy:

      * primary_monitor.redacted_text -> redacted variant wins,
      * else any BLOCK on primary or upstream -> fallback,
      * else -> candidate is fine.
    """
    if primary_monitor.redacted_text is not None:
        return primary_monitor.redacted_text
    if primary_monitor.verdict == GuardrailVerdict.BLOCK:
        return BLOCKED_ANSWER_FALLBACK
    if (
        upstream_monitor is not None
        and upstream_monitor.verdict == GuardrailVerdict.BLOCK
    ):
        return BLOCKED_ANSWER_FALLBACK
    return candidate


# --------------------------------------------------------------------- #
# Single-agent pipeline (Step 3) - now monitor-aware
# --------------------------------------------------------------------- #
class SingleAgentPipeline:
    """One Retriever + one Agent -> one ChatResponse."""

    def __init__(
        self,
        retriever: Retriever,
        agent: BaseAgent,
        *,
        default_top_k: int = 5,
        guardrail_monitor: Optional[GuardrailMonitor] = None,
    ) -> None:
        self.retriever = retriever
        self.agent = agent
        self.default_top_k = default_top_k
        self.guardrail_monitor = guardrail_monitor or GuardrailMonitor()

    async def run(
        self,
        query: str,
        *,
        history: Optional[List[ChatMessage]] = None,
        session_id: Optional[UUID] = None,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        k = top_k or self.default_top_k

        retrieved: List[DocumentChunk] = self.retriever.retrieve(query, top_k=k)

        context = AgentContext(
            session_id=session_id,
            query=query,
            history=_to_history_dicts(history),
            retrieved_chunks=retrieved,
        )

        logger.info(
            "single_agent_pipeline_run",
            session_id=str(session_id) if session_id else None,
            query_preview=query[:80],
            n_chunks=len(retrieved),
        )

        result = await self.agent.run(context)
        sources = _to_retrieved_chunks(retrieved)

        monitor_report = self.guardrail_monitor.inspect(
            result.output,
            agent_role=self.agent.role,
            context=retrieved,
        )

        final_answer = _select_final_answer(
            candidate=result.output,
            primary_monitor=monitor_report,
        )

        trace = AgentTrace(
            agent_role=self.agent.role,
            input=query,
            output=result.output,
            retrieved_chunks=sources,
            guardrail_report=_monitor_to_report(monitor_report),
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
        )

        total_latency = int((time.perf_counter() - started) * 1000)

        return ChatResponse(
            request_id=context.request_id,
            session_id=session_id,
            answer=final_answer,
            traces=[trace],
            sources=sources,
            total_tokens=result.tokens_used,
            total_latency_ms=total_latency,
        )


# --------------------------------------------------------------------- #
# Multi-agent orchestrator (Step 4 + Step 5 monitoring)
# --------------------------------------------------------------------- #
class MultiAgentOrchestrator:
    """
    Sequential orchestrator: Researcher -> Analyst -> (optional) Critic,
    with the GuardrailMonitor inspecting every agent output.
    """

    def __init__(
        self,
        retriever: Retriever,
        researcher: BaseAgent,
        analyst: BaseAgent,
        critic: Optional[CriticAgent] = None,
        *,
        default_top_k: int = 5,
        enable_critic: bool = True,
        guardrail_monitor: Optional[GuardrailMonitor] = None,
    ) -> None:
        self.retriever = retriever
        self.researcher = researcher
        self.analyst = analyst
        self.critic = critic
        self.default_top_k = default_top_k
        self.enable_critic = enable_critic
        self.guardrail_monitor = guardrail_monitor or GuardrailMonitor()

    async def run(
        self,
        query: str,
        *,
        history: Optional[List[ChatMessage]] = None,
        session_id: Optional[UUID] = None,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        k = top_k or self.default_top_k

        retrieved: List[DocumentChunk] = self.retriever.retrieve(query, top_k=k)
        sources = _to_retrieved_chunks(retrieved)

        context = AgentContext(
            session_id=session_id,
            query=query,
            history=_to_history_dicts(history),
            retrieved_chunks=retrieved,
        )

        critic_enabled = self.enable_critic and self.critic is not None
        logger.info(
            "multi_agent_orchestrator_run",
            session_id=str(session_id) if session_id else None,
            query_preview=query[:80],
            n_chunks=len(retrieved),
            critic_enabled=critic_enabled,
        )

        traces: List[AgentTrace] = []
        total_tokens = 0

        # 1. Researcher
        researcher_result = await self.researcher.run(context)
        total_tokens += researcher_result.tokens_used
        researcher_monitor = self.guardrail_monitor.inspect(
            researcher_result.output,
            agent_role=self.researcher.role,
            context=retrieved,
        )
        traces.append(
            AgentTrace(
                agent_role=self.researcher.role,
                input=query,
                output=researcher_result.output,
                retrieved_chunks=sources,
                guardrail_report=_monitor_to_report(researcher_monitor),
                latency_ms=researcher_result.latency_ms,
                tokens_used=researcher_result.tokens_used,
            )
        )

        # 2. Analyst
        analyst_result = await self.analyst.run(context)
        total_tokens += analyst_result.tokens_used
        analyst_monitor = self.guardrail_monitor.inspect(
            analyst_result.output,
            agent_role=self.analyst.role,
            context=retrieved,
        )

        # 3. Critic (optional)
        final_answer = analyst_result.output

        if critic_enabled and self.critic is not None:
            critic_result = await self.critic.run(context)
            total_tokens += critic_result.tokens_used

            review = parse_critic_output(
                critic_result.output, fallback=analyst_result.output
            )

            if review.verdict != GuardrailVerdict.PASS:
                final_answer = review.revised_answer

            # Re-monitor the answer the user will actually see, because
            # the Critic's revised version is new text we haven't audited.
            final_monitor = self.guardrail_monitor.inspect(
                final_answer,
                agent_role=AgentRole.CRITIC,
                context=retrieved,
            )

            analyst_report = _merge_monitor_with_critic(
                critic_verdict=review.verdict,
                critic_flags=review.flags,
                critic_notes=review.notes,
                monitor=analyst_monitor,
            )
            critic_report = _merge_monitor_with_critic(
                critic_verdict=review.verdict,
                critic_flags=review.flags,
                critic_notes=review.notes,
                monitor=final_monitor,
            )

            final_answer = _select_final_answer(
                candidate=final_answer,
                primary_monitor=final_monitor,
                upstream_monitor=analyst_monitor,
            )

            traces.append(
                AgentTrace(
                    agent_role=self.analyst.role,
                    input=researcher_result.output,
                    output=analyst_result.output,
                    retrieved_chunks=sources,
                    guardrail_report=analyst_report,
                    latency_ms=analyst_result.latency_ms,
                    tokens_used=analyst_result.tokens_used,
                )
            )
            traces.append(
                AgentTrace(
                    agent_role=self.critic.role,
                    input=analyst_result.output,
                    output=critic_result.output,
                    retrieved_chunks=sources,
                    guardrail_report=critic_report,
                    latency_ms=critic_result.latency_ms,
                    tokens_used=critic_result.tokens_used,
                )
            )
        else:
            # No critic: the monitor is the only safety net.
            analyst_report = _monitor_to_report(analyst_monitor)
            final_answer = _select_final_answer(
                candidate=final_answer,
                primary_monitor=analyst_monitor,
            )

            traces.append(
                AgentTrace(
                    agent_role=self.analyst.role,
                    input=researcher_result.output,
                    output=analyst_result.output,
                    retrieved_chunks=sources,
                    guardrail_report=analyst_report,
                    latency_ms=analyst_result.latency_ms,
                    tokens_used=analyst_result.tokens_used,
                )
            )

        total_latency = int((time.perf_counter() - started) * 1000)

        return ChatResponse(
            request_id=context.request_id,
            session_id=session_id,
            answer=final_answer,
            traces=traces,
            sources=sources,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
        )
