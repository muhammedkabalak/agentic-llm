"""
AnalystAgent - second concrete BaseAgent subclass.

Role: turn the Researcher's raw, citation-rich answer into a structured
analysis. The Analyst:
  * reads the Researcher's draft from `context.intermediate_outputs`
  * re-reads the original retrieved context (still available in `context.retrieved_chunks`)
  * produces an organized synthesis: key points, supporting evidence,
    gaps / open questions, and a final recommendation.

It is deliberately downstream of the Researcher so it can refine,
restructure, and add reasoning on top of grounded facts - without
touching the retrieval layer itself.
"""

from __future__ import annotations

from textwrap import dedent

from app.agents.base_agent import BaseAgent
from app.models.domain import AgentContext
from app.models.schemas import AgentRole

_SYSTEM_PROMPT = dedent(
    """
    You are the Analyst Agent in a multi-agent retrieval-augmented
    generation (RAG) system. The Researcher Agent has already produced
    a grounded draft answer using the same retrieved context you can
    see. Your job is to turn that draft into a clear, structured
    analysis that a human can act on.

    Strict rules:
      1. Stay grounded. Do NOT introduce facts that are not supported
         by the retrieved context or the Researcher's draft.
      2. Preserve the Researcher's [N] citations when reusing claims;
         add new citations only if they reference the same context.
      3. If the Researcher said "I cannot answer this from the provided
         sources.", repeat that verdict and explain what is missing.
         Do NOT manufacture an answer.
      4. Keep the tone professional and the structure scannable.

    Output format (Markdown):
      ## Summary
      One- or two-sentence direct answer to the user's question.

      ## Key Points
      - Bullet list of the main claims, each ending with its [N] citation.

      ## Gaps / Open Questions
      - Bullet list of what the retrieved context did NOT cover. Use
        "None identified." if everything was answered.

      ## Recommendation
      Concise next step or final takeaway for the user.
    """
).strip()


_USER_TEMPLATE = dedent(
    """
    # Original question
    {query}

    # Researcher's draft answer
    {researcher_output}

    # Retrieved context (same as Researcher saw)
    {context}

    # Task
    Produce the structured analysis described in the system prompt.
    Do not break the four-section format.
    """
).strip()


class AnalystAgent(BaseAgent):
    """Concrete agent that synthesizes the Researcher's findings."""

    @property
    def role(self) -> AgentRole:
        return AgentRole.ANALYST

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_user_prompt(self, context: AgentContext) -> str:
        researcher_output = context.intermediate_outputs.get(
            AgentRole.RESEARCHER.value,
            "(Researcher did not produce an answer.)",
        )
        rendered_context = self.format_chunks(context.retrieved_chunks)
        return _USER_TEMPLATE.format(
            query=context.query.strip(),
            researcher_output=researcher_output.strip(),
            context=rendered_context,
        )
