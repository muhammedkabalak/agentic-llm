"""
CriticAgent - third concrete BaseAgent subclass.

Role: independent reviewer. The Critic re-reads the Analyst's output
against the SAME retrieved context, then issues a structured verdict:
    * VERDICT: pass | warn | block
    * FLAGS:   short tags (e.g. "unsupported_claim", "missing_citation")
    * NOTES:   one-paragraph rationale
    * REVISED_ANSWER: a corrected/tightened version of the Analyst's
                      output the orchestrator can choose to use.

Why a separate agent? It enforces the project's "guardrails" goal
without leaking that responsibility into the Researcher or Analyst -
each agent has a single, auditable job. The Step-5 Guardrails Monitor
will live alongside this and use the Critic's flags as one signal.

The Critic's output deliberately starts with `VERDICT: <value>` on the
first line so the orchestrator can parse it deterministically without
asking the LLM for JSON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from textwrap import dedent
from typing import List

from app.agents.base_agent import BaseAgent
from app.models.domain import AgentContext
from app.models.schemas import AgentRole, GuardrailVerdict

_SYSTEM_PROMPT = dedent(
    """
    You are the Critic Agent in a multi-agent retrieval-augmented
    generation (RAG) system. Your job is to AUDIT the Analyst's answer
    against the retrieved context. You do NOT produce a brand-new answer;
    you verify and, where needed, propose a tightened revision.

    Audit checklist:
      1. Grounding: every factual claim must trace to the retrieved
         context. Flag any claim that does not.
      2. Citations: claims that should cite a source must use [N]
         indices that match the context block.
      3. Refusal honesty: if the context does not support an answer,
         the Analyst should say so plainly. Flag overreach.
      4. Structure: the Analyst's four-section Markdown layout should
         be preserved.

    Output format (plain text, line-anchored - DO NOT add extra prose
    before or after):

    VERDICT: <pass|warn|block>
    FLAGS: <comma-separated short tags, or "none">
    NOTES: <one short paragraph explaining the verdict>
    REVISED_ANSWER:
    <the Analyst's output, lightly edited if needed, or the literal
    word "UNCHANGED" on its own line if no edit is required>

    Verdict scale:
      * pass  - fully grounded, well-cited, well-structured.
      * warn  - mostly fine but contains minor issues you flagged.
      * block - contains unsupported claims or hallucination; the
                REVISED_ANSWER should remove or refute them.
    """
).strip()

_USER_TEMPLATE = dedent(
    """
    # Original question
    {query}

    # Retrieved context (the only ground truth)
    {context}

    # Analyst's answer (under audit)
    {analyst_output}

    # Task
    Audit the Analyst's answer and emit the four-line VERDICT/FLAGS/NOTES/
    REVISED_ANSWER block exactly as specified.
    """
).strip()


# --------------------------------------------------------------------- #
# Parsed critic output (consumed by the orchestrator / guardrails)
# --------------------------------------------------------------------- #
@dataclass
class CriticReview:
    verdict: GuardrailVerdict
    flags: List[str]
    notes: str
    revised_answer: str
    raw: str


_VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(pass|warn|block)\s*$", re.IGNORECASE)
_FLAGS_RE = re.compile(r"^\s*FLAGS:\s*(.*)$", re.IGNORECASE)
_NOTES_RE = re.compile(r"^\s*NOTES:\s*(.*)$", re.IGNORECASE)
_REVISED_HDR_RE = re.compile(r"^\s*REVISED_ANSWER:\s*$", re.IGNORECASE)


def parse_critic_output(text: str, *, fallback: str) -> CriticReview:
    """
    Best-effort parser for the Critic's structured output. If the
    model misbehaves we degrade gracefully to a "warn" verdict so
    the system stays usable.
    """
    lines = text.splitlines()
    verdict = GuardrailVerdict.WARN
    flags: List[str] = []
    notes = ""
    revised_lines: List[str] = []
    in_revised = False

    for line in lines:
        if in_revised:
            revised_lines.append(line)
            continue
        m = _VERDICT_RE.match(line)
        if m:
            verdict = GuardrailVerdict(m.group(1).lower())
            continue
        m = _FLAGS_RE.match(line)
        if m:
            raw = m.group(1).strip()
            if raw and raw.lower() != "none":
                flags = [f.strip() for f in raw.split(",") if f.strip()]
            continue
        m = _NOTES_RE.match(line)
        if m:
            notes = m.group(1).strip()
            continue
        if _REVISED_HDR_RE.match(line):
            in_revised = True
            continue

    revised_block = "\n".join(revised_lines).strip()
    if not revised_block or revised_block.upper() == "UNCHANGED":
        revised_answer = fallback
    else:
        revised_answer = revised_block

    return CriticReview(
        verdict=verdict,
        flags=flags,
        notes=notes,
        revised_answer=revised_answer,
        raw=text,
    )


class CriticAgent(BaseAgent):
    """Concrete agent that audits the Analyst's answer."""

    @property
    def role(self) -> AgentRole:
        return AgentRole.CRITIC

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_user_prompt(self, context: AgentContext) -> str:
        analyst_output = context.intermediate_outputs.get(
            AgentRole.ANALYST.value,
            "(Analyst did not produce an answer.)",
        )
        rendered_context = self.format_chunks(context.retrieved_chunks)
        return _USER_TEMPLATE.format(
            query=context.query.strip(),
            context=rendered_context,
            analyst_output=analyst_output.strip(),
        )
