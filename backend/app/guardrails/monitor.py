"""
GuardrailMonitor - aggregates individual detectors into a single
report per agent output.

Why a separate class instead of a free function? Two reasons:

1. The monitor is *injectable*. Tests (or future deployments) can pass
   a custom detector list - e.g. add a Presidio-based PII detector or
   a hosted toxicity model - without touching the orchestrator.
2. It owns the severity-to-verdict mapping in one place so every
   agent output is judged identically.

The monitor never raises. A buggy detector logs an error and the
remaining detectors still run. That keeps the inference path resilient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from app.guardrails.checks import all_detectors, redact_pii
from app.models.domain import DocumentChunk
from app.models.schemas import (
    AgentRole,
    GuardrailCategory,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailVerdict,
)
from app.services.logging_service import get_logger

logger = get_logger(__name__)


Detector = Callable[..., List[GuardrailFinding]]


@dataclass
class GuardrailMonitorReport:
    """
    Output of a single monitor inspection.

    The monitor never mutates the agent's text directly; if PII was
    found, ``redacted_text`` holds a sanitized copy that the
    orchestrator may choose to surface to the user.
    """

    agent_role: AgentRole
    verdict: GuardrailVerdict
    findings: List[GuardrailFinding] = field(default_factory=list)
    redacted_text: Optional[str] = None

    @property
    def flags(self) -> List[str]:
        """De-duplicated category names, useful for trace/UI badges."""
        seen: List[str] = []
        for f in self.findings:
            cat = f.category.value
            if cat not in seen:
                seen.append(cat)
        return seen

    @property
    def has_pii(self) -> bool:
        return any(
            f.category == GuardrailCategory.PII for f in self.findings
        )

    def summary(self) -> str:
        """Human-readable one-liner suitable for a guardrail note."""
        if not self.findings:
            return "Guardrail monitor: clean."
        parts = [f"{f.category.value}({f.severity.value})" for f in self.findings]
        return "Guardrail monitor flagged: " + ", ".join(parts)


# --------------------------------------------------------------------- #
# Severity -> verdict
# --------------------------------------------------------------------- #
_SEVERITY_TO_VERDICT = {
    GuardrailSeverity.INFO: GuardrailVerdict.PASS,
    GuardrailSeverity.WARN: GuardrailVerdict.WARN,
    GuardrailSeverity.BLOCK: GuardrailVerdict.BLOCK,
}

_VERDICT_RANK = {
    GuardrailVerdict.PASS: 0,
    GuardrailVerdict.WARN: 1,
    GuardrailVerdict.BLOCK: 2,
}


def merge_verdicts(*verdicts: GuardrailVerdict) -> GuardrailVerdict:
    """Return the worst (highest-ranked) verdict among the inputs."""
    worst = GuardrailVerdict.PASS
    for v in verdicts:
        if _VERDICT_RANK[v] > _VERDICT_RANK[worst]:
            worst = v
    return worst


# --------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------- #
class GuardrailMonitor:
    """
    Runs a fixed set of detectors over an agent's output and produces
    a `GuardrailMonitorReport`.
    """

    def __init__(
        self,
        detectors: Optional[Sequence[Detector]] = None,
        *,
        redact: bool = True,
    ) -> None:
        self.detectors: tuple[Detector, ...] = tuple(
            detectors if detectors is not None else all_detectors()
        )
        self.redact = redact

    def inspect(
        self,
        text: str,
        *,
        agent_role: AgentRole,
        context: Optional[Sequence[DocumentChunk]] = None,
    ) -> GuardrailMonitorReport:
        findings: List[GuardrailFinding] = []
        for det in self.detectors:
            try:
                result = det(text, context=context)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "guardrail_detector_failed",
                    detector=getattr(det, "__name__", repr(det)),
                    error=str(exc),
                )
                continue
            findings.extend(result)

        verdict = self._verdict_from_findings(findings)
        redacted_text: Optional[str] = None
        if self.redact and any(
            f.category == GuardrailCategory.PII for f in findings
        ):
            redacted_text = redact_pii(text)

        report = GuardrailMonitorReport(
            agent_role=agent_role,
            verdict=verdict,
            findings=findings,
            redacted_text=redacted_text,
        )

        logger.info(
            "guardrail_monitor_inspect",
            agent_role=agent_role.value,
            verdict=verdict.value,
            n_findings=len(findings),
            categories=report.flags,
        )
        return report

    @staticmethod
    def _verdict_from_findings(
        findings: List[GuardrailFinding],
    ) -> GuardrailVerdict:
        if not findings:
            return GuardrailVerdict.PASS
        worst = GuardrailVerdict.PASS
        for f in findings:
            v = _SEVERITY_TO_VERDICT[f.severity]
            if _VERDICT_RANK[v] > _VERDICT_RANK[worst]:
                worst = v
        return worst


__all__ = ["GuardrailMonitor", "GuardrailMonitorReport", "merge_verdicts"]
