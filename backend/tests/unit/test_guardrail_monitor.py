"""
Unit tests for GuardrailMonitor.

These exercise the aggregation/decision logic that sits on top of the
individual detectors:
  * Worst severity wins -> verdict
  * PII triggers redaction
  * Custom detector list is honoured
  * Buggy detectors don't crash the monitor
  * `merge_verdicts` returns the worst input
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from app.guardrails.monitor import GuardrailMonitor, merge_verdicts
from app.models.domain import DocumentChunk
from app.models.schemas import (
    AgentRole,
    GuardrailCategory,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailVerdict,
)


def _detector_with(*findings: GuardrailFinding):
    def fn(text: str, *, context: Optional[Sequence[DocumentChunk]] = None) -> List[GuardrailFinding]:
        return list(findings)
    return fn


def test_monitor_pass_when_no_findings() -> None:
    monitor = GuardrailMonitor(detectors=[_detector_with()])
    report = monitor.inspect("clean text", agent_role=AgentRole.RESEARCHER)
    assert report.verdict == GuardrailVerdict.PASS
    assert report.findings == []
    assert report.redacted_text is None
    assert report.flags == []


def test_monitor_warn_takes_worst_severity() -> None:
    f_warn = GuardrailFinding(
        category=GuardrailCategory.BIAS,
        severity=GuardrailSeverity.WARN,
        message="x",
        detector="t",
    )
    f_info = GuardrailFinding(
        category=GuardrailCategory.HALLUCINATION,
        severity=GuardrailSeverity.INFO,
        message="y",
        detector="t",
    )
    monitor = GuardrailMonitor(detectors=[_detector_with(f_info, f_warn)])
    report = monitor.inspect("text", agent_role=AgentRole.ANALYST)
    assert report.verdict == GuardrailVerdict.WARN
    assert len(report.findings) == 2


def test_monitor_block_overrides_warn() -> None:
    f_block = GuardrailFinding(
        category=GuardrailCategory.RADICALIZATION,
        severity=GuardrailSeverity.BLOCK,
        message="x",
        detector="t",
    )
    f_warn = GuardrailFinding(
        category=GuardrailCategory.BIAS,
        severity=GuardrailSeverity.WARN,
        message="y",
        detector="t",
    )
    monitor = GuardrailMonitor(detectors=[_detector_with(f_warn, f_block)])
    report = monitor.inspect("text", agent_role=AgentRole.CRITIC)
    assert report.verdict == GuardrailVerdict.BLOCK


def test_monitor_pii_triggers_redaction_with_default_detectors() -> None:
    # Use the default detectors so the PII pipeline is wired end-to-end.
    monitor = GuardrailMonitor()
    text = "Reach out at jane@example.com for the report."
    report = monitor.inspect(text, agent_role=AgentRole.RESEARCHER)
    assert report.has_pii
    assert report.verdict == GuardrailVerdict.BLOCK
    assert report.redacted_text is not None
    assert "[REDACTED:EMAIL]" in report.redacted_text
    assert "jane@example.com" not in report.redacted_text


def test_monitor_redact_disabled_keeps_text_none() -> None:
    monitor = GuardrailMonitor(redact=False)
    report = monitor.inspect(
        "Reach out at jane@example.com.", agent_role=AgentRole.RESEARCHER
    )
    assert report.has_pii
    assert report.redacted_text is None


def test_monitor_buggy_detector_is_skipped() -> None:
    def boom(text, *, context=None):
        raise RuntimeError("detector exploded")

    f_ok = GuardrailFinding(
        category=GuardrailCategory.BIAS,
        severity=GuardrailSeverity.WARN,
        message="x",
        detector="t",
    )
    monitor = GuardrailMonitor(detectors=[boom, _detector_with(f_ok)])
    report = monitor.inspect("text", agent_role=AgentRole.RESEARCHER)
    # The good detector still ran, the bad one was swallowed.
    assert report.verdict == GuardrailVerdict.WARN
    assert len(report.findings) == 1


def test_monitor_passes_context_to_detectors() -> None:
    seen_context: dict = {}

    def capture(text, *, context=None):
        seen_context["ctx"] = context
        return []

    monitor = GuardrailMonitor(detectors=[capture])
    chunks = [DocumentChunk(content="hello world", source="s")]
    monitor.inspect("text", agent_role=AgentRole.RESEARCHER, context=chunks)
    assert seen_context["ctx"] is chunks


def test_monitor_flags_are_unique_and_ordered() -> None:
    fa = GuardrailFinding(
        category=GuardrailCategory.PII,
        severity=GuardrailSeverity.BLOCK,
        message="a",
        detector="t",
    )
    fb = GuardrailFinding(
        category=GuardrailCategory.PII,
        severity=GuardrailSeverity.WARN,
        message="b",
        detector="t",
    )
    fc = GuardrailFinding(
        category=GuardrailCategory.HALLUCINATION,
        severity=GuardrailSeverity.WARN,
        message="c",
        detector="t",
    )
    monitor = GuardrailMonitor(detectors=[_detector_with(fa, fb, fc)])
    report = monitor.inspect("text", agent_role=AgentRole.ANALYST)
    assert report.flags == ["pii", "hallucination"]


def test_merge_verdicts_returns_worst() -> None:
    assert merge_verdicts(GuardrailVerdict.PASS, GuardrailVerdict.PASS) == GuardrailVerdict.PASS
    assert merge_verdicts(GuardrailVerdict.PASS, GuardrailVerdict.WARN) == GuardrailVerdict.WARN
    assert merge_verdicts(GuardrailVerdict.WARN, GuardrailVerdict.BLOCK) == GuardrailVerdict.BLOCK
    assert merge_verdicts(GuardrailVerdict.PASS) == GuardrailVerdict.PASS
    assert merge_verdicts() == GuardrailVerdict.PASS
