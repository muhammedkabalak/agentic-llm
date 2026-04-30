"""
Guardrails package (Step 5).

Independent monitoring layer that audits every agent output against a
fixed set of risk categories - hallucination, bias, PII, toxicity,
radicalization, prompt-injection echoes, and refusal-overreach - and
returns a structured `GuardrailMonitorReport`.

The monitor is intentionally decoupled from the Critic agent: the
Critic is an LLM-driven reviewer, while the monitor is deterministic
heuristics that work even when no LLM is available. The orchestrator
combines both signals into a single `GuardrailReport`.
"""

from app.guardrails.checks import (
    detect_bias,
    detect_hallucination,
    detect_pii,
    detect_prompt_injection,
    detect_radicalization,
    detect_refusal_overreach,
    detect_toxicity,
    redact_pii,
)
from app.guardrails.monitor import GuardrailMonitor, GuardrailMonitorReport

__all__ = [
    "GuardrailMonitor",
    "GuardrailMonitorReport",
    "detect_bias",
    "detect_hallucination",
    "detect_pii",
    "detect_prompt_injection",
    "detect_radicalization",
    "detect_refusal_overreach",
    "detect_toxicity",
    "redact_pii",
]
