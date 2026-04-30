"""
Individual guardrail detectors.

Every detector is a pure function:

    detector(text, *, context=None) -> List[GuardrailFinding]

* Pure: no I/O, no LLM calls, no global state. Cheap to call on every
  agent output, deterministic, and fully testable offline.
* Heuristic-based: regex + keyword matching. The point is to catch the
  obvious failures the LLM-based Critic might miss (or be jail-broken
  past), not to replace human review.
* Composable: the GuardrailMonitor just wires these together and
  aggregates the findings.

Categories implemented (from the project's 'Agent Monitoring & Risk
Management' requirement):
  * PII             - email / phone / IBAN / credit card / TC kimlik
  * Hallucination   - claims unsupported by retrieved context (when
                      context is supplied)
  * Toxicity        - slurs, harassment, profanity targeting people
  * Bias            - over-generalisations like "all X are Y"
  * Radicalization  - calls to violence, extremist talking points
  * Prompt injection - common jailbreak / instruction-override echoes
  * Refusal overreach - "I cannot help" patterns where context is rich
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from app.models.domain import DocumentChunk
from app.models.schemas import (
    GuardrailCategory,
    GuardrailFinding,
    GuardrailSeverity,
)


# --------------------------------------------------------------------- #
# PII
# --------------------------------------------------------------------- #
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
# International-ish phone: +<country><digits> or 10+ contiguous digits
_PHONE_RE = re.compile(r"(?:\+\d[\d\s().\-]{7,}\d)|(?:\b\d{10,}\b)")
# Credit cards: 13-19 digit groups (Luhn check applied separately)
_CC_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# IBAN (rough): 2 letters + 2 digits + up to 30 alphanumerics
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
# TC Kimlik: 11 digits, first not 0, simple check digit rule below
_TCK_RE = re.compile(r"\b[1-9]\d{10}\b")
# US SSN
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _luhn_ok(digits: str) -> bool:
    s = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return s % 10 == 0


def _tckn_ok(digits: str) -> bool:
    if len(digits) != 11 or digits[0] == "0":
        return False
    d = [int(c) for c in digits]
    c10 = ((sum(d[0:9:2]) * 7) - sum(d[1:8:2])) % 10
    c11 = (sum(d[0:10])) % 10
    return c10 == d[9] and c11 == d[10]


def detect_pii(
    text: str, *, context: Optional[Sequence[DocumentChunk]] = None
) -> List[GuardrailFinding]:
    """Detect direct PII leaks in ``text``."""
    findings: List[GuardrailFinding] = []

    for m in _EMAIL_RE.finditer(text):
        findings.append(
            GuardrailFinding(
                category=GuardrailCategory.PII,
                severity=GuardrailSeverity.BLOCK,
                message="Email address detected.",
                evidence=m.group(0),
                detector="pii.email",
            )
        )

    for m in _SSN_RE.finditer(text):
        findings.append(
            GuardrailFinding(
                category=GuardrailCategory.PII,
                severity=GuardrailSeverity.BLOCK,
                message="US SSN-shaped number detected.",
                evidence=m.group(0),
                detector="pii.ssn",
            )
        )

    for m in _IBAN_RE.finditer(text):
        findings.append(
            GuardrailFinding(
                category=GuardrailCategory.PII,
                severity=GuardrailSeverity.BLOCK,
                message="IBAN-shaped string detected.",
                evidence=m.group(0),
                detector="pii.iban",
            )
        )

    for m in _CC_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.PII,
                    severity=GuardrailSeverity.BLOCK,
                    message="Credit-card-shaped number passing Luhn detected.",
                    evidence=m.group(0),
                    detector="pii.credit_card",
                )
            )

    for m in _TCK_RE.finditer(text):
        if _tckn_ok(m.group(0)):
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.PII,
                    severity=GuardrailSeverity.BLOCK,
                    message="Turkish national ID (TCKN) detected.",
                    evidence=m.group(0),
                    detector="pii.tckn",
                )
            )

    for m in _PHONE_RE.finditer(text):
        # Avoid double-flagging the credit-card / IBAN / TCKN matches.
        if any(f.evidence and m.group(0) in f.evidence for f in findings):
            continue
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 15:
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.PII,
                    severity=GuardrailSeverity.WARN,
                    message="Phone-number-shaped sequence detected.",
                    evidence=m.group(0),
                    detector="pii.phone",
                )
            )

    return findings


def redact_pii(text: str) -> str:
    """Return ``text`` with detected PII replaced by ``[REDACTED:<kind>]``.

    The order matters: most-specific patterns first so a credit card is
    not partially redacted as a phone number.
    """
    redacted = text

    redacted = _EMAIL_RE.sub("[REDACTED:EMAIL]", redacted)
    redacted = _SSN_RE.sub("[REDACTED:SSN]", redacted)
    redacted = _IBAN_RE.sub("[REDACTED:IBAN]", redacted)

    def _cc_sub(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return "[REDACTED:CC]"
        return m.group(0)

    redacted = _CC_RE.sub(_cc_sub, redacted)

    def _tckn_sub(m: re.Match[str]) -> str:
        return "[REDACTED:TCKN]" if _tckn_ok(m.group(0)) else m.group(0)

    redacted = _TCK_RE.sub(_tckn_sub, redacted)

    def _phone_sub(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 15:
            return "[REDACTED:PHONE]"
        return m.group(0)

    redacted = _PHONE_RE.sub(_phone_sub, redacted)
    return redacted


# --------------------------------------------------------------------- #
# Toxicity / bias
# --------------------------------------------------------------------- #
_TOXIC_KEYWORDS = (
    "idiot",
    "moron",
    "stupid bastard",
    "shut up",
    "kill yourself",
    "kys",
    "go die",
    "worthless piece",
)
_BIAS_PATTERNS = (
    re.compile(r"\ball\s+(?:women|men|jews|muslims|christians|asians|africans|americans|blacks|whites|gays|lesbians)\s+are\b", re.IGNORECASE),
    re.compile(r"\b(?:women|men|jews|muslims|christians|asians|africans|americans|blacks|whites|gays|lesbians)\s+are\s+inherently\b", re.IGNORECASE),
)


def detect_toxicity(
    text: str, *, context: Optional[Sequence[DocumentChunk]] = None
) -> List[GuardrailFinding]:
    findings: List[GuardrailFinding] = []
    lower = text.lower()
    for kw in _TOXIC_KEYWORDS:
        if kw in lower:
            severity = (
                GuardrailSeverity.BLOCK
                if kw in {"kill yourself", "kys", "go die"}
                else GuardrailSeverity.WARN
            )
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.TOXICITY,
                    severity=severity,
                    message=f"Toxic phrase detected: '{kw}'.",
                    evidence=kw,
                    detector="toxicity.keyword",
                )
            )
    return findings


def detect_bias(
    text: str, *, context: Optional[Sequence[DocumentChunk]] = None
) -> List[GuardrailFinding]:
    findings: List[GuardrailFinding] = []
    for pat in _BIAS_PATTERNS:
        m = pat.search(text)
        if m:
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.BIAS,
                    severity=GuardrailSeverity.WARN,
                    message="Sweeping generalisation about a demographic group.",
                    evidence=m.group(0),
                    detector="bias.generalization",
                )
            )
    return findings


# --------------------------------------------------------------------- #
# Radicalization
# --------------------------------------------------------------------- #
_RADICAL_PATTERNS = (
    re.compile(r"\bjoin\s+(?:isis|al[- ]?qaeda|the\s+caliphate)\b", re.IGNORECASE),
    re.compile(r"\bglory\s+to\s+(?:hitler|the\s+f[üu]hrer)\b", re.IGNORECASE),
    re.compile(r"\b(?:white|aryan)\s+supremacy\s+is\s+(?:right|the\s+answer)\b", re.IGNORECASE),
)
_VIOLENCE_INCITEMENT = (
    re.compile(r"\byou\s+should\s+(?:kill|attack|bomb|shoot)\s+(?:them|him|her|the\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\blet'?s\s+(?:kill|attack|bomb|murder)\s+\w+", re.IGNORECASE),
    re.compile(r"\bdeath\s+to\s+(?:all\s+)?(?:jews|muslims|christians|americans|kurds|turks|gays)\b", re.IGNORECASE),
)


def detect_radicalization(
    text: str, *, context: Optional[Sequence[DocumentChunk]] = None
) -> List[GuardrailFinding]:
    findings: List[GuardrailFinding] = []
    for pat in _RADICAL_PATTERNS:
        m = pat.search(text)
        if m:
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.RADICALIZATION,
                    severity=GuardrailSeverity.BLOCK,
                    message="Extremist endorsement detected.",
                    evidence=m.group(0),
                    detector="radical.endorsement",
                )
            )
    for pat in _VIOLENCE_INCITEMENT:
        m = pat.search(text)
        if m:
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.RADICALIZATION,
                    severity=GuardrailSeverity.BLOCK,
                    message="Incitement to violence detected.",
                    evidence=m.group(0),
                    detector="radical.incitement",
                )
            )
    return findings


# --------------------------------------------------------------------- #
# Prompt injection
# --------------------------------------------------------------------- #
_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:the\s+)?system\s+prompt\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?:dan|jailbroken|unrestricted)\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(?:your|the)\s+(?:system|hidden)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:if\s+you\s+have\s+no|an?\s+unrestricted)\b", re.IGNORECASE),
)


def detect_prompt_injection(
    text: str, *, context: Optional[Sequence[DocumentChunk]] = None
) -> List[GuardrailFinding]:
    findings: List[GuardrailFinding] = []
    for pat in _PROMPT_INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.PROMPT_INJECTION,
                    severity=GuardrailSeverity.BLOCK,
                    message="Prompt-injection / jailbreak phrase detected in output.",
                    evidence=m.group(0),
                    detector="prompt_injection.pattern",
                )
            )
    return findings


# --------------------------------------------------------------------- #
# Hallucination
# --------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]")
_STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "they", "them",
    "their", "there", "which", "where", "when", "what", "would", "could",
    "should", "about", "into", "than", "then", "also", "more", "most",
    "some", "such", "only", "very", "much", "many", "make", "made",
    "your", "yours", "ours", "ourselves", "itself", "because", "while",
    "after", "before", "between", "across", "every", "other", "those",
    "these", "however", "therefore", "thus", "still", "even", "just",
    "like", "really", "actually", "based", "above", "below",
}


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS
    }


def _sentences(text: str) -> List[str]:
    """Crude sentence splitter that keeps Markdown bullets as their own
    sentence so each claim can be checked individually."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789. )")
        if not line:
            continue
        if line.startswith("#") or line.startswith(">"):
            continue
        # Pull out punctuation-terminated sentences
        matched = _SENTENCE_RE.findall(line)
        if matched:
            out.extend(s.strip() for s in matched)
        else:
            out.append(line)
    return [s for s in out if len(s) > 12]


def detect_hallucination(
    text: str,
    *,
    context: Optional[Sequence[DocumentChunk]] = None,
    overlap_threshold: float = 0.18,
) -> List[GuardrailFinding]:
    """
    Lightweight grounding check: every sentence of the agent's output
    should share enough vocabulary with at least one retrieved chunk.

    This is deliberately permissive (low threshold). It is meant to
    catch outputs that are *clearly* unrelated to the retrieved context
    - the kind of failure where the LLM ignored the context block
    entirely - not to police every wording choice.
    """
    if not context:
        return []

    context_text = "\n".join(c.content for c in context)
    context_tokens = _tokens(context_text)
    if not context_tokens:
        return []

    findings: List[GuardrailFinding] = []
    for sent in _sentences(text):
        tokens = _tokens(sent)
        if len(tokens) < 6:
            # Too short to make a confident claim about; skip.
            continue
        overlap = len(tokens & context_tokens) / max(len(tokens), 1)
        if overlap < overlap_threshold:
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.HALLUCINATION,
                    severity=GuardrailSeverity.WARN,
                    message=(
                        "Sentence shares little vocabulary with the "
                        "retrieved context (possible hallucination)."
                    ),
                    evidence=sent[:200],
                    detector="hallucination.token_overlap",
                )
            )
    return findings


# --------------------------------------------------------------------- #
# Refusal overreach
# --------------------------------------------------------------------- #
_REFUSAL_PATTERNS = (
    re.compile(r"\bi\s+(?:cannot|can'?t|am\s+unable\s+to)\s+(?:help|answer|assist)\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:do\s+not|don'?t)\s+have\s+(?:enough\s+)?information\b", re.IGNORECASE),
    re.compile(r"\bsorry,?\s+i\s+can'?t\s+help\b", re.IGNORECASE),
)


def detect_refusal_overreach(
    text: str,
    *,
    context: Optional[Sequence[DocumentChunk]] = None,
    min_chunks: int = 2,
    min_context_chars: int = 400,
) -> List[GuardrailFinding]:
    """
    If the retrieved context is rich but the agent still refused, flag
    it. This is the inverse of the refusal-honesty check the Critic
    does and protects against over-cautious agents wasting good
    context.
    """
    if not context:
        return []

    n_chunks = len(context)
    total_chars = sum(len(c.content) for c in context)
    if n_chunks < min_chunks or total_chars < min_context_chars:
        return []

    findings: List[GuardrailFinding] = []
    for pat in _REFUSAL_PATTERNS:
        m = pat.search(text)
        if m:
            findings.append(
                GuardrailFinding(
                    category=GuardrailCategory.REFUSAL_OVERREACH,
                    severity=GuardrailSeverity.WARN,
                    message=(
                        "Agent refused even though the retrieved "
                        "context appears sufficient."
                    ),
                    evidence=m.group(0),
                    detector="refusal.overreach",
                )
            )
            break  # one finding is enough
    return findings


# --------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------- #
def all_detectors() -> tuple:
    """Default detector tuple used by the GuardrailMonitor."""
    return (
        detect_pii,
        detect_toxicity,
        detect_bias,
        detect_radicalization,
        detect_prompt_injection,
        detect_hallucination,
        detect_refusal_overreach,
    )


__all__ = [
    "detect_pii",
    "detect_toxicity",
    "detect_bias",
    "detect_radicalization",
    "detect_prompt_injection",
    "detect_hallucination",
    "detect_refusal_overreach",
    "redact_pii",
    "all_detectors",
]
