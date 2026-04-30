"""
Extrinsic metrics: how well did the system complete the user's task?

These score a generated `ChatResponse` against the labelled `EvalCase`
or against the actual retrieved context. They are deliberately
robust to noisy LLM output (so an answer that happens to contain
`-` or extra whitespace doesn't tank a perfectly correct response).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from app.evaluation.metrics.intrinsic import _tokens
from app.models.schemas import ChatResponse, GuardrailVerdict, RetrievedChunk


# --------------------------------------------------------------------- #
# Task completion
# --------------------------------------------------------------------- #
def task_completion_score(
    answer: str,
    *,
    expected_keywords: Optional[Iterable[str]] = None,
    expected_substring: Optional[str] = None,
) -> float:
    """
    1.0 if the answer completes the task as defined by the labels:
      * if `expected_substring` is given, it must appear (case-insensitive),
      * else if `expected_keywords` are given, ALL must appear,
      * else returns 1.0 (no labels - vacuously satisfied).

    Returns a partial credit float in [0, 1] when keywords are partially
    matched.
    """
    hay = (answer or "").lower()

    if expected_substring:
        return 1.0 if expected_substring.lower() in hay else 0.0

    keywords = [k for k in (expected_keywords or []) if k]
    if not keywords:
        return 1.0

    hits = sum(1 for k in keywords if k.lower() in hay)
    return hits / len(keywords)


# --------------------------------------------------------------------- #
# Faithfulness vs retrieved context
# --------------------------------------------------------------------- #
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]")


def _sentences(text: str) -> List[str]:
    out: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*0123456789. )")
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        matched = _SENTENCE_RE.findall(line)
        if matched:
            out.extend(s.strip() for s in matched)
        else:
            out.append(line)
    return [s for s in out if len(_tokens(s)) >= 4]


def faithfulness_score(
    answer: str,
    contexts: Sequence[str],
    *,
    overlap_threshold: float = 0.18,
) -> float:
    """
    Fraction of answer sentences whose tokens overlap the context
    above ``overlap_threshold``. Higher = more grounded.

    This is the positive-sense complement of the Step-5 hallucination
    detector: we count how many sentences ARE supported, instead of
    flagging the unsupported ones.

    Returns 1.0 when there are no scoreable sentences (e.g. the answer
    is a one-liner) - a vacuously-true score is the safest choice for
    aggregation.
    """
    if not contexts:
        return 0.0
    ref_tokens = set()
    for c in contexts:
        ref_tokens.update(_tokens(c))
    if not ref_tokens:
        return 0.0

    sents = _sentences(answer)
    if not sents:
        return 1.0

    grounded = 0
    for s in sents:
        toks = set(_tokens(s))
        if not toks:
            continue
        overlap = len(toks & ref_tokens) / len(toks)
        if overlap >= overlap_threshold:
            grounded += 1
    return grounded / len(sents)


# --------------------------------------------------------------------- #
# Retrieval @ k
# --------------------------------------------------------------------- #
def retrieval_at_k(
    expected_sources: Iterable[str],
    retrieved: Sequence[RetrievedChunk],
    *,
    k: Optional[int] = None,
) -> float:
    """
    Fraction of expected sources that appear among the top-``k``
    retrieved chunk sources. ``k=None`` uses all retrieved chunks.

    Returns 1.0 when no expected sources are supplied (skipped case).
    """
    expected = [s for s in (expected_sources or []) if s]
    if not expected:
        return 1.0
    top = list(retrieved or [])[: k] if k is not None else list(retrieved or [])
    sources_seen = {c.source for c in top if c.source}
    hits = sum(1 for s in expected if s in sources_seen)
    return hits / len(expected)


# --------------------------------------------------------------------- #
# Guardrail pass score (binary 0 / 1)
# --------------------------------------------------------------------- #
def guardrail_pass_score(response: ChatResponse) -> float:
    """1.0 if every trace's guardrail verdict is PASS, 0.0 otherwise.

    A trace without a guardrail report counts as PASS - older / simpler
    pipelines may not attach one.
    """
    for trace in response.traces:
        report = trace.guardrail_report
        if report is None:
            continue
        if report.verdict != GuardrailVerdict.PASS:
            return 0.0
    return 1.0


__all__ = [
    "task_completion_score",
    "faithfulness_score",
    "retrieval_at_k",
    "guardrail_pass_score",
]
