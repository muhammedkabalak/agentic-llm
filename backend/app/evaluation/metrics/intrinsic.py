"""
Intrinsic metrics.

Pure functions, returning floats in [0, 1] (or non-negative real
numbers for perplexity-like scores). They never touch the LLM, never
hit the network, and never need a labelled dataset beyond the
reference text the caller already has.

Implemented from scratch on purpose:
  * It removes a heavy NLTK / sacrebleu dependency for what is, in
    practice, a small classroom-scale eval.
  * It makes the scoring fully transparent and unit-testable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# --------------------------------------------------------------------- #
# BLEU-like score (1- and 2-gram precision with brevity penalty)
# --------------------------------------------------------------------- #
def _modified_precision(
    candidate: Sequence[str], reference: Sequence[str], n: int
) -> float:
    cand_ng = _ngrams(candidate, n)
    if not cand_ng:
        return 0.0
    ref_counts = Counter(_ngrams(reference, n))
    cand_counts = Counter(cand_ng)
    overlap = 0
    for ng, cnt in cand_counts.items():
        overlap += min(cnt, ref_counts.get(ng, 0))
    return overlap / len(cand_ng)


def bleu_like_score(candidate: str, reference: str) -> float:
    """
    Geometric mean of 1- and 2-gram modified precision with a brevity
    penalty. Returns a value in [0, 1]. Empty inputs return 0.0.
    """
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0

    p1 = _modified_precision(cand, ref, 1)
    p2 = _modified_precision(cand, ref, 2)
    if p1 == 0.0 or p2 == 0.0:
        return 0.0

    bp = 1.0 if len(cand) >= len(ref) else math.exp(1.0 - len(ref) / len(cand))
    return bp * math.sqrt(p1 * p2)


# --------------------------------------------------------------------- #
# ROUGE-L (LCS-based F1)
# --------------------------------------------------------------------- #
def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    # Space-optimized O(min(len(a), len(b))) DP.
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ai in a:
        curr = [0] * (len(b) + 1)
        for j, bj in enumerate(b, start=1):
            if ai == bj:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_score(candidate: str, reference: str) -> float:
    """ROUGE-L F1 in [0, 1]."""
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    p = lcs / len(cand)
    r = lcs / len(ref)
    return (2 * p * r) / (p + r)


# --------------------------------------------------------------------- #
# Token overlap (Jaccard-style, vocab-only)
# --------------------------------------------------------------------- #
def token_overlap(candidate: str, reference: str) -> float:
    """Jaccard similarity on the token vocabularies (in [0, 1])."""
    a = set(_tokens(candidate))
    b = set(_tokens(reference))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------- #
# Keyword coverage
# --------------------------------------------------------------------- #
def keyword_coverage(text: str, keywords: Iterable[str]) -> float:
    """Fraction of `keywords` that appear (case-insensitive) in `text`.

    Returns 1.0 when no keywords are supplied (vacuously true).
    """
    kws = [k for k in (keywords or []) if k]
    if not kws:
        return 1.0
    hay = (text or "").lower()
    hits = sum(1 for k in kws if k.lower() in hay)
    return hits / len(kws)


# --------------------------------------------------------------------- #
# Perplexity proxy (no LLM)
# --------------------------------------------------------------------- #
def _char_bigram_distribution(text: str) -> Counter:
    """Character bigrams from a normalised lowercase string."""
    s = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(s) < 2:
        return Counter()
    return Counter(s[i : i + 2] for i in range(len(s) - 1))


def perplexity_proxy(candidate: str, reference: str) -> float:
    """
    A lightweight, deterministic stand-in for true LLM perplexity.

    Real perplexity needs token-level log-likelihoods from a language
    model; we don't have one in tests. Instead we measure how
    "surprised" a *char-bigram* model trained on `reference` would be
    when scoring `candidate`. Lower is better, like real perplexity.
    Returns a positive float; identical strings score near 1.0.

    Formula: perplexity = exp( (1/N) * sum(-log P(bigram)) )
    where P is a Laplace-smoothed distribution over reference bigrams.
    """
    cand_bigrams = list(_char_bigram_distribution(candidate).elements())
    ref_dist = _char_bigram_distribution(reference)
    if not cand_bigrams or not ref_dist:
        return float("inf")
    vocab_size = max(len(ref_dist), 1)
    total = sum(ref_dist.values()) + vocab_size  # +1 smoothing per type
    log_sum = 0.0
    for bg in cand_bigrams:
        p = (ref_dist.get(bg, 0) + 1) / total
        log_sum += -math.log(p)
    return math.exp(log_sum / len(cand_bigrams))


__all__ = [
    "bleu_like_score",
    "rouge_l_score",
    "token_overlap",
    "keyword_coverage",
    "perplexity_proxy",
]
