"""
Unit tests for the Step 6 evaluation metrics.

These lock in the math: every metric should be deterministic, return
floats in the expected range, and behave correctly on edge cases
(empty inputs, perfect matches, total mismatches).
"""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics.extrinsic import (
    faithfulness_score,
    guardrail_pass_score,
    retrieval_at_k,
    task_completion_score,
)
from app.evaluation.metrics.intrinsic import (
    bleu_like_score,
    keyword_coverage,
    perplexity_proxy,
    rouge_l_score,
    token_overlap,
)
from app.models.schemas import (
    AgentRole,
    AgentTrace,
    ChatResponse,
    GuardrailReport,
    GuardrailVerdict,
    RetrievedChunk,
)


# --------------------------------------------------------------------- #
# Intrinsic metrics
# --------------------------------------------------------------------- #
def test_bleu_like_perfect_match() -> None:
    text = "Retrieval augmented generation combines retrieval with generation."
    assert bleu_like_score(text, text) == pytest.approx(1.0, abs=1e-9)


def test_bleu_like_total_mismatch_is_zero() -> None:
    assert bleu_like_score("apple banana cherry", "xyz qrs uvw") == 0.0


def test_bleu_like_partial_match_in_unit_interval() -> None:
    s = bleu_like_score(
        "RAG combines retrieval and generation",
        "RAG combines retrieval with generation",
    )
    assert 0.0 < s < 1.0


def test_bleu_like_empty_inputs_zero() -> None:
    assert bleu_like_score("", "anything") == 0.0
    assert bleu_like_score("anything", "") == 0.0


def test_bleu_like_brevity_penalty_shorter_candidate() -> None:
    long_ref = "the quick brown fox jumps over the lazy dog"
    short_cand = "the quick brown fox"
    short = bleu_like_score(short_cand, long_ref)
    full = bleu_like_score(long_ref, long_ref)
    assert short < full


def test_rouge_l_perfect_match() -> None:
    text = "vector databases store embeddings for similarity search"
    assert rouge_l_score(text, text) == pytest.approx(1.0)


def test_rouge_l_zero_on_disjoint() -> None:
    assert rouge_l_score("apple banana", "xyz qrs") == 0.0


def test_rouge_l_partial_match() -> None:
    s = rouge_l_score(
        "vector dbs store embeddings",
        "vector databases store embeddings for search",
    )
    assert 0.0 < s < 1.0


def test_rouge_l_empty_inputs_zero() -> None:
    assert rouge_l_score("", "x") == 0.0
    assert rouge_l_score("x", "") == 0.0


def test_token_overlap_full_and_disjoint() -> None:
    assert token_overlap("a b c", "a b c") == 1.0
    assert token_overlap("a b c", "x y z") == 0.0
    # Both empty -> vacuously identical
    assert token_overlap("", "") == 1.0
    assert token_overlap("a", "") == 0.0


def test_token_overlap_jaccard() -> None:
    # vocab a={a,b,c}, b={b,c,d} -> |intersection|=2, |union|=4 -> 0.5
    assert token_overlap("a b c", "b c d") == 0.5


def test_keyword_coverage() -> None:
    assert keyword_coverage("RAG combines retrieval and generation", ["retrieval", "generation"]) == 1.0
    assert keyword_coverage("RAG combines retrieval", ["retrieval", "generation"]) == 0.5
    # Vacuous (no keywords) -> 1.0
    assert keyword_coverage("anything", []) == 1.0
    # Empty text
    assert keyword_coverage("", ["retrieval"]) == 0.0


def test_perplexity_proxy_lower_for_similar_text() -> None:
    ref = "vector databases store embeddings for similarity search"
    near = "vector databases store embeddings"
    far = "completely unrelated nonsense words about astrophysics"
    p_near = perplexity_proxy(near, ref)
    p_far = perplexity_proxy(far, ref)
    assert math.isfinite(p_near)
    assert math.isfinite(p_far)
    assert p_near < p_far


def test_perplexity_proxy_empty_inputs_inf() -> None:
    assert perplexity_proxy("", "x y z") == float("inf")
    assert perplexity_proxy("x y z", "") == float("inf")


# --------------------------------------------------------------------- #
# Extrinsic metrics
# --------------------------------------------------------------------- #
def test_task_completion_keywords_all_partial_none() -> None:
    answer = "RAG combines retrieval with generation."
    assert task_completion_score(
        answer, expected_keywords=["retrieval", "generation"]
    ) == 1.0
    assert task_completion_score(
        answer, expected_keywords=["retrieval", "embeddings"]
    ) == 0.5
    assert task_completion_score(
        answer, expected_keywords=["xyz", "qrs"]
    ) == 0.0


def test_task_completion_substring_takes_precedence() -> None:
    answer = "Use RAG for grounded answers."
    assert task_completion_score(answer, expected_substring="RAG") == 1.0
    assert task_completion_score(answer, expected_substring="missing") == 0.0


def test_task_completion_no_labels_is_one() -> None:
    assert task_completion_score("anything") == 1.0


def test_faithfulness_high_when_answer_in_context() -> None:
    contexts = [
        "Vector databases store embeddings for similarity search.",
        "RAG combines retrieval with generation.",
    ]
    answer = (
        "Vector databases store embeddings for similarity search. "
        "RAG combines retrieval with generation."
    )
    assert faithfulness_score(answer, contexts) >= 0.9


def test_faithfulness_low_when_off_topic() -> None:
    contexts = ["RAG combines retrieval with generation."]
    answer = (
        "The capital of France is Paris and the Eiffel tower is famous worldwide."
    )
    assert faithfulness_score(answer, contexts) < 0.5


def test_faithfulness_no_context_zero() -> None:
    assert faithfulness_score("anything", []) == 0.0


def test_retrieval_at_k_full_partial_zero_vacuous() -> None:
    chunks = [
        RetrievedChunk(content="a", source="doc-a.md"),
        RetrievedChunk(content="b", source="doc-b.md"),
        RetrievedChunk(content="c", source="doc-c.md"),
    ]
    assert retrieval_at_k(["doc-a.md", "doc-b.md"], chunks) == 1.0
    assert retrieval_at_k(["doc-a.md", "doc-z.md"], chunks) == 0.5
    assert retrieval_at_k(["doc-z.md"], chunks) == 0.0
    assert retrieval_at_k([], chunks) == 1.0


def test_retrieval_at_k_respects_k() -> None:
    chunks = [
        RetrievedChunk(content="a", source="doc-a.md"),
        RetrievedChunk(content="b", source="doc-b.md"),
    ]
    assert retrieval_at_k(["doc-a.md"], chunks, k=1) == 1.0
    assert retrieval_at_k(["doc-b.md"], chunks, k=1) == 0.0


def _trace(verdict: GuardrailVerdict) -> AgentTrace:
    return AgentTrace(
        agent_role=AgentRole.RESEARCHER,
        input="q",
        output="o",
        guardrail_report=GuardrailReport(verdict=verdict),
    )


def test_guardrail_pass_score_all_pass() -> None:
    response = ChatResponse(
        answer="x",
        traces=[_trace(GuardrailVerdict.PASS), _trace(GuardrailVerdict.PASS)],
    )
    assert guardrail_pass_score(response) == 1.0


def test_guardrail_pass_score_one_warn_fails() -> None:
    response = ChatResponse(
        answer="x",
        traces=[_trace(GuardrailVerdict.PASS), _trace(GuardrailVerdict.WARN)],
    )
    assert guardrail_pass_score(response) == 0.0


def test_guardrail_pass_score_block_fails() -> None:
    response = ChatResponse(
        answer="x", traces=[_trace(GuardrailVerdict.BLOCK)]
    )
    assert guardrail_pass_score(response) == 0.0


def test_guardrail_pass_score_no_reports_passes() -> None:
    response = ChatResponse(
        answer="x",
        traces=[
            AgentTrace(agent_role=AgentRole.RESEARCHER, input="q", output="o")
        ],
    )
    assert guardrail_pass_score(response) == 1.0
