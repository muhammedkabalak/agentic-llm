"""
Evaluator: runs an orchestrator (Single or Multi) over an EvalDataset
and produces an EvalRunReport.

Design notes:
  * The pipeline is a duck-typed object exposing
        async run(query, *, history, session_id, top_k) -> ChatResponse
    so the same Evaluator works against `SingleAgentPipeline`,
    `MultiAgentOrchestrator`, or any future variant.
  * Errors in a single case are caught and recorded (`error` on the
    case result) so one bad case never aborts the whole run.
  * Aggregation skips metrics that were 'not applicable' to a case
    (empty expected_keywords / expected_sources / contexts) so an
    average is not biased by vacuously-1.0 scores.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from app.evaluation.dataset import EvalCase, EvalDataset
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
from app.evaluation.report import (
    AggregateMetrics,
    EvalCaseResult,
    EvalRunReport,
)
from app.models.schemas import ChatMode, ChatResponse, GuardrailVerdict
from app.services.logging_service import get_logger

logger = get_logger(__name__)


class _RunnablePipeline(Protocol):
    """Minimal async interface the Evaluator needs."""

    async def run(
        self,
        query: str,
        *,
        history: Any = None,
        session_id: Any = None,
        top_k: Any = None,
    ) -> ChatResponse: ...


# Metrics that are conditional on label presence; if the case lacks
# the relevant labels we skip the metric for that case rather than
# averaging over a vacuously-perfect score.
_EXTRINSIC_CONDITIONAL = {
    "task_completion",
    "retrieval_at_k",
    "faithfulness",
}
_INTRINSIC_CONDITIONAL = {
    "bleu_like",
    "rouge_l",
    "token_overlap",
    "keyword_coverage",
    "perplexity_proxy",
}


def _score_case(
    case: EvalCase, response: ChatResponse
) -> Dict[str, float]:
    """Compute every metric we know how to compute for this case."""
    metrics: Dict[str, float] = {}

    # --- Intrinsic (need expected_answer or contexts) ---
    if case.expected_answer:
        metrics["bleu_like"] = bleu_like_score(
            response.answer, case.expected_answer
        )
        metrics["rouge_l"] = rouge_l_score(
            response.answer, case.expected_answer
        )
        metrics["token_overlap"] = token_overlap(
            response.answer, case.expected_answer
        )
        ppl = perplexity_proxy(response.answer, case.expected_answer)
        # Clamp perplexity for aggregation sanity (still recorded raw).
        if ppl != float("inf"):
            metrics["perplexity_proxy"] = ppl

    if case.expected_keywords:
        metrics["keyword_coverage"] = keyword_coverage(
            response.answer, case.expected_keywords
        )

    # --- Extrinsic ---
    if case.expected_keywords:
        metrics["task_completion"] = task_completion_score(
            response.answer, expected_keywords=case.expected_keywords
        )
    if case.expected_sources:
        metrics["retrieval_at_k"] = retrieval_at_k(
            case.expected_sources, response.sources
        )

    contexts = case.contexts or [c.content for c in response.sources]
    if contexts:
        metrics["faithfulness"] = faithfulness_score(
            response.answer, contexts
        )

    metrics["guardrail_pass"] = guardrail_pass_score(response)
    return metrics


class Evaluator:
    """Run a pipeline over a dataset and emit an `EvalRunReport`."""

    def __init__(
        self,
        pipeline: _RunnablePipeline,
        *,
        mode: ChatMode = ChatMode.MULTI,
        top_k: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        self.mode = mode
        self.top_k = top_k

    async def run(
        self,
        dataset: EvalDataset,
        *,
        run_id: Optional[str] = None,
    ) -> EvalRunReport:
        rid = run_id or f"eval-{uuid.uuid4().hex[:8]}"
        started = datetime.utcnow()
        case_results: List[EvalCaseResult] = []

        logger.info(
            "eval_run_start",
            run_id=rid,
            dataset=dataset.name,
            n_cases=len(dataset),
            mode=self.mode.value,
        )

        for case in dataset:
            case_results.append(await self._run_case(case))

        finished = datetime.utcnow()
        aggregate = self._aggregate(case_results)
        logger.info(
            "eval_run_end",
            run_id=rid,
            n_cases=len(case_results),
            n_errors=aggregate.n_errors,
            guardrail_pass_rate=aggregate.guardrail_pass_rate,
        )

        return EvalRunReport(
            run_id=rid,
            dataset_name=dataset.name,
            mode=self.mode,
            n_cases=len(case_results),
            aggregate=aggregate,
            cases=case_results,
            started_at=started,
            finished_at=finished,
        )

    async def _run_case(self, case: EvalCase) -> EvalCaseResult:
        t0 = time.perf_counter()
        try:
            response = await self.pipeline.run(
                case.query,
                history=None,
                session_id=None,
                top_k=self.top_k,
            )
        except Exception as exc:  # pragma: no cover - defensive
            elapsed = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "eval_case_failed",
                case_id=case.case_id,
                error=str(exc),
            )
            return EvalCaseResult(
                case_id=case.case_id,
                query=case.query,
                answer="",
                metrics={},
                latency_ms=elapsed,
                tokens_used=0,
                error=str(exc),
            )

        metrics = _score_case(case, response)
        verdict, flags = _summarise_guardrail(response)
        return EvalCaseResult(
            case_id=case.case_id,
            query=case.query,
            answer=response.answer,
            metrics=metrics,
            latency_ms=response.total_latency_ms,
            tokens_used=response.total_tokens,
            guardrail_verdict=verdict,
            guardrail_flags=flags,
        )

    @staticmethod
    def _aggregate(results: List[EvalCaseResult]) -> AggregateMetrics:
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        latency_total = 0
        tokens_total = 0
        n_errors = 0
        n_pass = 0
        n_guardrail_scored = 0

        for r in results:
            if r.error:
                n_errors += 1
                continue
            latency_total += r.latency_ms
            tokens_total += r.tokens_used
            for name, value in r.metrics.items():
                sums[name] = sums.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1
            if "guardrail_pass" in r.metrics:
                n_guardrail_scored += 1
                if r.metrics["guardrail_pass"] >= 0.999:
                    n_pass += 1

        completed = len(results) - n_errors
        means = {name: sums[name] / counts[name] for name in sums}
        return AggregateMetrics(
            means=means,
            counts=counts,
            avg_latency_ms=(latency_total / completed) if completed else 0.0,
            avg_tokens=(tokens_total / completed) if completed else 0.0,
            guardrail_pass_rate=(
                n_pass / n_guardrail_scored if n_guardrail_scored else 0.0
            ),
            n_cases=len(results),
            n_errors=n_errors,
        )


def _summarise_guardrail(response: ChatResponse) -> tuple[Optional[str], list[str]]:
    """Return (worst_verdict, all_flags) across all traces."""
    worst = GuardrailVerdict.PASS
    rank = {GuardrailVerdict.PASS: 0, GuardrailVerdict.WARN: 1, GuardrailVerdict.BLOCK: 2}
    seen_flags: list[str] = []
    found_any = False
    for trace in response.traces:
        report = trace.guardrail_report
        if report is None:
            continue
        found_any = True
        if rank[report.verdict] > rank[worst]:
            worst = report.verdict
        for f in report.flags:
            if f not in seen_flags:
                seen_flags.append(f)
    return (worst.value if found_any else None, seen_flags)


__all__ = ["Evaluator"]
