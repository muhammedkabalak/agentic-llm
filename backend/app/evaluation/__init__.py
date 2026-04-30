"""
Evaluation package (Step 6).

Public surface:
    EvalCase, EvalDataset                    -- inputs
    EvalCaseResult, EvalRunReport            -- outputs
    Evaluator                                -- runs a pipeline over a dataset

The metrics are split into two families that mirror the project
document's "Evaluation" requirement:

* Intrinsic metrics (no ground-truth needed beyond the reference text):
  BLEU-like, ROUGE-L, char-bigram perplexity proxy, keyword coverage.
* Extrinsic metrics (require labelled cases):
  task completion, faithfulness vs retrieved context, retrieval@k,
  guardrail pass-rate, latency, tokens.

All metrics are pure functions returning floats in [0, 1] (or non-
negative numbers for latency/tokens). They run offline -- no LLM
calls, no model downloads -- so the whole eval suite is fast and CI-
friendly.
"""

from app.evaluation.dataset import EvalCase, EvalDataset
from app.evaluation.evaluator import Evaluator
from app.evaluation.report import (
    AggregateMetrics,
    EvalCaseResult,
    EvalRunReport,
)

__all__ = [
    "AggregateMetrics",
    "EvalCase",
    "EvalCaseResult",
    "EvalDataset",
    "EvalRunReport",
    "Evaluator",
]
