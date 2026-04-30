"""
Eval report types (Pydantic so they round-trip over HTTP).

A run produces:
  * one `EvalCaseResult` per case (raw answer + per-metric scores),
  * one `AggregateMetrics` object (means across cases),
  * one `EvalRunReport` that bundles them with run-level metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.schemas import ChatMode


class EvalCaseResult(BaseModel):
    """Per-case result. ``metrics`` is a flat name -> float map so new
    metrics can be added without changing the schema."""

    case_id: Optional[str] = None
    query: str
    answer: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    latency_ms: int = 0
    tokens_used: int = 0
    guardrail_verdict: Optional[str] = None
    guardrail_flags: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class AggregateMetrics(BaseModel):
    """Per-metric summary stats across all completed cases."""

    means: Dict[str, float] = Field(default_factory=dict)
    counts: Dict[str, int] = Field(default_factory=dict)
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    guardrail_pass_rate: float = 0.0
    n_cases: int = 0
    n_errors: int = 0


class EvalRunReport(BaseModel):
    """Top-level report returned by ``Evaluator.run``."""

    run_id: str
    dataset_name: Optional[str] = None
    mode: ChatMode = ChatMode.MULTI
    n_cases: int = 0
    aggregate: AggregateMetrics = Field(default_factory=AggregateMetrics)
    cases: List[EvalCaseResult] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = ["AggregateMetrics", "EvalCaseResult", "EvalRunReport"]
