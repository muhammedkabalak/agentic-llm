"""
Evaluation endpoint.

POST /eval/run
    Body: {"cases": [...], "mode": "single|multi", "top_k": int}
    Runs every case through the selected pipeline and returns an
    `EvalRunReport`.

GET /eval/sample-dataset
    Returns the baked-in demo dataset useful for client-side smoke
    tests and the README.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.agents.orchestrator import MultiAgentOrchestrator, SingleAgentPipeline
from app.api.dependencies import (
    multi_agent_orchestrator_dep,
    single_agent_pipeline_dep,
)
from app.evaluation.dataset import EvalCase, EvalDataset, sample_dataset
from app.evaluation.evaluator import Evaluator
from app.evaluation.report import EvalRunReport
from app.models.schemas import ChatMode
from app.services.llm_provider import LLMProviderError
from app.services.logging_service import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])


# --------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------- #
class EvalCaseInput(BaseModel):
    """Wire-level case (mirrors `EvalCase` but Pydantic-validated)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=4_000)
    case_id: Optional[str] = Field(default=None, max_length=128)
    expected_answer: Optional[str] = Field(default=None, max_length=10_000)
    expected_keywords: List[str] = Field(default_factory=list)
    expected_sources: List[str] = Field(default_factory=list)
    contexts: List[str] = Field(default_factory=list)


class EvalRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: List[EvalCaseInput] = Field(..., min_length=1, max_length=200)
    mode: ChatMode = Field(default=ChatMode.MULTI)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    dataset_name: Optional[str] = Field(default=None, max_length=128)


class SampleDatasetResponse(BaseModel):
    name: str
    cases: List[EvalCaseInput]


# --------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------- #
@router.post("/run", response_model=EvalRunReport)
async def run_eval(
    payload: EvalRunRequest,
    single_pipeline: SingleAgentPipeline = Depends(single_agent_pipeline_dep),
    multi_pipeline: MultiAgentOrchestrator = Depends(multi_agent_orchestrator_dep),
) -> EvalRunReport:
    pipeline = (
        multi_pipeline if payload.mode == ChatMode.MULTI else single_pipeline
    )
    dataset = EvalDataset(
        name=payload.dataset_name,
        cases=[
            EvalCase(
                query=c.query,
                case_id=c.case_id,
                expected_answer=c.expected_answer,
                expected_keywords=list(c.expected_keywords),
                expected_sources=list(c.expected_sources),
                contexts=list(c.contexts),
            )
            for c in payload.cases
        ],
    )
    evaluator = Evaluator(
        pipeline=pipeline, mode=payload.mode, top_k=payload.top_k
    )
    try:
        return await evaluator.run(dataset)
    except LLMProviderError as exc:
        logger.error("eval_endpoint_llm_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM call failed during evaluation.",
        ) from exc


@router.get("/sample-dataset", response_model=SampleDatasetResponse)
async def get_sample_dataset() -> SampleDatasetResponse:
    ds = sample_dataset()
    return SampleDatasetResponse(
        name=ds.name or "sample",
        cases=[
            EvalCaseInput(
                query=c.query,
                case_id=c.case_id,
                expected_answer=c.expected_answer,
                expected_keywords=list(c.expected_keywords),
                expected_sources=list(c.expected_sources),
                contexts=list(c.contexts),
            )
            for c in ds.cases
        ],
    )
