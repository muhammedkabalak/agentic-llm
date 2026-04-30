"""
Chat endpoint - RAG-powered, dispatches between Single and Multi-agent
pipelines based on `payload.mode`.

  * mode="single" -> SingleAgentPipeline (Researcher only)
  * mode="multi"  -> MultiAgentOrchestrator (Researcher -> Analyst -> Critic)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.agents.orchestrator import MultiAgentOrchestrator, SingleAgentPipeline
from app.api.dependencies import (
    multi_agent_orchestrator_dep,
    single_agent_pipeline_dep,
)
from app.models.schemas import ChatMode, ChatRequest, ChatResponse
from app.services.llm_provider import LLMProviderError
from app.services.logging_service import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    single_pipeline: SingleAgentPipeline = Depends(single_agent_pipeline_dep),
    multi_pipeline: MultiAgentOrchestrator = Depends(multi_agent_orchestrator_dep),
) -> ChatResponse:
    """
    RAG-powered chat. Dispatches on `payload.mode`:
      1. Retrieve top_k chunks from the vector store.
      2. Run the selected pipeline against the retrieved context.
      3. Return the grounded answer with per-agent traces, sources,
         and (in multi mode) a Critic-derived guardrail report.
    """
    pipeline = (
        multi_pipeline if payload.mode == ChatMode.MULTI else single_pipeline
    )
    try:
        return await pipeline.run(
            query=payload.query,
            history=payload.history,
            session_id=payload.session_id,
            top_k=payload.top_k,
        )
    except LLMProviderError as exc:
        logger.error(
            "chat_endpoint_llm_error", error=str(exc), mode=payload.mode.value
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM call failed.",
        ) from exc
