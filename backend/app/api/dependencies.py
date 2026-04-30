"""FastAPI dependency-injection providers."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.agents.analyst_agent import AnalystAgent
from app.agents.critic_agent import CriticAgent
from app.agents.orchestrator import MultiAgentOrchestrator, SingleAgentPipeline
from app.agents.researcher_agent import ResearcherAgent
from app.config import Settings, get_settings
from app.guardrails.monitor import GuardrailMonitor
from app.rag.embeddings import BaseEmbedder, get_embedder
from app.rag.ingestion_pipeline import IngestionPipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import BaseVectorStore, get_vector_store
from app.services.llm_provider import BaseLLMProvider, get_llm_provider


@lru_cache(maxsize=1)
def _cached_llm_provider() -> BaseLLMProvider:
    return get_llm_provider()


@lru_cache(maxsize=1)
def _cached_embedder() -> BaseEmbedder:
    return get_embedder()


@lru_cache(maxsize=1)
def _cached_vector_store() -> BaseVectorStore:
    return get_vector_store()


@lru_cache(maxsize=1)
def _cached_retriever() -> Retriever:
    return Retriever(_cached_embedder(), _cached_vector_store())


@lru_cache(maxsize=1)
def _cached_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline.from_components(
        embedder=_cached_embedder(),
        vector_store=_cached_vector_store(),
    )


@lru_cache(maxsize=1)
def _cached_researcher_agent() -> ResearcherAgent:
    return ResearcherAgent(_cached_llm_provider())


@lru_cache(maxsize=1)
def _cached_analyst_agent() -> AnalystAgent:
    return AnalystAgent(_cached_llm_provider())


@lru_cache(maxsize=1)
def _cached_critic_agent() -> CriticAgent:
    return CriticAgent(_cached_llm_provider())


@lru_cache(maxsize=1)
def _cached_guardrail_monitor() -> GuardrailMonitor:
    return GuardrailMonitor()


@lru_cache(maxsize=1)
def _cached_single_agent_pipeline() -> SingleAgentPipeline:
    return SingleAgentPipeline(
        retriever=_cached_retriever(),
        agent=_cached_researcher_agent(),
        guardrail_monitor=_cached_guardrail_monitor(),
    )


@lru_cache(maxsize=1)
def _cached_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    return MultiAgentOrchestrator(
        retriever=_cached_retriever(),
        researcher=_cached_researcher_agent(),
        analyst=_cached_analyst_agent(),
        critic=_cached_critic_agent(),
        enable_critic=True,
        guardrail_monitor=_cached_guardrail_monitor(),
    )


def settings_dep() -> Settings:
    return get_settings()


def llm_provider_dep(settings: Settings = Depends(settings_dep)) -> BaseLLMProvider:
    return _cached_llm_provider()


def embedder_dep(settings: Settings = Depends(settings_dep)) -> BaseEmbedder:
    return _cached_embedder()


def vector_store_dep(settings: Settings = Depends(settings_dep)) -> BaseVectorStore:
    return _cached_vector_store()


def retriever_dep(settings: Settings = Depends(settings_dep)) -> Retriever:
    return _cached_retriever()


def ingestion_pipeline_dep(settings: Settings = Depends(settings_dep)) -> IngestionPipeline:
    return _cached_ingestion_pipeline()


def researcher_agent_dep(settings: Settings = Depends(settings_dep)) -> ResearcherAgent:
    return _cached_researcher_agent()


def analyst_agent_dep(settings: Settings = Depends(settings_dep)) -> AnalystAgent:
    return _cached_analyst_agent()


def critic_agent_dep(settings: Settings = Depends(settings_dep)) -> CriticAgent:
    return _cached_critic_agent()


def guardrail_monitor_dep(
    settings: Settings = Depends(settings_dep),
) -> GuardrailMonitor:
    return _cached_guardrail_monitor()


def single_agent_pipeline_dep(
    settings: Settings = Depends(settings_dep),
) -> SingleAgentPipeline:
    return _cached_single_agent_pipeline()


def multi_agent_orchestrator_dep(
    settings: Settings = Depends(settings_dep),
) -> MultiAgentOrchestrator:
    return _cached_multi_agent_orchestrator()
