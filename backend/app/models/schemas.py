"""
API-layer Pydantic schemas (request / response DTOs).

These schemas define the public contract of the FastAPI service.
Keep them stable; internal domain entities live in `domain.py`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------- #
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CRITIC = "critic"
    ORCHESTRATOR = "orchestrator"


class ChatMode(str, Enum):
    """Pipeline selector for /chat. ``single`` = Researcher only,
    ``multi`` = Researcher -> Analyst -> Critic."""

    SINGLE = "single"
    MULTI = "multi"


class GuardrailVerdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class GuardrailCategory(str, Enum):
    """Risk categories tracked by the GuardrailMonitor (Step 5)."""

    PII = "pii"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    BIAS = "bias"
    RADICALIZATION = "radicalization"
    PROMPT_INJECTION = "prompt_injection"
    REFUSAL_OVERREACH = "refusal_overreach"


class GuardrailSeverity(str, Enum):
    """Severity of an individual finding."""

    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


# --------------------------------------------------------------------- #
# Generic
# --------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


# --------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=10_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=4_000)
    history: List[ChatMessage] = Field(default_factory=list)
    session_id: Optional[UUID] = None
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    mode: ChatMode = Field(default=ChatMode.MULTI)


class RetrievedChunk(BaseModel):
    content: str
    source: Optional[str] = None
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------- #
class GuardrailFinding(BaseModel):
    """A single risk signal produced by one detector."""

    category: GuardrailCategory
    severity: GuardrailSeverity
    message: str
    evidence: Optional[str] = None
    detector: str = "monitor"


class GuardrailReport(BaseModel):
    """Aggregate guardrail decision for one agent output."""

    verdict: GuardrailVerdict
    flags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    findings: List[GuardrailFinding] = Field(default_factory=list)
    monitor_verdict: Optional[GuardrailVerdict] = None
    redacted_text: Optional[str] = None


class AgentTrace(BaseModel):
    """Per-agent breakdown for observability / UI."""
    agent_role: AgentRole
    input: str
    output: str
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    guardrail_report: Optional[GuardrailReport] = None
    latency_ms: int = 0
    tokens_used: int = 0


class ChatResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    session_id: Optional[UUID] = None
    answer: str
    traces: List[AgentTrace] = Field(default_factory=list)
    sources: List[RetrievedChunk] = Field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------- #
class IngestTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    success: bool
    source: str
    n_chunks: int
    embedding_dim: int
    embedding_model: str
    stored_ids: List[str] = Field(default_factory=list)
    skipped_reason: Optional[str] = None
    collection_total: int = 0
    # Optional - populated when ingesting a PDF so the UI can show the
    # page count and how many characters of text were actually extracted.
    n_pages: Optional[int] = None
    n_chars_extracted: Optional[int] = None


class CollectionStatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    embedding_model: str
    embedding_dim: int
