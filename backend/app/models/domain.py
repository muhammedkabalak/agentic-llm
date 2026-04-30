"""
Internal domain entities used between agents, RAG, and guardrails.

These are NOT part of the public HTTP contract — keep them
flexible and refactor freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID, uuid4


@dataclass
class AgentContext:
    """The shared context object passed through an agent crew."""
    request_id: UUID = field(default_factory=uuid4)
    session_id: Optional[UUID] = None
    query: str = ""
    history: List[dict[str, str]] = field(default_factory=list)
    retrieved_chunks: List["DocumentChunk"] = field(default_factory=list)
    intermediate_outputs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DocumentChunk:
    """A retrieved chunk from the vector store."""
    content: str
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result returned from an agent's `run` method."""
    agent_role: str
    output: str
    tokens_used: int = 0
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
