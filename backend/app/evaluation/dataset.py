"""
Eval dataset abstractions.

`EvalCase` is the labelled unit: a query plus optional gold-standard
fields used by the extrinsic metrics. `EvalDataset` is a thin wrapper
around a list of cases that knows how to read JSON / JSONL / dict
sources so the same dataset can be persisted alongside the codebase
or shipped over HTTP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence


@dataclass
class EvalCase:
    """A single labelled evaluation case.

    Only `query` is required; any of the other fields may be ``None``
    or empty, in which case the corresponding metrics are skipped for
    that case (and excluded from the aggregate average).
    """

    query: str
    case_id: Optional[str] = None
    expected_answer: Optional[str] = None
    expected_keywords: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        if "query" not in data or not str(data["query"]).strip():
            raise ValueError("EvalCase requires a non-empty 'query'.")
        return cls(
            query=str(data["query"]),
            case_id=data.get("case_id"),
            expected_answer=data.get("expected_answer"),
            expected_keywords=list(data.get("expected_keywords", []) or []),
            expected_sources=list(data.get("expected_sources", []) or []),
            contexts=list(data.get("contexts", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class EvalDataset:
    """A collection of `EvalCase` instances."""

    cases: List[EvalCase] = field(default_factory=list)
    name: Optional[str] = None

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    @classmethod
    def from_iterable(
        cls, items: Iterable[dict], *, name: Optional[str] = None
    ) -> "EvalDataset":
        return cls(cases=[EvalCase.from_dict(d) for d in items], name=name)

    @classmethod
    def from_json(cls, path: Path | str) -> "EvalDataset":
        """Read a JSON array of case dicts."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(
                "Eval JSON must be a list of case objects at the top level."
            )
        return cls.from_iterable(data, name=p.stem)

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "EvalDataset":
        """Read newline-delimited JSON, one case per line."""
        p = Path(path)
        cases: List[EvalCase] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase.from_dict(json.loads(line)))
        return cls(cases=cases, name=p.stem)


def sample_dataset() -> EvalDataset:
    """A tiny baked-in dataset useful for smoke tests and demos."""
    return EvalDataset(
        name="sample",
        cases=[
            EvalCase(
                case_id="rag-1",
                query="What is RAG?",
                expected_answer=(
                    "Retrieval-augmented generation combines retrieval "
                    "with generation."
                ),
                expected_keywords=["retrieval", "generation"],
                expected_sources=["doc-a.md"],
            ),
            EvalCase(
                case_id="vdb-1",
                query="Where are embeddings stored?",
                expected_answer=(
                    "Vector databases store embeddings for similarity search."
                ),
                expected_keywords=["vector", "embeddings"],
                expected_sources=["doc-b.md"],
            ),
        ],
    )


__all__ = ["EvalCase", "EvalDataset", "sample_dataset"]
