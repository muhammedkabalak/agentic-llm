"""Unit tests for EvalCase / EvalDataset loaders."""

from __future__ import annotations

import json

import pytest

from app.evaluation.dataset import EvalCase, EvalDataset, sample_dataset


def test_eval_case_from_dict_minimal() -> None:
    c = EvalCase.from_dict({"query": "What is RAG?"})
    assert c.query == "What is RAG?"
    assert c.expected_keywords == []
    assert c.expected_sources == []
    assert c.contexts == []


def test_eval_case_from_dict_full() -> None:
    c = EvalCase.from_dict(
        {
            "case_id": "rag-1",
            "query": "What is RAG?",
            "expected_answer": "Retrieval-augmented generation.",
            "expected_keywords": ["retrieval", "generation"],
            "expected_sources": ["doc-a.md"],
            "contexts": ["Some context."],
            "metadata": {"tag": "smoke"},
        }
    )
    assert c.case_id == "rag-1"
    assert c.expected_keywords == ["retrieval", "generation"]
    assert c.expected_sources == ["doc-a.md"]
    assert c.contexts == ["Some context."]
    assert c.metadata == {"tag": "smoke"}


def test_eval_case_requires_query() -> None:
    with pytest.raises(ValueError):
        EvalCase.from_dict({})
    with pytest.raises(ValueError):
        EvalCase.from_dict({"query": "   "})


def test_eval_dataset_from_iterable() -> None:
    ds = EvalDataset.from_iterable(
        [{"query": "a"}, {"query": "b", "case_id": "x"}], name="demo"
    )
    assert ds.name == "demo"
    assert len(ds) == 2
    assert [c.query for c in ds] == ["a", "b"]
    assert ds.cases[1].case_id == "x"


def test_eval_dataset_from_json(tmp_path) -> None:
    path = tmp_path / "ds.json"
    path.write_text(
        json.dumps(
            [
                {"query": "hello"},
                {"query": "world", "expected_keywords": ["w"]},
            ]
        ),
        encoding="utf-8",
    )
    ds = EvalDataset.from_json(path)
    assert ds.name == "ds"
    assert len(ds) == 2
    assert ds.cases[1].expected_keywords == ["w"]


def test_eval_dataset_from_json_top_level_must_be_list(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"query": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError):
        EvalDataset.from_json(path)


def test_eval_dataset_from_jsonl(tmp_path) -> None:
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"query": "a"}\n\n{"query": "b"}\n', encoding="utf-8"
    )
    ds = EvalDataset.from_jsonl(path)
    assert len(ds) == 2  # blank line skipped
    assert [c.query for c in ds] == ["a", "b"]


def test_sample_dataset_has_cases() -> None:
    ds = sample_dataset()
    assert ds.name == "sample"
    assert len(ds) >= 2
    assert all(c.query for c in ds)
