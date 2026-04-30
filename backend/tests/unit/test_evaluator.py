"""
End-to-end Evaluator tests.

Wires HashEmbedder + InMemoryVectorStore + MockLLM + a real
SingleAgentPipeline / MultiAgentOrchestrator and runs the Evaluator
against a small dataset.
"""

from __future__ import annotations

import pytest

from app.agents.analyst_agent import AnalystAgent
from app.agents.critic_agent import CriticAgent
from app.agents.orchestrator import MultiAgentOrchestrator, SingleAgentPipeline
from app.agents.researcher_agent import ResearcherAgent
from app.evaluation.dataset import EvalCase, EvalDataset
from app.evaluation.evaluator import Evaluator
from app.models.domain import DocumentChunk
from app.models.schemas import ChatMode
from app.rag.retriever import Retriever

from tests.conftest import HashEmbedder, InMemoryVectorStore, MockLLM


def _seed(embedder: HashEmbedder, store: InMemoryVectorStore) -> None:
    docs = [
        ("RAG combines retrieval with generation.", "doc-a.md"),
        ("Vector databases store embeddings for similarity search.", "doc-b.md"),
        ("FastAPI is a Python web framework.", "doc-c.md"),
    ]
    chunks = [
        DocumentChunk(content=c, source=s, chunk_id=f"c-{i}")
        for i, (c, s) in enumerate(docs)
    ]
    embeddings = embedder.embed_documents([c.content for c in chunks])
    store.add(chunks, embeddings)


@pytest.mark.asyncio
async def test_evaluator_runs_single_agent_dataset(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    llm = MockLLM(canned_response="RAG combines retrieval with generation [1].")
    pipeline = SingleAgentPipeline(
        retriever=retriever, agent=ResearcherAgent(llm), default_top_k=3
    )

    dataset = EvalDataset(
        name="demo",
        cases=[
            EvalCase(
                case_id="rag-1",
                query="What is RAG?",
                expected_answer="RAG combines retrieval with generation.",
                expected_keywords=["retrieval", "generation"],
                expected_sources=["doc-a.md"],
            ),
            EvalCase(
                case_id="vdb-1",
                query="Vector dbs?",
                expected_keywords=["retrieval"],
                expected_sources=["doc-b.md"],
            ),
        ],
    )

    evaluator = Evaluator(pipeline=pipeline, mode=ChatMode.SINGLE, top_k=3)
    report = await evaluator.run(dataset)

    # Top-level shape
    assert report.run_id.startswith("eval-")
    assert report.dataset_name == "demo"
    assert report.n_cases == 2
    assert len(report.cases) == 2

    # Case-level metrics present
    case0 = report.cases[0]
    assert case0.case_id == "rag-1"
    assert case0.answer == "RAG combines retrieval with generation [1]."
    assert "bleu_like" in case0.metrics
    assert "rouge_l" in case0.metrics
    assert "keyword_coverage" in case0.metrics
    assert case0.metrics["keyword_coverage"] == 1.0
    assert case0.metrics["task_completion"] == 1.0
    assert case0.metrics["retrieval_at_k"] == 1.0
    assert case0.guardrail_verdict == "pass"

    # Aggregate
    agg = report.aggregate
    assert agg.n_cases == 2
    assert agg.n_errors == 0
    assert agg.guardrail_pass_rate == 1.0
    assert "task_completion" in agg.means
    # Both cases scored task_completion -> count == 2
    assert agg.counts["task_completion"] == 2


@pytest.mark.asyncio
async def test_evaluator_skips_metric_when_label_absent(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)
    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=2)
    llm = MockLLM(canned_response="Some answer.")
    pipeline = SingleAgentPipeline(
        retriever=retriever, agent=ResearcherAgent(llm), default_top_k=2
    )

    # No labels at all -> only guardrail_pass + faithfulness (from
    # retrieved sources) should be scored.
    dataset = EvalDataset(
        cases=[EvalCase(query="What is RAG?")],
    )
    report = await Evaluator(pipeline=pipeline, mode=ChatMode.SINGLE).run(dataset)

    case = report.cases[0]
    assert "bleu_like" not in case.metrics
    assert "task_completion" not in case.metrics
    assert "retrieval_at_k" not in case.metrics
    assert "guardrail_pass" in case.metrics
    # Faithfulness uses retrieved chunks as the fallback context source.
    assert "faithfulness" in case.metrics


@pytest.mark.asyncio
async def test_evaluator_records_pipeline_error(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    """A single failing case must not abort the run."""

    class BoomPipeline:
        async def run(self, query, *, history=None, session_id=None, top_k=None):
            raise RuntimeError("kaboom")

    dataset = EvalDataset(cases=[EvalCase(query="anything")])
    report = await Evaluator(pipeline=BoomPipeline()).run(dataset)

    assert report.n_cases == 1
    assert report.aggregate.n_errors == 1
    assert report.cases[0].error == "kaboom"
    assert report.cases[0].metrics == {}


@pytest.mark.asyncio
async def test_evaluator_runs_multi_agent(
    hash_embedder: HashEmbedder,
    memory_vector_store: InMemoryVectorStore,
) -> None:
    _seed(hash_embedder, memory_vector_store)

    critic_pass = (
        "VERDICT: pass\nFLAGS: none\nNOTES: ok\nREVISED_ANSWER:\nUNCHANGED\n"
    )

    def router(messages):
        sys = messages[0].content if messages else ""
        if "You are the Researcher Agent" in sys:
            return "Researcher draft [1]."
        if "You are the Analyst Agent" in sys:
            return "## Summary\nRAG combines retrieval with generation [1]."
        if "You are the Critic Agent" in sys:
            return critic_pass
        return ""

    llm = MockLLM(response_fn=router)
    retriever = Retriever(hash_embedder, memory_vector_store, default_top_k=3)
    orch = MultiAgentOrchestrator(
        retriever=retriever,
        researcher=ResearcherAgent(llm),
        analyst=AnalystAgent(llm),
        critic=CriticAgent(llm),
        default_top_k=3,
    )

    dataset = EvalDataset(
        cases=[
            EvalCase(
                case_id="rag-multi",
                query="What is RAG?",
                expected_answer="RAG combines retrieval with generation.",
                expected_keywords=["retrieval", "generation"],
                expected_sources=["doc-a.md"],
            )
        ],
    )
    report = await Evaluator(pipeline=orch, mode=ChatMode.MULTI).run(dataset)

    case = report.cases[0]
    assert case.metrics["task_completion"] == 1.0
    assert case.metrics["retrieval_at_k"] == 1.0
    assert case.guardrail_verdict == "pass"
    # Tokens come from three agents -> > 33 (single agent)
    assert case.tokens_used > 33
