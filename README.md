# [cite_start]Agentic LLM Creation and Monitoring [cite: 3]

[cite_start]A Retrieval-Augmented, Multi-Agent Question-Answering System with Independent Guardrail Monitoring and Offline Evaluation. [cite: 4]

## 📖 About The Project

[cite_start]This project is an end-to-end retrieval-augmented (RAG) multi-agent question-answering system developed for the Introduction to LLM course project titled "Agentic LLM Creation and Monitoring"[cite: 14]. [cite_start]It was completed for the 2025-2026 academic year[cite: 11]. [cite_start]It answers user queries by retrieving relevant context from a vector store and routing it through an ordered chain of three specialized agents[cite: 15]. [cite_start]By combining retrieval grounding with multi-agent self-criticism and a deterministic safety layer, the system significantly improves answer faithfulness and auditability compared to single-shot LLM baselines[cite: 22].

## ✨ Key Features

* [cite_start]**Multi-Agent Orchestration:** Queries traverse a sequential chain of three agents—a Researcher that drafts, an Analyst that structures, and a Critic that verifies—all sharing a common context object[cite: 15].
* [cite_start]**Advanced RAG Pipeline:** Supports raw text and PDF ingestion, overlapping chunking strategies, dense embeddings, and semantic retrieval with score thresholding[cite: 6, 154, 156].
* [cite_start]**Deterministic Guardrail Monitor:** An independent safety layer that inspects every agent output for PII, hallucination, toxicity, bias, radicalization, prompt-injection echoes, and refusal overreach[cite: 18]. [cite_start]It automatically redacts PII or substitutes safe fallbacks for severe violations[cite: 18].
* [cite_start]**Comprehensive Evaluation:** Features an offline evaluation pipeline that reports intrinsic metrics (e.g., BLEU-like, ROUGE-L, perplexity proxy) and extrinsic metrics (e.g., task completion, faithfulness, retrieval@k)[cite: 19].
* [cite_start]**Modern Web Interface:** A Vite, React, and TypeScript single-page application that visualizes agent traces, guardrail verdicts, and per-case evaluation results in real-time[cite: 20].

## 🛠️ Technology Stack

* [cite_start]**Backend:** Python 3.12, FastAPI, Pydantic, Structlog[cite: 16, 149, 151].
* [cite_start]**AI & Orchestration:** LangChain, OpenAI, and local Ollama support[cite: 17, 151].
* [cite_start]**Retrieval & Storage:** ChromaDB, `sentence-transformers`, `pypdf`[cite: 16, 151].
* [cite_start]**Frontend:** Node.js, React, Vite, TypeScript[cite: 20, 149, 151].
* [cite_start]**Testing:** 154 fully passing unit and integration tests utilizing Pytest for the backend and Vitest for the frontend[cite: 21, 150, 151].

## 🔗 Links & Resources

* [cite_start]**GitHub Repository:** [muhammedkabalak/agentic-llm](https://github.com/muhammedkabalak/agentic-llm) [cite: 9]
* [cite_start]**Presentation & Demo:** [Watch on YouTube](https://www.youtube.com/watch?v=m5z386WrJKE) [cite: 10]

## 👥 Team Members

[cite_start]All members participated in code review, testing, and the preparation of the report[cite: 7]. 

* [cite_start]**Muhammed Kabalak**: Architecture & Backend (Overall system design; FastAPI backend; agent base class and orchestrator; integration of all subsystems; project coordination)[cite: 6].
* [cite_start]**Mert Savaşer**: RAG/Retrieval (Chunking strategies, sentence-transformers embeddings, ChromaDB vector store, retriever with score thresholding, ingestion pipeline)[cite: 6].
* [cite_start]**Halil Ömer Soysal**: Multi-Agent Layer (Researcher, Analyst, and Critic agent prompts and parsers; shared AgentContext design; structured Critic output protocol)[cite: 6].
* [cite_start]**Muhammet Baha Öğütlü**: Guardrails & Safety (GuardrailMonitor module; PII / hallucination / toxicity / bias / radicalisation / prompt-injection / refusal-overreach detectors; redaction logic)[cite: 6].
* [cite_start]**Berke Eren Akçay**: Evaluation & Frontend (Intrinsic + extrinsic metrics, evaluator pipeline, React + TypeScript user interface, end-to-end smoke tests)[cite: 6].
