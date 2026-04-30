# RAG Multi-Agent Frontend

Vite + React + TypeScript UI for the RAG multi-agent backend.

## Screens
- **Chat** — converse with the crew (single or multi-agent mode), see
  per-agent traces, guardrail verdicts, redaction notices, and the
  retrieved sources behind every answer.
- **Ingest** — push text into the vector store; live collection
  stats badge.
- **Evaluation** — run a labelled JSON dataset through the system,
  inspect aggregate metrics (BLEU-like, ROUGE-L, faithfulness,
  retrieval@k, guardrail pass-rate) and per-case scores.

## Develop
```bash
npm install
npm run dev        # http://localhost:5173 (proxies /api -> :8000)
npm run test       # vitest
npm run lint       # tsc --noEmit
npm run build      # production bundle into dist/
```

The Vite dev server proxies `/api/*` to `http://localhost:8000/*`,
so no CORS configuration is needed during development. In
production, mount the backend at the same origin (or behind a
reverse proxy that forwards `/api/*`).
