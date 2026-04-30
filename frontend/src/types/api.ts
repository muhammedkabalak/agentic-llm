// TypeScript mirrors of the backend Pydantic schemas.
// Keep this file in sync with backend/app/models/schemas.py - it is
// the single source of truth on the wire.

export type ChatMode = "single" | "multi";
export type GuardrailVerdict = "pass" | "warn" | "block";
export type GuardrailSeverity = "info" | "warn" | "block";
export type AgentRole = "researcher" | "analyst" | "critic" | "orchestrator";
export type GuardrailCategory =
  | "pii"
  | "hallucination"
  | "toxicity"
  | "bias"
  | "radicalization"
  | "prompt_injection"
  | "refusal_overreach";

export interface RetrievedChunk {
  content: string;
  source: string | null;
  score: number;
  metadata: Record<string, unknown>;
}

export interface GuardrailFinding {
  category: GuardrailCategory;
  severity: GuardrailSeverity;
  message: string;
  evidence: string | null;
  detector: string;
}

export interface GuardrailReport {
  verdict: GuardrailVerdict;
  flags: string[];
  notes: string | null;
  findings: GuardrailFinding[];
  monitor_verdict: GuardrailVerdict | null;
  redacted_text: string | null;
}

export interface AgentTrace {
  agent_role: AgentRole;
  input: string;
  output: string;
  retrieved_chunks: RetrievedChunk[];
  guardrail_report: GuardrailReport | null;
  latency_ms: number;
  tokens_used: number;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  query: string;
  history?: ChatMessage[];
  session_id?: string | null;
  top_k?: number;
  temperature?: number | null;
  mode?: ChatMode;
}

export interface ChatResponse {
  request_id: string;
  session_id: string | null;
  answer: string;
  traces: AgentTrace[];
  sources: RetrievedChunk[];
  total_tokens: number;
  total_latency_ms: number;
  timestamp: string;
}

// Ingestion ---------------------------------------------------------------
export interface IngestTextRequest {
  text: string;
  source: string;
  metadata?: Record<string, unknown>;
}

export interface IngestResponse {
  success: boolean;
  source: string;
  n_chunks: number;
  embedding_dim: number;
  embedding_model: string;
  stored_ids: string[];
  skipped_reason: string | null;
  collection_total: number;
  // Optional - present when uploading a PDF.
  n_pages?: number | null;
  n_chars_extracted?: number | null;
}

export interface CollectionStats {
  collection_name: string;
  total_chunks: number;
  embedding_model: string;
  embedding_dim: number;
}

// Eval --------------------------------------------------------------------
export interface EvalCaseInput {
  query: string;
  case_id?: string | null;
  expected_answer?: string | null;
  expected_keywords?: string[];
  expected_sources?: string[];
  contexts?: string[];
}

export interface EvalRunRequest {
  cases: EvalCaseInput[];
  mode?: ChatMode;
  top_k?: number | null;
  dataset_name?: string | null;
}

export interface EvalCaseResult {
  case_id: string | null;
  query: string;
  answer: string;
  metrics: Record<string, number>;
  latency_ms: number;
  tokens_used: number;
  guardrail_verdict: string | null;
  guardrail_flags: string[];
  error: string | null;
}

export interface AggregateMetrics {
  means: Record<string, number>;
  counts: Record<string, number>;
  avg_latency_ms: number;
  avg_tokens: number;
  guardrail_pass_rate: number;
  n_cases: number;
  n_errors: number;
}

export interface EvalRunReport {
  run_id: string;
  dataset_name: string | null;
  mode: ChatMode;
  n_cases: number;
  aggregate: AggregateMetrics;
  cases: EvalCaseResult[];
  started_at: string;
  finished_at: string | null;
  metadata: Record<string, unknown>;
}

export interface SampleDatasetResponse {
  name: string;
  cases: EvalCaseInput[];
}

// Health ------------------------------------------------------------------
export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
}
