// Thin fetch wrapper. The dev server proxies /api/* -> http://localhost:8000/*
// (see vite.config.ts). In production builds the same /api prefix should be
// served by the reverse proxy / Docker compose.

import type {
  ChatRequest,
  ChatResponse,
  CollectionStats,
  EvalRunRequest,
  EvalRunReport,
  HealthResponse,
  IngestResponse,
  IngestTextRequest,
  SampleDatasetResponse,
} from "../types/api";

const API_BASE = "/api";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  // For multipart/form-data, the browser sets the Content-Type
  // (with boundary) automatically. Only set JSON content-type when
  // the caller sent a string body and didn't set one.
  const isFormData = init.body instanceof FormData;
  if (!isFormData && !headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch (err) {
    throw new ApiError(0, null, `Network error contacting backend: ${String(err)}`);
  }

  const text = await response.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }

  if (!response.ok) {
    const detail =
      (parsed && typeof parsed === "object" && "detail" in parsed
        ? (parsed as { detail: unknown }).detail
        : parsed) ?? response.statusText;
    throw new ApiError(
      response.status,
      detail,
      typeof detail === "string"
        ? detail
        : `Request failed with status ${response.status}`,
    );
  }

  return parsed as T;
}

// --- Endpoints -----------------------------------------------------------
export const api = {
  health: () => request<HealthResponse>("/health"),

  chat: (payload: ChatRequest) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  ingestText: (payload: IngestTextRequest) =>
    request<IngestResponse>("/ingest/text", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  /**
   * Upload a PDF file. The server extracts text via pypdf, then runs
   * it through the same chunk+embed+store pipeline as ingestText.
   *
   * @param file      The PDF File (from a file input or drop event).
   * @param options   Optional metadata + source override. Metadata is
   *                  serialised to JSON and sent as the `metadata_str`
   *                  form field; the backend parses it on arrival.
   */
  ingestPdf: (
    file: File,
    options: {
      metadata?: Record<string, unknown>;
      source?: string;
    } = {},
  ) => {
    const form = new FormData();
    form.append("file", file, file.name);
    if (options.metadata && Object.keys(options.metadata).length > 0) {
      form.append("metadata_str", JSON.stringify(options.metadata));
    }
    if (options.source) {
      form.append("source", options.source);
    }
    return request<IngestResponse>("/ingest/pdf", {
      method: "POST",
      body: form,
    });
  },

  collectionStats: () => request<CollectionStats>("/ingest/stats"),

  evalRun: (payload: EvalRunRequest) =>
    request<EvalRunReport>("/eval/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  evalSample: () => request<SampleDatasetResponse>("/eval/sample-dataset"),
};

export type Api = typeof api;
