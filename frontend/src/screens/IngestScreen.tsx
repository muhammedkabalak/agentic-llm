import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { CollectionStats, IngestResponse } from "../types/api";

export function IngestScreen() {
  // Shared state (collection stats, surfaced result, top-level error)
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Text-ingest state
  const [text, setText] = useState("");
  const [textSource, setTextSource] = useState("");
  const [textMeta, setTextMeta] = useState("{}");
  const [textPending, setTextPending] = useState(false);

  // PDF-ingest state
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfSource, setPdfSource] = useState("");
  const [pdfMeta, setPdfMeta] = useState("{}");
  const [pdfPending, setPdfPending] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const refreshStats = async () => {
    try {
      setStats(await api.collectionStats());
    } catch {
      // best-effort; don't fail the page if stats are unavailable
    }
  };

  useEffect(() => {
    refreshStats();
  }, []);

  // -- TEXT --------------------------------------------------------------
  const submitText = async () => {
    setError(null);
    setResult(null);
    if (!text.trim() || !textSource.trim()) {
      setError("Both text and source are required.");
      return;
    }
    let parsedMeta: Record<string, unknown> = {};
    if (textMeta.trim()) {
      try {
        parsedMeta = JSON.parse(textMeta);
        if (typeof parsedMeta !== "object" || Array.isArray(parsedMeta)) {
          throw new Error("metadata must be a JSON object");
        }
      } catch (e) {
        setError(`Metadata JSON is invalid: ${(e as Error).message}`);
        return;
      }
    }

    setTextPending(true);
    try {
      const response = await api.ingestText({
        text,
        source: textSource,
        metadata: parsedMeta,
      });
      setResult(response);
      await refreshStats();
    } catch (err) {
      setError(
        err instanceof ApiError ? `${err.status}: ${err.message}` : String(err),
      );
    } finally {
      setTextPending(false);
    }
  };

  // -- PDF ---------------------------------------------------------------
  const pickPdf = (file: File | undefined | null) => {
    setError(null);
    if (!file) {
      setPdfFile(null);
      return;
    }
    const isPdf =
      file.type === "application/pdf" ||
      file.name.toLowerCase().endsWith(".pdf");
    if (!isPdf) {
      setError(`'${file.name}' is not a PDF.`);
      return;
    }
    setPdfFile(file);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    pickPdf(e.dataTransfer.files?.[0]);
  };

  const submitPdf = async () => {
    setError(null);
    setResult(null);
    if (!pdfFile) {
      setError("Please choose a PDF file first.");
      return;
    }
    let parsedMeta: Record<string, unknown> = {};
    if (pdfMeta.trim()) {
      try {
        parsedMeta = JSON.parse(pdfMeta);
        if (typeof parsedMeta !== "object" || Array.isArray(parsedMeta)) {
          throw new Error("metadata must be a JSON object");
        }
      } catch (e) {
        setError(`Metadata JSON is invalid: ${(e as Error).message}`);
        return;
      }
    }

    setPdfPending(true);
    try {
      const response = await api.ingestPdf(pdfFile, {
        metadata: parsedMeta,
        source: pdfSource.trim() || undefined,
      });
      setResult(response);
      // On success, keep file selected for visual feedback but reset
      // the source/meta inputs so the user can upload another quickly.
      setPdfFile(null);
      setPdfSource("");
      setPdfMeta("{}");
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refreshStats();
    } catch (err) {
      setError(
        err instanceof ApiError ? `${err.status}: ${err.message}` : String(err),
      );
    } finally {
      setPdfPending(false);
    }
  };

  return (
    <div className="col" style={{ maxWidth: 760 }}>
      <header>
        <h2 style={{ margin: 0 }}>Ingest documents</h2>
        <p className="muted">
          Push text or a PDF into the vector store. The chunker +
          embedder + ChromaDB collection used by /chat all read from
          here.
        </p>
      </header>

      {stats && (
        <div className="row">
          <span className="badge badge--accent">
            {stats.total_chunks} chunks indexed
          </span>
          <span className="faint mono">
            {stats.embedding_model} · dim {stats.embedding_dim}
          </span>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* PDF upload                                                          */}
      {/* ------------------------------------------------------------------ */}
      <div className="card card--padded col">
        <div className="row" style={{ alignItems: "baseline" }}>
          <h3 style={{ margin: 0, flex: 1 }}>Upload a PDF</h3>
          <span className="faint mono" style={{ fontSize: 12 }}>
            POST /ingest/pdf
          </span>
        </div>
        <p className="muted" style={{ margin: 0 }}>
          Drag-and-drop or pick a file. The server extracts text with
          pypdf and feeds it through the same chunking + embedding +
          vector-store path as raw text.
        </p>

        <div
          className={`dropzone${isDragging ? " dropzone--active" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Choose PDF"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf,.pdf"
            style={{ display: "none" }}
            onChange={(e) => pickPdf(e.target.files?.[0])}
          />
          {pdfFile ? (
            <div className="col" style={{ gap: 4, alignItems: "center" }}>
              <div className="badge badge--pass">PDF selected</div>
              <div className="mono" style={{ fontSize: 13 }}>
                {pdfFile.name}
              </div>
              <div className="faint mono" style={{ fontSize: 12 }}>
                {(pdfFile.size / 1024).toFixed(1)} KB
              </div>
            </div>
          ) : (
            <div className="col" style={{ gap: 4, alignItems: "center" }}>
              <div style={{ fontSize: 22 }}>📄</div>
              <div>
                <strong>Drop a PDF here</strong>{" "}
                <span className="faint">or click to browse</span>
              </div>
              <div className="faint mono" style={{ fontSize: 11.5 }}>
                Up to 25 MB
              </div>
            </div>
          )}
        </div>

        <label>
          <div className="section-title">Source label (optional)</div>
          <input
            value={pdfSource}
            onChange={(e) => setPdfSource(e.target.value)}
            placeholder="defaults to the filename"
          />
        </label>
        <label>
          <div className="section-title">Metadata (JSON, optional)</div>
          <textarea
            value={pdfMeta}
            onChange={(e) => setPdfMeta(e.target.value)}
            rows={2}
            className="mono"
          />
        </label>

        <div className="row row--end">
          {pdfFile && (
            <button
              className="btn btn--ghost"
              onClick={() => {
                setPdfFile(null);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
              disabled={pdfPending}
            >
              Clear
            </button>
          )}
          <button
            className="btn"
            onClick={submitPdf}
            disabled={pdfPending || !pdfFile}
          >
            {pdfPending ? "Extracting + indexing…" : "Upload PDF"}
          </button>
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Text ingest                                                         */}
      {/* ------------------------------------------------------------------ */}
      <div className="card card--padded col">
        <div className="row" style={{ alignItems: "baseline" }}>
          <h3 style={{ margin: 0, flex: 1 }}>Or paste raw text</h3>
          <span className="faint mono" style={{ fontSize: 12 }}>
            POST /ingest/text
          </span>
        </div>
        <label>
          <div className="section-title">Source label</div>
          <input
            value={textSource}
            onChange={(e) => setTextSource(e.target.value)}
            placeholder="e.g. handbook/section-3.md"
          />
        </label>
        <label>
          <div className="section-title">Text</div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            placeholder="Paste the text you want chunked, embedded, and stored…"
          />
        </label>
        <label>
          <div className="section-title">Metadata (JSON)</div>
          <textarea
            value={textMeta}
            onChange={(e) => setTextMeta(e.target.value)}
            rows={2}
            className="mono"
          />
        </label>
        <div className="row row--end">
          <button
            className="btn"
            onClick={submitText}
            disabled={textPending}
          >
            {textPending ? "Ingesting…" : "Ingest text"}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="card card--padded">
          <div className="row" style={{ marginBottom: 8 }}>
            <span
              className={
                result.success ? "badge badge--pass" : "badge badge--block"
              }
            >
              {result.success ? "Ingested" : "Skipped"}
            </span>
            <span className="muted">{result.source}</span>
          </div>
          <dl className="kv">
            <dt>Chunks added</dt>
            <dd className="mono">{result.n_chunks}</dd>
            {result.n_pages != null && (
              <>
                <dt>PDF pages</dt>
                <dd className="mono">{result.n_pages}</dd>
              </>
            )}
            {result.n_chars_extracted != null && (
              <>
                <dt>Characters extracted</dt>
                <dd className="mono">
                  {result.n_chars_extracted.toLocaleString()}
                </dd>
              </>
            )}
            <dt>Embedding model</dt>
            <dd className="mono">{result.embedding_model}</dd>
            <dt>Embedding dim</dt>
            <dd className="mono">{result.embedding_dim}</dd>
            <dt>Collection total</dt>
            <dd className="mono">{result.collection_total}</dd>
            {result.skipped_reason && (
              <>
                <dt>Skipped because</dt>
                <dd className="muted">{result.skipped_reason}</dd>
              </>
            )}
          </dl>
        </div>
      )}
    </div>
  );
}
