import { useState } from "react";
import { api, ApiError } from "../api/client";
import type {
  ChatMode,
  EvalCaseInput,
  EvalRunReport,
} from "../types/api";

export function EvalScreen() {
  const [mode, setMode] = useState<ChatMode>("multi");
  const [datasetText, setDatasetText] = useState(
    JSON.stringify(
      [
        {
          case_id: "demo-1",
          query: "What is RAG?",
          expected_keywords: ["retrieval", "generation"],
          expected_sources: ["doc-a.md"],
        },
      ],
      null,
      2,
    ),
  );
  const [pending, setPending] = useState(false);
  const [report, setReport] = useState<EvalRunReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadSample = async () => {
    setError(null);
    try {
      const sample = await api.evalSample();
      setDatasetText(JSON.stringify(sample.cases, null, 2));
    } catch (err) {
      setError(String(err));
    }
  };

  const runEval = async () => {
    setError(null);
    setReport(null);
    let cases: EvalCaseInput[];
    try {
      const parsed = JSON.parse(datasetText);
      if (!Array.isArray(parsed) || parsed.length === 0) {
        throw new Error("Dataset must be a non-empty JSON array of cases.");
      }
      cases = parsed as EvalCaseInput[];
    } catch (e) {
      setError(`Dataset JSON is invalid: ${(e as Error).message}`);
      return;
    }

    setPending(true);
    try {
      const result = await api.evalRun({ cases, mode });
      setReport(result);
    } catch (err) {
      setError(
        err instanceof ApiError ? `${err.status}: ${err.message}` : String(err),
      );
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="col" style={{ maxWidth: 1100 }}>
      <header>
        <h2 style={{ margin: 0 }}>Evaluation pipeline</h2>
        <p className="muted">
          Score the agents on a labelled dataset. Intrinsic metrics
          (BLEU-like, ROUGE-L, perplexity proxy) and extrinsic ones
          (task completion, faithfulness, retrieval@k, guardrail
          pass-rate) are computed offline.
        </p>
      </header>

      <div className="card card--padded col">
        <div className="row">
          <label className="faint mono" style={{ fontSize: 12 }}>
            Mode
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as ChatMode)}
              style={{ marginLeft: 8 }}
            >
              <option value="multi">Multi-agent</option>
              <option value="single">Single-agent</option>
            </select>
          </label>
          <button className="btn btn--ghost" onClick={loadSample}>
            Load sample dataset
          </button>
          <button className="btn" onClick={runEval} disabled={pending}>
            {pending ? "Running…" : "Run evaluation"}
          </button>
        </div>
        <label>
          <div className="section-title">Dataset (JSON array of cases)</div>
          <textarea
            value={datasetText}
            onChange={(e) => setDatasetText(e.target.value)}
            rows={12}
            className="mono"
            spellCheck={false}
          />
        </label>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {report && <ReportView report={report} />}
    </div>
  );
}

function ReportView({ report }: { report: EvalRunReport }) {
  const metricNames = Object.keys(report.aggregate.means);

  return (
    <div className="col">
      <div className="card card--padded">
        <h3 style={{ margin: "0 0 6px 0" }}>Run summary</h3>
        <p className="faint mono">{report.run_id} · mode {report.mode}</p>
        <dl className="kv" style={{ marginTop: 8 }}>
          <dt>Cases</dt>
          <dd className="mono">{report.aggregate.n_cases}</dd>
          <dt>Errors</dt>
          <dd className="mono">{report.aggregate.n_errors}</dd>
          <dt>Avg latency</dt>
          <dd className="mono">{report.aggregate.avg_latency_ms.toFixed(1)} ms</dd>
          <dt>Avg tokens</dt>
          <dd className="mono">{report.aggregate.avg_tokens.toFixed(1)}</dd>
          <dt>Guardrail pass-rate</dt>
          <dd className="mono">
            {(report.aggregate.guardrail_pass_rate * 100).toFixed(1)}%
          </dd>
        </dl>

        {metricNames.length > 0 && (
          <>
            <div className="divider" />
            <h4 style={{ margin: "0 0 8px 0" }}>Aggregate metrics</h4>
            <table className="table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="num">Mean</th>
                  <th className="num">Cases scored</th>
                </tr>
              </thead>
              <tbody>
                {metricNames.map((name) => (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td className="num mono">
                      {report.aggregate.means[name].toFixed(3)}
                    </td>
                    <td className="num mono">
                      {report.aggregate.counts[name]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      <div className="card card--padded">
        <h3 style={{ margin: "0 0 8px 0" }}>Per-case results</h3>
        <table className="table">
          <thead>
            <tr>
              <th>Case</th>
              <th>Query</th>
              <th>Guardrail</th>
              {metricNames.map((m) => (
                <th key={m} className="num mono">{m}</th>
              ))}
              <th className="num">Latency</th>
              <th className="num">Tokens</th>
            </tr>
          </thead>
          <tbody>
            {report.cases.map((c, i) => (
              <tr key={i}>
                <td className="mono">{c.case_id ?? `#${i + 1}`}</td>
                <td>
                  <div>{c.query}</div>
                  {c.error && (
                    <div className="badge badge--block" style={{ marginTop: 4 }}>
                      error: {c.error}
                    </div>
                  )}
                </td>
                <td>
                  {c.guardrail_verdict ? (
                    <span
                      className={
                        c.guardrail_verdict === "pass"
                          ? "badge badge--pass"
                          : c.guardrail_verdict === "warn"
                            ? "badge badge--warn"
                            : "badge badge--block"
                      }
                    >
                      {c.guardrail_verdict}
                    </span>
                  ) : (
                    <span className="faint">—</span>
                  )}
                </td>
                {metricNames.map((m) => (
                  <td key={m} className="num mono">
                    {c.metrics[m] !== undefined ? c.metrics[m].toFixed(3) : "—"}
                  </td>
                ))}
                <td className="num mono">{c.latency_ms}</td>
                <td className="num mono">{c.tokens_used}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
