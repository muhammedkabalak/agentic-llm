import { useState } from "react";
import type { AgentTrace, GuardrailFinding } from "../types/api";
import { FlagBadge, RoleBadge, VerdictBadge } from "./Badge";

export function AgentTraceView({ traces }: { traces: AgentTrace[] }) {
  const [open, setOpen] = useState(false);
  if (traces.length === 0) return null;

  return (
    <div className="trace">
      <button
        className="trace__toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="trace__chevron" aria-hidden>
          ▸
        </span>
        Agent traces ({traces.length})
      </button>
      {open && (
        <div className="trace__steps">
          {traces.map((t, i) => (
            <TraceStep key={`${t.agent_role}-${i}`} trace={t} index={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

function TraceStep({ trace, index }: { trace: AgentTrace; index: number }) {
  const r = trace.guardrail_report;
  return (
    <article className="trace__step">
      <header>
        <div className="row" style={{ gap: 10 }}>
          <span className="trace__step-index">{index}</span>
          <RoleBadge role={trace.agent_role} />
          {r && <VerdictBadge verdict={r.verdict} />}
          {r?.flags.map((f) => <FlagBadge key={f} flag={f} />)}
        </div>
        <span className="faint mono" style={{ fontSize: 11.5 }}>
          {trace.latency_ms} ms · {trace.tokens_used} tok
        </span>
      </header>

      {r?.notes && <p className="trace__notes">{r.notes}</p>}

      <pre className="trace__output">{trace.output}</pre>

      {r && r.findings.length > 0 && <FindingsList findings={r.findings} />}

      {trace.retrieved_chunks.length > 0 && (
        <details>
          <summary>
            Retrieved context ({trace.retrieved_chunks.length})
          </summary>
          <div className="sources" style={{ marginTop: 8 }}>
            {trace.retrieved_chunks.map((c, j) => (
              <div key={j} className="source-pill">
                <span className="mono faint">{c.source ?? "unknown"}</span>
                <span className="faint">{c.score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </article>
  );
}

function FindingsList({ findings }: { findings: GuardrailFinding[] }) {
  return (
    <div className="findings">
      {findings.map((f, i) => (
        <div key={i} className="finding">
          <span
            className={
              f.severity === "block"
                ? "badge badge--block"
                : f.severity === "warn"
                  ? "badge badge--warn"
                  : "badge badge--neutral"
            }
          >
            {f.category}
          </span>
          <code>{f.detector}</code>
          <span className="muted">
            {f.message}
            {f.evidence ? ` — “${f.evidence}”` : ""}
          </span>
        </div>
      ))}
    </div>
  );
}
