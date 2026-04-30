import { useEffect, useState } from "react";
import { api } from "./api/client";
import { ChatScreen } from "./screens/ChatScreen";
import { EvalScreen } from "./screens/EvalScreen";
import { IngestScreen } from "./screens/IngestScreen";
import type { HealthResponse } from "./types/api";

type Screen = "chat" | "ingest" | "eval";

const NAV: { id: Screen; label: string; description: string }[] = [
  { id: "chat", label: "Chat", description: "Talk to the agents" },
  { id: "ingest", label: "Ingest", description: "Add docs to the index" },
  { id: "eval", label: "Evaluation", description: "Score the system" },
];

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "light";
  const stored = window.localStorage?.getItem("rag-theme");
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function App() {
  const [screen, setScreen] = useState<Screen>("chat");
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      window.localStorage?.setItem("rag-theme", theme);
    } catch {
      // ignore (private mode etc.)
    }
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((h) => {
        if (!cancelled) setHealth(h);
      })
      .catch(() => {
        if (!cancelled) setHealthError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app">
      <aside className="app__sidebar">
        <div className="app__brand">
          <div className="app__brand-mark" aria-hidden>✦</div>
          <div>
            RAG Crew
            <small>Multi-Agent</small>
          </div>
        </div>
        {NAV.map((n) => (
          <button
            key={n.id}
            className="nav-button"
            aria-current={screen === n.id ? "page" : undefined}
            onClick={() => setScreen(n.id)}
          >
            <span className="nav-button__dot" aria-hidden />
            <span>
              <div>{n.label}</div>
              <div className="faint" style={{ fontSize: 11.5, marginTop: 1 }}>
                {n.description}
              </div>
            </span>
          </button>
        ))}
      </aside>

      <header className="app__header">
        <div>
          <div className="app__title">{currentTitle(screen)}</div>
          <div className="app__subtitle">{currentSubtitle(screen)}</div>
        </div>
        <div className="row">
          <HealthIndicator health={health} error={healthError} />
          <button
            className="theme-toggle"
            onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          >
            {theme === "light" ? "☾" : "☀"}
          </button>
        </div>
      </header>

      <main className="app__main">
        {screen === "chat" && <ChatScreen />}
        {screen === "ingest" && (
          <div style={{ padding: "var(--space-7)" }}>
            <IngestScreen />
          </div>
        )}
        {screen === "eval" && (
          <div style={{ padding: "var(--space-7)" }}>
            <EvalScreen />
          </div>
        )}
      </main>
    </div>
  );
}

function currentTitle(screen: Screen): string {
  switch (screen) {
    case "chat":
      return "Chat";
    case "ingest":
      return "Ingest documents";
    case "eval":
      return "Evaluation pipeline";
  }
}

function currentSubtitle(screen: Screen): string {
  switch (screen) {
    case "chat":
      return "Researcher → Analyst → Critic, with the Guardrail Monitor in front of every output.";
    case "ingest":
      return "Chunked, embedded, and indexed in the same store /chat retrieves from.";
    case "eval":
      return "Intrinsic + extrinsic metrics computed offline against a labelled dataset.";
  }
}

function HealthIndicator({
  health,
  error,
}: {
  health: HealthResponse | null;
  error: boolean;
}) {
  if (error) {
    return <span className="badge badge--block">Backend offline</span>;
  }
  if (!health) {
    return <span className="badge badge--neutral">Pinging…</span>;
  }
  return (
    <span className="badge badge--pass">
      {health.app_name} v{health.version}
    </span>
  );
}
