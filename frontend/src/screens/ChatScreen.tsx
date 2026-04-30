import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import { AgentTraceView } from "../components/AgentTraceView";
import { VerdictBadge } from "../components/Badge";
import type {
  ChatMessage,
  ChatMode,
  ChatResponse,
} from "../types/api";

interface DisplayMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  response?: ChatResponse;
}

const newId = () =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;

export function ChatScreen() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<ChatMode>("multi");
  const [topK, setTopK] = useState(5);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-grow textarea
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [draft]);

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  const sendMessage = async () => {
    const query = draft.trim();
    if (!query || pending) return;
    setError(null);

    const userMsg: DisplayMessage = {
      id: newId(),
      role: "user",
      content: query,
    };
    setMessages((m) => [...m, userMsg]);
    setDraft("");
    setPending(true);

    const history: ChatMessage[] = messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({
        role: m.role as "user" | "assistant",
        content: m.content,
      }));

    try {
      const response = await api.chat({
        query,
        mode,
        top_k: topK,
        history,
      });
      setMessages((m) => [
        ...m,
        {
          id: response.request_id,
          role: "assistant",
          content: response.answer,
          response,
        },
      ]);
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? `${err.status ? err.status + ": " : ""}${err.message}`
          : String(err);
      setError(msg);
    } finally {
      setPending(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      sendMessage();
    }
    if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      // Plain Enter sends; Shift+Enter inserts newline.
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat">
      <div className="chat__list">
        <div className="chat__inner">
          {messages.length === 0 && !pending && (
            <div className="chat__empty">
              <div className="card chat__empty-card">
                <div className="chat__empty-icon">✦</div>
                <h3>Ask the RAG Crew</h3>
                <p>
                  Three specialised agents — Researcher, Analyst, and Critic —
                  ground every answer in your ingested documents and run it
                  through a Guardrail Monitor before it reaches you.
                </p>
                <p className="faint" style={{ fontSize: 12.5, marginTop: 12 }}>
                  Press <span className="kbd">Enter</span> to send,
                  <span className="kbd" style={{ marginLeft: 6 }}>
                    Shift + Enter
                  </span>{" "}
                  for a new line.
                </p>
              </div>
            </div>
          )}

          {messages.map((m) =>
            m.role === "user" ? (
              <UserMessage key={m.id} content={m.content} />
            ) : (
              <AssistantMessage key={m.id} message={m} />
            ),
          )}

          {pending && <PendingMessage mode={mode} />}
          <div ref={listEndRef} />
        </div>
      </div>

      <div className="composer">
        <div className="composer__inner">
          {error && (
            <div className="error-banner" style={{ marginBottom: 12 }}>
              ⚠ {error}
            </div>
          )}
          <div className="composer__surface">
            <textarea
              ref={textareaRef}
              className="composer__textarea"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={
                mode === "multi"
                  ? "Ask anything — Researcher → Analyst → Critic will answer."
                  : "Ask anything — the Researcher will answer."
              }
              rows={1}
              disabled={pending}
            />
            <div className="composer__controls">
              <div className="composer__row">
                <select
                  className="composer__select"
                  value={mode}
                  onChange={(e) => setMode(e.target.value as ChatMode)}
                  aria-label="Pipeline mode"
                  title="Pipeline mode"
                >
                  <option value="multi">Multi-agent</option>
                  <option value="single">Single-agent</option>
                </select>
                <input
                  className="composer__topk mono"
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(e) =>
                    setTopK(
                      Math.max(
                        1,
                        Math.min(20, Number(e.target.value) || 5),
                      ),
                    )
                  }
                  aria-label="top_k"
                  title="top_k retrieval"
                />
                <button
                  className="composer__send"
                  onClick={sendMessage}
                  disabled={pending || !draft.trim()}
                  aria-label="Send message"
                  title="Send"
                >
                  {pending ? <span className="typing-dots-mini">…</span> : "↑"}
                </button>
              </div>
            </div>
          </div>
          <div className="composer__hint">
            <span className="kbd">Enter</span> to send ·
            <span className="kbd" style={{ marginLeft: 6 }}>
              Shift + Enter
            </span>{" "}
            for newline · grounded answers from your indexed docs
          </div>
        </div>
      </div>
    </div>
  );
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="msg msg--user" data-testid="chat-bubble-user">
      <div className="msg__avatar msg__avatar--user" aria-hidden>
        You
      </div>
      <div className="msg__bubble">
        <div className="msg__text">{content}</div>
      </div>
    </div>
  );
}

function PendingMessage({ mode }: { mode: ChatMode }) {
  return (
    <div className="msg">
      <div className="msg__avatar msg__avatar--assistant" aria-hidden>
        ✦
      </div>
      <div className="msg__bubble">
        <div className="msg__role">Assistant</div>
        <div className="muted" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="typing-dots" aria-label="thinking">
            <span /> <span /> <span />
          </span>
          {mode === "multi" ? "Researcher → Analyst → Critic" : "Researcher"}
        </div>
      </div>
    </div>
  );
}

function AssistantMessage({ message }: { message: DisplayMessage }) {
  const r = message.response;
  const finalReport =
    r?.traces.find((t) => t.agent_role === "critic")?.guardrail_report ??
    r?.traces.find((t) => t.agent_role === "analyst")?.guardrail_report ??
    r?.traces[0]?.guardrail_report ??
    null;

  const isRedacted =
    !!finalReport?.redacted_text &&
    finalReport.redacted_text === message.content;

  const isBlocked =
    !!finalReport && finalReport.verdict === "block" && !isRedacted;

  return (
    <div className="msg" data-testid="chat-bubble-assistant">
      <div className="msg__avatar msg__avatar--assistant" aria-hidden>
        ✦
      </div>
      <div className="msg__bubble">
        <div className="msg__role">Assistant</div>
        <div className="msg__text">{message.content}</div>

        <div className="msg__meta">
          {finalReport && <VerdictBadge verdict={finalReport.verdict} />}
          {isRedacted && <span className="badge badge--warn">PII redacted</span>}
          {isBlocked && (
            <span className="badge badge--block">Answer withheld</span>
          )}
          {r && (
            <span className="faint mono" style={{ fontSize: 11.5 }}>
              {r.total_latency_ms} ms · {r.total_tokens} tokens ·{" "}
              {r.sources.length} sources
            </span>
          )}
        </div>

        {r && r.sources.length > 0 && (
          <div className="sources">
            {r.sources.map((s, i) => (
              <div key={i} className="source-pill">
                <span className="mono faint">{s.source ?? "unknown"}</span>
                <span className="faint">{s.score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}

        {r && <AgentTraceView traces={r.traces} />}
      </div>
    </div>
  );
}
