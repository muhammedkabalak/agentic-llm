import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AgentTraceView } from "./AgentTraceView";
import type { AgentTrace } from "../types/api";

const baseTrace = (overrides: Partial<AgentTrace> = {}): AgentTrace => ({
  agent_role: "researcher",
  input: "What is RAG?",
  output: "Retrieval-augmented generation.",
  retrieved_chunks: [],
  guardrail_report: {
    verdict: "pass",
    flags: [],
    notes: null,
    findings: [],
    monitor_verdict: "pass",
    redacted_text: null,
  },
  latency_ms: 12,
  tokens_used: 33,
  ...overrides,
});

describe("AgentTraceView", () => {
  it("renders nothing when there are no traces", () => {
    const { container } = render(<AgentTraceView traces={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("hides trace steps until the toggle is clicked", () => {
    render(<AgentTraceView traces={[baseTrace()]} />);
    expect(screen.queryByText(/Retrieval-augmented/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Agent traces/i }));
    expect(screen.getByText(/Retrieval-augmented/)).toBeInTheDocument();
  });

  it("renders verdict and flags from the guardrail report", () => {
    const trace = baseTrace({
      agent_role: "analyst",
      guardrail_report: {
        verdict: "block",
        flags: ["pii"],
        notes: "Email leaked.",
        findings: [
          {
            category: "pii",
            severity: "block",
            message: "Email address detected.",
            evidence: "x@y.com",
            detector: "pii.email",
          },
        ],
        monitor_verdict: "block",
        redacted_text: "Reach out at [REDACTED:EMAIL].",
      },
    });
    render(<AgentTraceView traces={[trace]} />);
    fireEvent.click(screen.getByRole("button", { name: /Agent traces/i }));

    expect(screen.getByText("Blocked")).toBeInTheDocument();
    expect(screen.getAllByText("pii").length).toBeGreaterThan(0);
    expect(screen.getByText("pii.email")).toBeInTheDocument();
    expect(screen.getByText(/Email address detected/)).toBeInTheDocument();
    expect(screen.getByText("Analyst")).toBeInTheDocument();
  });

  it("shows latency and token metadata per step", () => {
    render(<AgentTraceView traces={[baseTrace({ latency_ms: 187, tokens_used: 256 })]} />);
    fireEvent.click(screen.getByRole("button", { name: /Agent traces/i }));
    expect(screen.getByText(/187 ms · 256 tok/)).toBeInTheDocument();
  });

  it("renders all three agent steps in order", () => {
    const traces = [
      baseTrace({ agent_role: "researcher", output: "draft" }),
      baseTrace({ agent_role: "analyst", output: "synthesis" }),
      baseTrace({ agent_role: "critic", output: "review" }),
    ];
    render(<AgentTraceView traces={traces} />);
    fireEvent.click(screen.getByRole("button", { name: /Agent traces \(3\)/i }));
    const roleHeadings = screen.getAllByText(/Researcher|Analyst|Critic/);
    const labels = roleHeadings.map((el) => el.textContent);
    expect(labels).toEqual(
      expect.arrayContaining(["Researcher", "Analyst", "Critic"]),
    );
  });
});
