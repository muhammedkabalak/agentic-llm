import type { GuardrailVerdict, AgentRole } from "../types/api";

const VERDICT_LABELS: Record<GuardrailVerdict, string> = {
  pass: "Pass",
  warn: "Warn",
  block: "Blocked",
};

export function VerdictBadge({ verdict }: { verdict: GuardrailVerdict }) {
  const cls =
    verdict === "pass"
      ? "badge badge--pass"
      : verdict === "warn"
        ? "badge badge--warn"
        : "badge badge--block";
  return <span className={cls}>{VERDICT_LABELS[verdict]}</span>;
}

export function FlagBadge({ flag }: { flag: string }) {
  return <span className="badge badge--neutral mono">{flag}</span>;
}

const ROLE_LABELS: Record<AgentRole, string> = {
  researcher: "Researcher",
  analyst: "Analyst",
  critic: "Critic",
  orchestrator: "Orchestrator",
};

export function RoleBadge({ role }: { role: AgentRole }) {
  return <span className="badge badge--accent">{ROLE_LABELS[role]}</span>;
}
