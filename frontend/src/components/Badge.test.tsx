import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FlagBadge, RoleBadge, VerdictBadge } from "./Badge";

describe("Badge primitives", () => {
  it("maps verdicts to readable labels", () => {
    render(<VerdictBadge verdict="pass" />);
    render(<VerdictBadge verdict="warn" />);
    render(<VerdictBadge verdict="block" />);
    expect(screen.getByText("Pass")).toBeInTheDocument();
    expect(screen.getByText("Warn")).toBeInTheDocument();
    expect(screen.getByText("Blocked")).toBeInTheDocument();
  });

  it("renders raw flag text", () => {
    render(<FlagBadge flag="missing_citation" />);
    expect(screen.getByText("missing_citation")).toBeInTheDocument();
  });

  it("renders agent role with capitalised label", () => {
    render(<RoleBadge role="critic" />);
    expect(screen.getByText("Critic")).toBeInTheDocument();
  });
});
