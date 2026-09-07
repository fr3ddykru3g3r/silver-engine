"""Render a plain-English, judge-facing summary of the frozen fallback experiment.

This renderer does not recompute, select, or improve any scientific result. It only
translates fields already present in summary.json into a compact Markdown view.
Development-only and locked-test boundaries are carried into the output explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


STATE_LABELS = {
    "NO_XRS": "X-ray feed unavailable",
    "NO_PROTON": "proton-context feed unavailable",
    "NO_XRS_OR_PROTON": "X-ray + proton-context feeds unavailable",
}
STATE_TO_MODALITY = {
    "NO_XRS": "XRS",
    "NO_PROTON": "PROTON",
    "NO_XRS_OR_PROTON": "XRS_AND_PROTON",
}
DURATIONS = (24, 72, 168)


def _metric(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def validate_summary(summary: dict) -> None:
    if summary.get("locked_test_accessed") is not False:
        raise ValueError("judge summary requires locked_test_accessed=false")
    if summary.get("monitor_used") is not False:
        raise ValueError("judge summary requires monitor_used=false")
    if summary.get("imputation_used") is not False:
        raise ValueError("judge summary requires imputation_used=false")
    if summary.get("reconstruction_used") is not False:
        raise ValueError("judge summary requires reconstruction_used=false")
    if summary.get("retraining_at_outage_time") is not False:
        raise ValueError("judge summary requires retraining_at_outage_time=false")

    decisions = summary.get("operator_state_decision")
    scenarios = summary.get("scenarios")
    states = summary.get("states")
    if not isinstance(decisions, dict) or not isinstance(scenarios, dict) or not isinstance(states, dict):
        raise ValueError("judge summary requires states, scenarios, and operator_state_decision")
    for state, modality in STATE_TO_MODALITY.items():
        if state not in decisions:
            raise ValueError(f"missing operator decision for {state}")
        permission = decisions[state].get("permission")
        if permission not in {"DEGRADED", "ABSTAIN"}:
            raise ValueError(f"unexpected permission for {state}: {permission}")
        if decisions[state].get("normal_allowed") is not False:
            raise ValueError(f"NORMAL must remain forbidden for {state}")
        for hours in DURATIONS:
            key = f"{modality}_{hours}H"
            scenario = scenarios.get(key)
            if not isinstance(scenario, dict):
                raise ValueError(f"missing scenario {key}")
            if scenario.get("availability_state") != state:
                raise ValueError(f"scenario {key} has wrong availability state")
            gate = scenario.get("operator_gate")
            if not isinstance(gate, dict) or "passed" not in gate:
                raise ValueError(f"scenario {key} is missing operator gate")


def render_markdown(summary: dict) -> str:
    validate_summary(summary)

    full = summary["states"].get("FULL", {})
    full_max_tss = full.get("whole_score", {}).get("MAX_TSS", {})
    rows = int(full_max_tss.get("rows", 0))
    positives = int(full_max_tss.get("positives", 0))

    lines = [
        "# IRIS-SEP judge summary — development-only evidence",
        "",
        "## The question in one sentence",
        "",
        "When a solar-radiation warning system loses an input feed, is it safer to switch to a model trained for the sensors that remain, or to refuse to issue a normal forecast?",
        "",
        "## Why this experiment matters",
        "",
        "A forecast can still output a plausible-looking probability after a sensor failure. This experiment tests whether that number remains decision-worthy instead of assuming that a finite probability is automatically trustworthy.",
        "",
        "## What we did",
        "",
        "We froze separate fallback models for each sensor-availability state before simulating 24-hour, 72-hour, and 168-hour event-bearing outages. Missing measurements were not filled in, reconstructed, or used to retrain the model during the outage. Each fallback had to pass the same predeclared safety gate at all three outage durations to earn even a DEGRADED permission; otherwise the system must ABSTAIN.",
        "",
        "## Result a non-specialist judge can read",
        "",
        "| Missing information | 24 h gate | 72 h gate | 168 h gate | Final permission | Normal forecast allowed? |",
        "|---|---:|---:|---:|---|---:|",
    ]

    for state, label in STATE_LABELS.items():
        modality = STATE_TO_MODALITY[state]
        gates = []
        for hours in DURATIONS:
            passed = bool(summary["scenarios"][f"{modality}_{hours}H"]["operator_gate"]["passed"])
            gates.append("PASS" if passed else "FAIL")
        decision = summary["operator_state_decision"][state]
        lines.append(
            f"| {label} | {gates[0]} | {gates[1]} | {gates[2]} | {decision['permission']} | {_yes_no(decision['normal_allowed'])} |"
        )

    lines += [
        "",
        "## Clean-data reference",
        "",
        f"The inspected development score block contains {rows} forecast rows and {positives} positive rows. Under the frozen MAX_TSS policy, the full-input reference has TSS {_metric(full_max_tss.get('TSS'))}, POD {_metric(full_max_tss.get('POD'))}, FAR {_metric(full_max_tss.get('FAR'))}, Brier score {_metric(full_max_tss.get('BRIER'))}, and ECE {_metric(full_max_tss.get('ECE'))}.",
        "",
        "These numbers are included so a judge sees both skill and failure cost. A high false-alarm rate is not hidden by the fallback analysis.",
        "",
        "## Integrity boundary",
        "",
        f"- Locked test accessed: **{_yes_no(summary['locked_test_accessed'])}**",
        f"- Previously inspected monitor used: **{_yes_no(summary['monitor_used'])}**",
        f"- Missing values imputed: **{_yes_no(summary['imputation_used'])}**",
        f"- Missing values reconstructed: **{_yes_no(summary['reconstruction_used'])}**",
        f"- Model retrained when the outage occurred: **{_yes_no(summary['retraining_at_outage_time'])}**",
        "",
        "## Claim boundary",
        "",
        "This is development-only evidence. It can support the design decision to use availability-conditioned fallback or abstention, but it cannot establish operational superiority, real-world deployment readiness, or final performance until the independent locked evaluation is opened under the frozen protocol.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
