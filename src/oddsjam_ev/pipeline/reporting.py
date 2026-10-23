"""Deterministic private Markdown; operational status never implies strategy validity."""

import json


def render_report(spec: dict, summaries: dict) -> str:
    """Render aggregate evidence only; no source rows or arbitrary error messages."""
    parts = [
        "# Local research report",
        f"Mode: **{spec['mode']}**. Knowledge cutoff: `{spec['as_of']}`.",
        "Scheduling is disabled. This is research output, not betting advice or authorization.",
        "## Data status",
        "```json\n" + json.dumps(spec["freshness"], indent=2, sort_keys=True) + "\n```",
        "## Source schema and coverage diagnostics",
        "```json\n" + json.dumps(spec["source_diagnostics"], indent=2, sort_keys=True) + "\n```",
        "Receipt recency does not establish export recency or complete coverage. "
        "Historical reanalysis does not constitute new observations.",
    ]
    for name, summary in summaries.items():
        parts += [
            "## " + name.replace("_", " ").title(),
            "```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```",
        ]
    parts += [
        "## Interpretation limits",
        "Real-money, paper and unknown populations remain separate. Overlapping views and "
        "filter attribution are not additive. Opportunity identities are proxies. "
        "Unknown source timezone, chronology and scope remain unknown. Reported profit is "
        "preserved; fee inclusion is unverified. Core Exchange history reflects selectively "
        "chosen trades, not the scanner universe. Exchange workflow is taker-only by user "
        "assumption, not verified fills. Unchecked paper outcomes remain hypothetical. "
        "Canonical product and settlement exclusions still apply. No approvals, enrollments, "
        "launches, filter edits or promotions are performed by this pipeline.",
    ]
    return "\n\n".join(parts) + "\n"
