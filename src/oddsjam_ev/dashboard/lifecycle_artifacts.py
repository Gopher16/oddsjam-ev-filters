"""
lifecycle_artifacts.py
===============================================================================
Purpose
-------
Attach explicit immutable lifecycle evidence to exact selected experiment definitions.

What this module provides
-------------------------
Trusted-root verification and version-filtered decision/configuration/monitoring views.

Design Notes
------------
No live journal, external scanner or automatic latest publication is read.
===============================================================================
"""

import json
from pathlib import Path

from oddsjam_ev.analysis.filter_lifecycle import replay_lifecycle
from oddsjam_ev.dashboard.artifacts import within
from oddsjam_ev.io.filter_lifecycle import load_lifecycle_report


def load_lifecycle_attachment(cfg, definitions):
    """Verify an explicitly attached publication and retain exact definition lineage.

    Parameters
    ----------
    cfg : DashboardConfig
        Trusted optional publication path under the evidence root.
    definitions : dict or None
        Verified frozen definitions matching the selected historical report.

    Returns
    -------
    dict or None
        Filtered decision/application evidence, with no fallback or implicit status.
    """
    if cfg.lifecycle_report_publication is None:
        return None
    root = Path(cfg.evidence_root).resolve()
    path = within(root, root / cfg.lifecycle_report_publication)
    manifest = json.loads(within(root, path / "manifest.json").read_text())
    for file in manifest["output_sha256"]:
        if Path(file).name != file:
            raise ValueError("Invalid lifecycle publication filename")
        within(root, path / file)
    report = load_lifecycle_report(path)
    allowed = {e["version_id"]: e for e in (definitions or {}).get("events", [])}
    state = replay_lifecycle(report["events"])
    decisions = [
        e
        for e in state["decisions"].values()
        if e["payload"]["source"]["definition_version_id"] in allowed
    ]
    for e in decisions:
        source = e["payload"]["source"]
        if source["definition"] != allowed[source["definition_version_id"]]:
            raise ValueError("Lifecycle definition lineage mismatch")
    ids = {e["event_id"] for e in decisions}
    changes = {
        key: v
        for key, v in state["changes"].items()
        if v["event"]["payload"]["decision_event_id"] in ids
    }
    filters = {c["event"]["payload"]["filter_id"] for c in changes.values()}
    applications = report["tables"]["applications"].loc[lambda f: f.filter_id.isin(filters)]
    app_ids = set(applications.application_id)
    tables = {
        name: frame.loc[frame.application_id.isin(app_ids)].copy()
        for name, frame in report["tables"].items()
    }
    return {
        "manifest": report["manifest"],
        "decisions": decisions,
        "changes": changes,
        "evaluations": [
            e
            for e in state["evaluations"].values()
            if any(
                r["definition_version_id"] in allowed for r in e["payload"]["assessment"]["results"]
            )
        ],
        "tables": tables,
    }
