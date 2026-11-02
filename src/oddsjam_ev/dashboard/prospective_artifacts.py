"""
prospective_artifacts.py
===============================================================================
Purpose
-------
Attach explicit immutable prospective reports to matching frozen definitions.

What this module provides
-------------------------
Trusted-root, byte-verified report loading without consulting live registers.

Design Notes
------------
Prospective outcome snapshots may be newer than historical discovery snapshots.
They are joined by exact definition identity, never by a similar filter name.
===============================================================================
"""

import json
from pathlib import Path

from oddsjam_ev.dashboard.artifacts import within
from oddsjam_ev.io.experiment_reports import load_experiment_report


def load_prospective_report(cfg, definitions):
    """Load verified version-specific results for the selected definition snapshot.

    Parameters
    ----------
    cfg : DashboardConfig
        Trusted optional report path, relative to the evidence root.
    definitions : dict or None
        Already verified definitions matching this research report.

    Returns
    -------
    dict or None
        Manifest and matching tables. Every byte is verified on each load.
    """
    if cfg.prospective_report_publication is None:
        return None
    root = Path(cfg.evidence_root).resolve()
    path = within(root, root / cfg.prospective_report_publication)
    manifest = json.loads(within(root, path / "manifest.json").read_text())
    for file in manifest["output_sha256"]:
        if Path(file).name != file:
            raise ValueError("Invalid report filename")
        within(root, path / file)
    tables = load_experiment_report(path)
    allowed = {e["version_id"]: e for e in (definitions or {}).get("events", [])}
    registrations = [
        e
        for e in manifest["specification"]["events"]
        if e["kind"] == "REGISTER" and e["payload"]["protocol"].get("contract_version") == 2
    ]
    ids = set()
    for event in registrations:
        d = event["payload"]["protocol"]["definition"]
        if d["version_id"] in allowed:
            if d != allowed[d["version_id"]]:
                raise ValueError("Definition/report identity mismatch")
            ids.add(event["experiment_id"])
    result = {
        name: table.loc[table.experiment_id.isin(ids)].copy()
        for name, table in tables.items()
        if name.startswith("prospective_")
    }
    return {"manifest": manifest, "tables": result}
