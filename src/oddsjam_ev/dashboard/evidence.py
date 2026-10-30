"""
evidence.py
===============================================================================
Purpose
-------
Read explicitly linked within-filter evidence without rebuilding research.

What this module provides
-------------------------
Root confinement, selected-report lineage checks and display-only selection.

Design Notes
------------
Every display re-verifies bytes. Missing links never trigger latest-file discovery.
===============================================================================
"""

import json
from pathlib import Path

from oddsjam_ev.dashboard.artifacts import _digest, _require, within
from oddsjam_ev.io.signal_prioritization import load_prioritization
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery


def load_evidence(run, cfg):
    """Verify evidence pinned to the selected report, or return an unavailable state.

    Parameters
    ----------
    run : PublishedRun
        Verified selected dashboard report.
    cfg : DashboardConfig
        Trusted root and explicit run-ID publication links.

    Returns
    -------
    dict or None
        Verified tables and manifests; None means no configured attachment.
    """
    links = cfg.evidence_publications.get(run.result["run_id"])
    if links is None:
        return None
    root = Path(cfg.evidence_root).resolve()
    paths = {k: within(root, root / v) for k, v in links.items()}
    manifests = {}
    for name, path in paths.items():
        manifests[name] = json.loads(within(root, path / "manifest.json").read_text())
        for file in manifests[name]["output_sha256"]:
            _require(Path(file).name == file)
            within(root, path / file)
    d, p = (manifests[k]["specification"] for k in ("discovery", "prioritization"))
    _require(d["canonical_bundle_id"] == run.manifests["canonical"]["bundle_id"])
    _require(d["atlas_id"] == run.manifests["atlas"]["atlas_id"])
    for stage in ("canonical", "atlas"):
        _require(
            d["input_manifest_sha256"][stage] == run.result["outputs"][stage]["manifest_sha256"]
        )
    _require(p["discovery_publication_id"] == manifests["discovery"]["publication_id"])
    _require(p["discovery_manifest_sha256"] == _digest(paths["discovery"] / "manifest.json"))
    _require(p["diagnostics_id"] == d["diagnostics_id"])
    _require(p["diagnostics_manifest_sha256"] == d["input_manifest_sha256"]["diagnostics"])
    family = load_within_filter_discovery(paths["discovery"])
    priority = load_prioritization(paths["prioritization"])
    candidates = family["signal_candidates"]
    stats = priority["signal_statistics"]
    _require(set(candidates.candidate_id) == set(stats.candidate_id))
    _require(not candidates.candidate_id.duplicated().any())
    _require(not stats.candidate_id.duplicated().any())
    return {"family": family, "priority": priority, "manifests": manifests}


def candidate_rows(evidence, mode="all", product="all", minimum=100):
    """Select persisted rows without changing testing families or recomputing metrics.

    Parameters
    ----------
    evidence : dict
        Verified optional evidence tables.
    mode, product : str
        Funding and product selectors; all leaves populations separate.
    minimum : int
        Display-only minimum eligible subgroup wagers.

    Returns
    -------
    pandas.DataFrame
        Exact definitions with published statistics and presentation decisions.
    """
    rows = evidence["family"]["signal_candidates"].merge(
        evidence["priority"]["signal_statistics"],
        on=["candidate_id", "parent_id", "filter_id", "execution_status", "product"],
        validate="one_to_one",
    )
    decisions = evidence["priority"]["signal_priorities"]
    rows = rows.merge(
        decisions[
            [
                "candidate_id",
                "category",
                "included_shortlist",
                "suppression_reason",
                "presentation_order",
            ]
        ],
        on="candidate_id",
        validate="one_to_one",
    ).sort_values("presentation_order")
    if mode != "all":
        rows = rows.loc[rows.execution_status.eq(mode)]
    if product != "all":
        rows = rows.loc[rows["product"].eq(product)]
    return rows.loc[rows.subgroup_eligible_count.ge(minimum)].copy()


def load_paper_definitions(run, cfg, evidence):
    """Read a pinned definition snapshot and retain exact selected-evidence versions.

    Parameters
    ----------
    run : PublishedRun
        Selected verified research report.
    cfg : DashboardConfig
        Trusted optional publication path under evidence_root.
    evidence : dict or None
        Verified discovery and prioritization attachment.

    Returns
    -------
    dict or None
        Matching versions and publication metadata. No live register is read.
    """
    from oddsjam_ev.io.paper_definitions import load_definition_publication

    if cfg.paper_definitions_publication is None:
        return None
    root = Path(cfg.evidence_root).resolve()
    path = within(root, root / cfg.paper_definitions_publication)
    within(root, path / "manifest.json")
    within(root, path / "definitions.json")
    publication = load_definition_publication(path)
    if evidence is None:
        return {"manifest": publication["manifest"], "events": []}
    dm, pm = (evidence["manifests"][k] for k in ("discovery", "prioritization"))
    events = [
        event
        for event in publication["events"]
        if (
            event["source"]["canonical_bundle_id"] == run.manifests["canonical"]["bundle_id"]
            and event["source"]["discovery_publication_id"] == dm["publication_id"]
            and event["source"]["prioritization_publication_id"] == pm["publication_id"]
            and event["source"]["discovery_manifest_sha256"] == fingerprint_manifest(dm)
            and event["source"]["prioritization_manifest_sha256"] == fingerprint_manifest(pm)
        )
    ]
    # Preserve supersession across report boundaries; an older matching version
    # cannot become current just because its successor references other evidence.
    latest = {e["test_key"]: e["version_id"] for e in publication["events"]}
    return {"manifest": publication["manifest"], "events": events, "latest": latest}


def fingerprint_manifest(manifest):
    """Hash the canonical persisted manifest representation used by publication writers."""
    import hashlib

    return hashlib.sha256(
        (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    ).hexdigest()


def load_decision_sheets(run, cfg, evidence):
    """Verify optional configuration sheets against exact selected report evidence.

    Parameters
    ----------
    run : PublishedRun
        Selected verified dashboard run.
    cfg : DashboardConfig
        Trusted publication path under evidence_root.
    evidence : dict or None
        Verified within-filter evidence for this report.

    Returns
    -------
    dict or None
        Matching immutable records; None means no configured publication.
    """
    from oddsjam_ev.io.decision_records import load_decisions

    if cfg.decision_records_publication is None:
        return None
    root = Path(cfg.evidence_root).resolve()
    path = within(root, root / cfg.decision_records_publication)
    for file in ("manifest.json", "decisions.json", "audit.json"):
        within(root, path / file)
    publication = load_decisions(path)
    spec = publication["manifest"]["specification"]
    if evidence is None:
        return {**publication, "records": []}
    _require(spec["canonical_bundle_id"] == run.manifests["canonical"]["bundle_id"])
    for stage in ("canonical", "atlas"):
        _require(
            spec[stage + "_manifest_sha256"] == run.result["outputs"][stage]["manifest_sha256"]
        )
    for stage in ("discovery", "prioritization"):
        _require(spec[stage + "_publication_id"] == evidence["manifests"][stage]["publication_id"])
        _require(
            spec[stage + "_manifest_sha256"] == fingerprint_manifest(evidence["manifests"][stage])
        )
    return publication
