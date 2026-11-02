"""
acceptance.py
===============================================================================
Purpose
-------
Audit explicit dashboard publications without fitting models or changing journals.

What this module provides
-------------------------
Repeatable integrity, reconciliation, chart coverage and loader benchmarks.

Design Notes
------------
This is a publication smoke gate, not browser acceptance or strategy validation.
Missing optional publications are reported, never silently counted as verified.
===============================================================================
"""

import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from oddsjam_ev.dashboard.artifacts import load_published_run
from oddsjam_ev.dashboard.decision_artifacts import load_workspace
from oddsjam_ev.dashboard.evidence import (
    load_decision_sheets,
    load_evidence,
    load_paper_definitions,
)
from oddsjam_ev.dashboard.explanation_artifacts import load_explanations
from oddsjam_ev.dashboard.lifecycle_artifacts import load_lifecycle_attachment
from oddsjam_ev.dashboard.pages.visual_history import CHARTS
from oddsjam_ev.dashboard.prospective_artifacts import load_prospective_report
from oddsjam_ev.dashboard.queries import validate_display_contract


def reconcile_visuals(canonical, visuals):
    """Reconcile unique candidates and population economics without adding overlaps.

    Parameters
    ----------
    canonical : pandas.DataFrame
        Verified wager-grain facts.
    visuals : dict
        Verified visual tables from the same canonical publication.

    Returns
    -------
    dict
        Counts and population reconciliation status. Raises on any mismatch.
    """
    candidates = visuals["visual_candidates"]
    if candidates.bet_id.duplicated().any() or set(candidates.bet_id) != set(canonical.bet_id):
        raise ValueError("Visual/canonical wager reconciliation failed")
    facts = canonical.set_index("bet_id").loc[candidates.bet_id]
    for source, target in (("reported_stake_usd", "stake"), ("reported_profit_usd", "profit")):
        if not np.allclose(
            pd.to_numeric(facts[source], errors="coerce"), candidates[target], equal_nan=True
        ):
            raise ValueError("Visual/canonical economic reconciliation failed")
    overall = visuals["visual_cohorts"].query("scope_kind == 'population' and view == 'overall'")
    groups = candidates.groupby(["execution_status", "product"], dropna=False)
    if len(overall) != groups.ngroups:
        raise ValueError("Population partition reconciliation failed")
    for (funding, product), rows in groups:
        summary = overall.loc[overall.execution_status.eq(funding) & overall["product"].eq(product)]
        if len(summary) != 1:
            raise ValueError("Population partition reconciliation failed")
        summary = summary.iloc[0]
        eligible = rows.loc[rows.diagnostic_eligible]
        if summary.candidate_count != len(rows) or summary.eligible_count != len(eligible):
            raise ValueError("Population denominator reconciliation failed")
        if not eligible.empty:
            expected = [
                eligible.stake.sum(),
                eligible.profit.sum(),
                eligible.profit.sum() / eligible.stake.sum(),
                eligible.unit_return.mean(),
            ]
            actual = (
                summary[
                    [
                        "eligible_stake_usd",
                        "reported_profit_usd",
                        "stake_weighted_roi",
                        "equal_stake_mean_return",
                    ]
                ]
                .astype(float)
                .to_numpy()
            )
            if not np.allclose(expected, actual, rtol=1e-10, atol=1e-10):
                raise ValueError("Population economics reconciliation failed")
    return {
        "canonical_wagers": len(canonical),
        "visual_wagers": len(candidates),
        "eligible_wagers": int(candidates.diagnostic_eligible.sum()),
        "population_economics_reconciled": groups.ngroups,
        "memberships_are_not_independent_wagers": True,
    }


def inspect_release(cfg, run_id, repeats=3):
    """Verify an explicit release surface and report aggregate evidence only.

    Parameters
    ----------
    cfg : DashboardConfig
        Trusted configuration with exact optional publication attachments.
    run_id : str
        Explicit historical report identity; no newest-file selection.
    repeats : int
        One to five byte-verified warm loader repetitions after the cold load.

    Returns
    -------
    dict
        Non-row-level reconciliation, source blockers, attachment availability and
        measured cold/warm seconds plus process peak resident memory. Exceptions
        propagate on corrupt or inconsistent publications; no partial pass report.

    Notes
    -----
    Timings exclude browser rendering. RSS includes imports and is a process high
    water mark, not incremental allocation. Repeats never retrain or republish.
    """
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("Acceptance repeats must be an integer between 1 and 5")
    start = perf_counter()
    run = load_published_run(Path(cfg.artifact_root), run_id)
    validate_display_contract(run.tables)
    evidence = load_evidence(run, cfg)
    decisions = load_decision_sheets(run, cfg, evidence)
    definitions = load_paper_definitions(run, cfg, evidence)
    prospective = load_prospective_report(cfg, definitions)
    lifecycle = load_lifecycle_attachment(cfg, definitions)
    workspace = load_workspace(run, cfg, evidence, decisions)
    models = load_explanations(run, cfg, workspace)
    cold = perf_counter() - start
    attachments = {
        k: "verified" if v is not None else "not_attached"
        for k, v in dict(
            evidence=evidence,
            settings=decisions,
            definitions=definitions,
            prospective=prospective,
            lifecycle=lifecycle,
            workspace=workspace,
            models=models,
        ).items()
    }
    reconciliation, charts, populations = {}, [], []
    if workspace is not None:
        canonical = run.tables["canonical"]["bet_facts"]
        reconciliation = reconcile_visuals(canonical, workspace["visuals"])
        readiness = workspace["visuals"]["visual_readiness"]
        if set(readiness.chart_id) != set(CHARTS.values()):
            raise ValueError("Requested chart inventory is incomplete")
        for chart_id, rows in readiness.groupby("chart_id", sort=True):
            blocked = rows.loc[rows.supported_candidates.eq(0)]
            if blocked.remediation.fillna("").str.strip().eq("").any():
                raise ValueError("Blocked chart has no remediation")
            charts.append(
                {
                    "chart_id": chart_id,
                    "published_scopes": len(rows),
                    "supported_scopes": int(rows.supported_candidates.gt(0).sum()),
                    "statuses": sorted(rows.status.unique().tolist()),
                    "remediation": sorted(set(blocked.remediation.dropna())),
                }
            )
    if models is not None:
        for publication in models["publications"].values():
            rows = publication["predictions"]  # loader already checks additivity/splits/lineage
            populations.append(
                {
                    "publication_id": publication["publication_id"],
                    "kind": publication["kind"],
                    "predicted_wagers": len(rows),
                    "declined_populations": sum(
                        p["status"] == "declined" for p in publication["audit"]["populations"]
                    ),
                }
            )
    warm = []
    for _ in range(repeats):
        start = perf_counter()
        run.verify()
        e = load_evidence(run, cfg)
        d = load_decision_sheets(run, cfg, e)
        defs = load_paper_definitions(run, cfg, e)
        load_prospective_report(cfg, defs)
        load_lifecycle_attachment(cfg, defs)
        w = load_workspace(run, cfg, e, d, workspace)
        load_explanations(run, cfg, w, models)
        warm.append(perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "contract_version": 1,
        "run_id": run_id,
        "status": "verified_publications_not_strategy_or_browser_approval",
        "attachments": attachments,
        "reconciliation": reconciliation,
        "charts": charts,
        "models": populations,
        "performance": {
            "cold_seconds": cold,
            "warm_seconds": warm,
            "warm_median_seconds": float(np.median(warm)),
            "peak_process_rss_mib": rss / (1024**2 if sys.platform == "darwin" else 1024),
            "python": platform.python_version(),
            "platform": platform.system(),
            "scope": "loaders_only_no_fit_no_browser",
        },
        "remaining_acceptance": ["synthetic_workflow_suite", "real_browser_walkthrough"],
    }


def check_performance(report, max_cold_seconds, max_warm_seconds, max_rss_mib):
    """Check explicit engineering budgets, independent of research eligibility.

    Parameters
    ----------
    report : dict
        Completed publication smoke report.
    max_cold_seconds, max_warm_seconds, max_rss_mib : float
        Positive finite local release budgets. Every warm repetition must fit.

    Returns
    -------
    dict
        Limits and violations; no statistical or production eligibility claims.
    """
    limits = {
        "cold_seconds": max_cold_seconds,
        "warm_seconds": max_warm_seconds,
        "peak_process_rss_mib": max_rss_mib,
    }
    if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in limits.values()):
        raise ValueError("Positive finite performance budgets required")
    measured = report["performance"]
    values = {
        "cold_seconds": measured["cold_seconds"],
        "warm_seconds": max(measured["warm_seconds"]),
        "peak_process_rss_mib": measured["peak_process_rss_mib"],
    }
    violations = [k for k, v in values.items() if not np.isfinite(v) or v > limits[k]]
    return {"limits": limits, "violations": violations, "passed": not violations}
