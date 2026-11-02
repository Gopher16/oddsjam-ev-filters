"""Cross-publication acceptance and a version-preserving paper modification cycle."""

# ruff: noqa: F811, F401

from dataclasses import asdict, replace

import pytest
import yaml
from test_dashboard_workspace import attached, workspace_sources
from test_filter_lifecycle import (
    approve_apply,
    decide,
    event,
    followup,
    observe,
    propose,
    reviewed,
)
from test_paper_definitions import definition_sources
from test_promotion import observations, policy
from test_prospective_integration import prospective

from oddsjam_ev.analysis.explanatory_models import ModelPolicy
from oddsjam_ev.analysis.promotion import evaluate_family
from oddsjam_ev.dashboard.acceptance import inspect_release, reconcile_visuals
from oddsjam_ev.io.explanatory_models import publish_explanatory_models
from oddsjam_ev.io.filter_lifecycle import load_lifecycle_report, publish_lifecycle_report


def test_release_publications_reconcile_replay_and_reject_corruption(workspace_sources, tmp_path):
    run, cfg, queues, visuals = workspace_sources
    canonical = run.result["outputs"]["canonical"]["path"]
    models = publish_explanatory_models(canonical, visuals, tmp_path / "models", ModelPolicy())
    cfg = replace(
        cfg, explanation_publications={run.result["run_id"]: {"global": str(models), "subsets": []}}
    )
    # A durable synthetic config supports a separate real-browser walkthrough.
    (tmp_path / "dashboard.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    before = {
        p: p.read_bytes()
        for base in (queues, visuals, models)
        for p in base.iterdir()
        if p.is_file()
    }
    report = inspect_release(cfg, run.result["run_id"], repeats=1)
    assert report["attachments"]["models"] == "verified"
    assert report["attachments"]["lifecycle"] == "not_attached"
    assert report["reconciliation"]["canonical_wagers"] == report["reconciliation"]["visual_wagers"]
    assert len(report["charts"]) == 10
    assert report["models"][0]["declined_populations"] > 0  # never fake a model for sparse data
    assert report["performance"]["warm_median_seconds"] > 0
    again = inspect_release(cfg, run.result["run_id"], repeats=1)
    for name in ("charts", "models", "reconciliation", "attachments"):
        assert again[name] == report[name]
    assert before == {p: p.read_bytes() for p in before}
    with pytest.raises(ValueError, match="repeats"):
        inspect_release(cfg, run.result["run_id"], repeats=True)
    (models / "models.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        inspect_release(cfg, run.result["run_id"], repeats=1)
    (models / "models.json").write_bytes(before[models / "models.json"])


@pytest.mark.parametrize(
    "case,wins,other,blockers,expected",
    [
        ("positive", 90, 50, [], True),
        ("negative", 20, 50, [], False),
        ("ambiguous", 90, 50, ["unresolved_protocol_deviation"], False),
        ("no_winner", 50, 50, [], False),
    ],
)
def test_terminal_scenarios_keep_full_claim_family(case, wins, other, blockers, expected):
    result = evaluate_family(
        policy(), {"version-1": observations(wins, other)}, {"version-1": blockers}
    )
    assert len(result["hypotheses"]) == 8
    assert bool(result["results"][0]["eligible_for_human_pilot_review"]) is expected, case


def test_discovery_definition_review_modification_monitoring_keeps_versions(
    reviewed, tmp_path, monkeypatch
):
    # reviewed builds frozen discovery/definition/approval, actual launch, both
    # arms, availability/outcomes and explicit scheduled review through public APIs.
    c = reviewed
    original = c["definition"]["version_id"]
    next_version = followup(c, tmp_path, monkeypatch)
    decision = decide(c, monkeypatch, "modify", **next_version)
    assert next_version["followup_version_id"] != original
    change = propose(c, monkeypatch)
    applied = approve_apply(c, monkeypatch, change)
    observation = observe(c, monkeypatch)
    event(
        c,
        monkeypatch,
        "SETTLE",
        {
            "observation_event_id": observation["event_id"],
            "result": "won",
            "outcome_available_at_utc": "2030-02-04T20:00:00Z",
            "evidence_reference": "synthetic release settlement",
        },
        "2030-02-04T20:01:00Z",
    )
    output = publish_lifecycle_report(c["journal"], tmp_path / "reports", "2030-02-04T20:01:00Z")
    before = {p: p.read_bytes() for p in output.iterdir() if p.is_file()}
    assert (
        publish_lifecycle_report(c["journal"], tmp_path / "reports", "2030-02-04T20:01:00Z")
        == output
    )
    report = load_lifecycle_report(output)
    assert before == {p: p.read_bytes() for p in before}
    assert decision["payload"]["production_authorized"] is False
    rows = report["tables"]["observations"]
    assert set(rows.application_id) == {applied["event_id"]}
    assert report["tables"]["applications"].iloc[0].status == "TEST"
    assert change["payload"]["decision_event_id"] == decision["event_id"]
    # Publication corruption is never accepted as a completed feedback cycle.
    (output / "applications.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_lifecycle_report(output)


def test_economic_reconciliation_detects_matching_ids_with_changed_values():
    from test_visual_diagnostics import inputs

    from oddsjam_ev.analysis.visual_config import VisualConfig
    from oddsjam_ev.analysis.visual_diagnostics import build_visual_diagnostics

    canonical, dimensions, diagnostics = inputs()
    tables, _ = build_visual_diagnostics(canonical, dimensions, diagnostics, VisualConfig())
    assert reconcile_visuals(canonical["bet_facts"], tables)["population_economics_reconciled"] == 2
    tables["visual_candidates"].loc[0, "profit"] += 1
    with pytest.raises(ValueError, match="economic reconciliation"):
        reconcile_visuals(canonical["bet_facts"], tables)
    tables["visual_candidates"].loc[0, "profit"] -= 1
    mask = tables["visual_cohorts"].scope_kind.eq("population") & tables["visual_cohorts"].view.eq(
        "overall"
    )
    tables["visual_cohorts"].loc[mask, "stake_weighted_roi"] = 42
    with pytest.raises(ValueError, match="economics reconciliation"):
        reconcile_visuals(canonical["bet_facts"], tables)


def test_performance_budgets_are_explicit_and_check_every_repetition():
    from oddsjam_ev.dashboard.acceptance import check_performance

    report = {
        "performance": {"cold_seconds": 2, "warm_seconds": [1, 9, 1], "peak_process_rss_mib": 2000}
    }
    assert check_performance(report, 3, 10, 3000)["passed"]
    assert check_performance(report, 3, 2, 3000)["violations"] == ["warm_seconds"]
    assert check_performance(report, 1, 2, 1000)["violations"] == [
        "cold_seconds",
        "warm_seconds",
        "peak_process_rss_mib",
    ]
    for bad in (True, 0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            check_performance(report, bad, 2, 3000)
