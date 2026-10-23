"""Prospective timing, unchecked paper signals, mode conflicts and pinned reports."""

import subprocess
import sys
from pathlib import Path

import pytest
from experiment_fixtures import approved, enroll, launch
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.experiment_outcomes import build_experiment_outcomes
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.experiment_register import read_experiment_events, record_experiment_event
from oddsjam_ev.io.experiment_reports import load_experiment_report, publish_experiment_report
from oddsjam_ev.io.reconcile import REAL_FILTER, classify_execution


def linked(tmp_path, monkeypatch):
    root, reg, bundle, decisions, approval, canonical, protocol = approved(tmp_path, monkeypatch)
    observation = enroll(root, launch(root, reg, monkeypatch), monkeypatch)
    tables = load_canonical_bundle(canonical)
    source = tables["source_features"].iloc[0]._ingest_record_id
    last = record_experiment_event(
        root,
        reg["experiment_id"],
        "LINK",
        {
            "enrollment_id": "one",
            "source_record_id": source,
            "evidence_reference": "synthetic exact source identity",
        },
        "actor",
        "link explicitly",
        observation["event_id"],
    )
    return root, last, canonical, tables


def test_unchecked_signals_are_valid_paper_research_not_execution(tmp_path, monkeypatch):
    root, last, c, tables = linked(tmp_path, monkeypatch)
    result = build_experiment_outcomes(read_experiment_events(root), tables, "2030-01-04T12:00:00Z")
    row = result["experiment_outcomes"].iloc[0]
    assert row.prospective_paper_roi_eligible and row.availability_status == "not_checked"
    metric = result["experiment_metrics"].iloc[0]
    assert metric.eligible_count == 1 and metric.hypothetical_reported_roi == pytest.approx(1.2)
    assert not metric.funded_execution_authorized
    assert not result["experiment_audit"].promotion_allowed.any()
    assert result["experiment_availability"].iloc[0].denominator_enrollments == 1
    assert classify_execution("PROD - Other", "2026-10-22") == "paper"


def test_funding_conflicts_pending_and_retrospective_never_enter_roi(tmp_path, monkeypatch):
    root, last, c, tables = linked(tmp_path, monkeypatch)
    events = read_experiment_events(root)
    tables["bet_facts"].loc[:, "saved_filter_names"] = REAL_FILTER
    result = build_experiment_outcomes(events, tables, "2030-01-04T12:00:00Z")
    assert not result["experiment_outcomes"].prospective_paper_roi_eligible.any()
    assert result["experiment_metrics"].iloc[0].execution_status == "real_money"
    assert result["experiment_metrics"].eligible_stake_usd.isna().all()
    tables["bet_facts"].loc[:, "saved_filter_names"] = "TEST - A"
    tables["bet_facts"].loc[:, "reported_roi_eligible"] = False
    tables["bet_facts"].loc[:, "settlement_status"] = "pending"
    result = build_experiment_outcomes(events, tables, "2030-01-02T12:00:00Z")
    assert result["experiment_audit"].iloc[0].pending_count == 1
    assert result["experiment_metrics"].eligible_stake_usd.isna().all()
    early = build_experiment_outcomes(
        read_experiment_events(root, "2030-01-01T11:30:00Z"), tables, "2030-01-01T11:30:00Z"
    )
    assert early["experiment_outcomes"].empty


def test_checked_unavailable_and_inconclusive_checks_are_retained(tmp_path, monkeypatch):
    root, last, c, tables = linked(tmp_path, monkeypatch)
    checked = record_experiment_event(
        root,
        last["experiment_id"],
        "CHECK",
        {
            "enrollment_id": "one",
            "status": "checked_unavailable",
            "reason": "Price no longer offered",
            "checked_at_utc": "2030-01-01T12:03:00Z",
            "evidence_reference": "synthetic check",
        },
        "actor",
        "retain failure",
        last["event_id"],
    )
    record_experiment_event(
        root,
        last["experiment_id"],
        "CHECK",
        {
            "enrollment_id": "one",
            "status": "inconclusive",
            "reason": "Could not confirm limit",
            "checked_at_utc": "2030-01-01T12:03:00Z",
            "evidence_reference": "synthetic later check",
        },
        "actor",
        "retain both checks",
        checked["event_id"],
    )
    result = build_experiment_outcomes(read_experiment_events(root), tables, "2030-01-04T12:00:00Z")
    assert result["experiment_checks"].status.tolist() == ["checked_unavailable", "inconclusive"]
    assert result["experiment_metrics"].iloc[0].availability_status == "inconclusive"


def test_pinned_report_replay_cutoff_corruption_and_failure(tmp_path, monkeypatch):
    root, last, c, tables = linked(tmp_path, monkeypatch)
    output = publish_experiment_report(root, c, tmp_path / "reports", "2030-01-04T12:00:00Z")
    before = load_experiment_report(output)
    assert (
        publish_experiment_report(root, c, tmp_path / "reports", "2030-01-04T12:00:00Z") == output
    )
    for name, frame in before.items():
        assert_frame_equal(frame, load_experiment_report(output)[name])
    with pytest.raises(ValueError, match="after report cutoff"):
        publish_experiment_report(root, c, tmp_path / "reports", "2020-01-01T00:00:00Z")
    (output / "experiment_outcomes.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_experiment_report(output)
    import oddsjam_ev.io.experiment_reports as module

    def fail(*args, **kwargs):
        raise OSError("synthetic failure")

    monkeypatch.setattr(module, "write_json", fail)
    with pytest.raises(OSError):
        publish_experiment_report(root, c, tmp_path / "failed", "2030-01-04T12:00:00Z")
    assert not list((tmp_path / "failed").iterdir())


def test_cli_empty_register_and_no_fabricated_launches(tmp_path):
    from test_atlas_bundle import canonical

    source = canonical(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/build_experiment_report.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--register",
            str(tmp_path / "empty"),
            "--canonical",
            str(source),
            "--as-of",
            "2030-01-01T00:00:00Z",
            "--output-root",
            str(tmp_path / "reports"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out = next((tmp_path / "reports").glob("*/manifest.json")).parent
    assert all(frame.empty for frame in load_experiment_report(out).values())


def test_review_requires_pinned_report_and_explicit_human_outcome(tmp_path, monkeypatch):
    import oddsjam_ev.io.experiment_register as module

    root, last, canonical, _ = linked(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-02T12:01:00Z")
    awaiting = record_experiment_event(
        root,
        last["experiment_id"],
        "TRANSITION",
        {"to_status": "AWAITING_OUTCOMES"},
        "actor",
        "enrollment window complete",
        last["event_id"],
    )
    with pytest.raises(ValueError, match="pinned outcome report"):
        record_experiment_event(
            root,
            last["experiment_id"],
            "TRANSITION",
            {"to_status": "READY_FOR_REVIEW"},
            "actor",
            "review",
            awaiting["event_id"],
        )
    report = publish_experiment_report(
        root, canonical, tmp_path / "reports", "2030-01-02T12:01:00Z"
    )
    ready = record_experiment_event(
        root,
        last["experiment_id"],
        "TRANSITION",
        {"to_status": "READY_FOR_REVIEW", "review_assessment_reference": str(report)},
        "actor",
        "review pinned evidence",
        awaiting["event_id"],
    )
    with pytest.raises(ValueError, match="Explicit human review"):
        record_experiment_event(
            root,
            last["experiment_id"],
            "TRANSITION",
            {"to_status": "CLOSED"},
            "actor",
            "close",
            ready["event_id"],
        )
    closed = record_experiment_event(
        root,
        last["experiment_id"],
        "TRANSITION",
        {"to_status": "CLOSED", "review_outcome": "inconclusive"},
        "actor",
        "human review; no funded promotion",
        ready["event_id"],
    )
    assert closed["payload"]["review_outcome"] == "inconclusive"
    assert not closed["external_execution_performed"]
