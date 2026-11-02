"""Synthetic reviewed outcomes, confirmed scanner versions and nonpooled monitoring."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_paper_definitions import definition_sources, save  # noqa: F401, F811
from test_prospective_integration import (  # noqa: F401, F811
    command,
    enroll,
    launch,
    prospective,
    settle,
)

from oddsjam_ev.analysis.decision_config import CONTROLS
from oddsjam_ev.analysis.filter_lifecycle import monitoring_tables, replay_lifecycle
from oddsjam_ev.dashboard.artifacts import DashboardConfig
from oddsjam_ev.dashboard.lifecycle_artifacts import load_lifecycle_attachment
from oddsjam_ev.io.experiment_reports import publish_experiment_report
from oddsjam_ev.io.filter_lifecycle import (
    load_lifecycle_report,
    publish_lifecycle_report,
    read_lifecycle,
    record_lifecycle_event,
)
from oddsjam_ev.io.paper_definitions import publish_definitions
from oddsjam_ev.io.source_csv import fingerprint

ROOT = Path(__file__).resolve().parents[1]
UI_FIXTURE = None


def settings(book="Book"):
    result = {
        k: {
            "state": "unset",
            "value": None,
            "basis": "prospective_design",
            "reason": "Explicitly reviewed unset control",
        }
        for k in CONTROLS
    }
    result["sportsbooks"].update(state="observed", value=[book])
    result["paper_sizing"].update(
        state="observed",
        value={"method": "fixed_paper_stake", "stake_usd": 1, "applies_to": "both_arms"},
    )
    return result


def event(c, monkeypatch, kind, payload, now="2030-02-04T01:00:00Z"):
    import oddsjam_ev.io.filter_lifecycle as module

    monkeypatch.setattr(module, "utc_now", lambda: now)
    e = record_lifecycle_event(
        c["journal"],
        kind,
        payload,
        "synthetic reviewer",
        "Explicit synthetic review/evidence",
        c.get("head"),
    )
    c["head"] = e["event_id"]
    return e


@pytest.fixture
def reviewed(prospective, tmp_path, monkeypatch):  # noqa: F811
    c = prospective
    c["journal"] = tmp_path / "lifecycle"
    policy = json.loads((ROOT / "configs/filter_review_policy.example.json").read_text())
    policy["evidence_reference"] = "synthetic explicit descriptive policy review"
    policy["monitoring"]["min_wagers"] = 1
    policy["monitoring"]["min_event_groups"] = 1
    policy["monitoring"]["reference_means"] = {"advertised_ev_percent": 3}
    policy["monitoring"]["absolute_alert_deltas"] = {"advertised_ev_percent": 1}
    c["policy"] = event(c, monkeypatch, "POLICY", policy, "2030-01-01T13:00:00Z")
    launch(c, monkeypatch)
    enroll(c, monkeypatch)
    enroll(c, monkeypatch, "two", False)
    settle(c, monkeypatch, result="lost")
    settle(c, monkeypatch, "two", "won")
    command(
        c,
        monkeypatch,
        "TRANSITION",
        {
            "to_status": "AWAITING_OUTCOMES",
            "ended_at_utc": "2030-02-02T00:00:00Z",
            "evidence_reference": "confirmed end",
        },
        "2030-02-02T00:01:00Z",
    )
    from test_atlas_bundle import canonical

    import oddsjam_ev.io.experiment_register as register

    directory = tmp_path / "outcome-source"
    directory.mkdir()
    c["canonical"] = canonical(directory)
    monkeypatch.setattr(register, "utc_now", lambda: "2030-02-04T00:01:00Z")
    report = publish_experiment_report(
        c["root"], c["canonical"], tmp_path / "outcomes", "2030-02-04T00:00:00Z"
    )
    c["review"] = command(
        c,
        monkeypatch,
        "REVIEW",
        {
            "scheduled_review_utc": "2030-02-04T00:00:00Z",
            "review_assessment_reference": str(report),
            "evidence_reference": "review losing subgroup and missing availability",
            "notes": "Descriptive evidence only; no pass",
        },
        "2030-02-04T00:02:00Z",
    )
    monkeypatch.setattr(register, "utc_now", lambda: "2030-02-04T00:03:00Z")
    report = publish_experiment_report(
        c["root"], c["canonical"], tmp_path / "outcomes", "2030-02-04T00:02:00Z"
    )
    command(
        c,
        monkeypatch,
        "TRANSITION",
        {
            "to_status": "READY_FOR_REVIEW",
            "review_assessment_reference": str(report),
            "evidence_reference": "human readiness, inconclusive allowed after followup",
        },
        "2030-02-04T00:03:00Z",
    )
    c["report"] = publish_experiment_report(
        c["root"], c["canonical"], tmp_path / "outcomes", "2030-02-04T00:03:00Z"
    )
    c["decision_payload"] = {
        "policy_event_id": c["policy"]["event_id"],
        "experiment_register": str(c["root"]),
        "experiment_id": c["last"]["experiment_id"],
        "review_report": str(c["report"]),
        "review_event_id": c["review"]["event_id"],
        "action": "inconclusive",
        "evidence_reference": "review evidence with blocking reasons retained",
    }
    return c


def followup(c, tmp_path, monkeypatch, book="Book"):
    import oddsjam_ev.io.paper_definitions as module

    monkeypatch.setattr(module, "utc_now", lambda: "2030-02-04T00:10:00Z")
    inputs = c["definition_inputs"]
    p = inputs[3]["protocol"]
    p.update(
        planned_start_utc="2030-02-05T00:00:00Z",
        planned_end_utc="2030-03-05T00:00:00Z",
        review_dates_utc=["2030-03-07T00:00:00Z"],
    )
    p["baseline"]["filter_config"] = {k: v["value"] for k, v in settings(book).items()}
    old = c.get("followup_id", c["definition"]["version_id"])
    d = save(inputs, expected_version_id=old)
    publication = publish_definitions(inputs[4], tmp_path / "followup", "2030-02-04T00:10:00Z")
    c["followup_id"] = d["version_id"]
    return {"followup_publication": str(publication), "followup_version_id": d["version_id"]}


def decide(c, monkeypatch, action="inconclusive", **extra):
    e = event(c, monkeypatch, "DECISION", {**c["decision_payload"], "action": action, **extra})
    c["decision"] = e
    return e


def translations(after):
    return [
        {
            "control": k,
            "state": v["state"],
            "value": v["value"],
            "semantics_verified": True,
            "platform_version": "synthetic-v1",
            "evidence_reference": "exact setting supported by UI",
            "observed_at_utc": "2030-02-04T01:00:00Z",
        }
        for k, v in after.items()
    ]


def propose(c, monkeypatch, after=None, **extra):
    after = after or settings()
    return event(
        c,
        monkeypatch,
        "CHANGE",
        {
            "filter_id": "test-filter",
            "decision_event_id": c["decision"]["event_id"],
            "evidence_reference": "human proposal only",
            "change_kind": "new_test",
            "target_status": "TEST",
            "previous_application_id": None,
            "planned_effective_at_utc": "2030-02-04T02:00:00Z",
            "after_settings": after,
            "ui_evidence": translations(after),
            **extra,
        },
        "2030-02-04T01:01:00Z",
    )


def approve_apply(c, monkeypatch, change):
    approval = event(
        c,
        monkeypatch,
        "APPROVE_CHANGE",
        {
            "change_event_id": change["event_id"],
            "evidence_reference": "explicit reviewed configuration approval",
        },
        "2030-02-04T01:02:00Z",
    )
    applied = event(
        c,
        monkeypatch,
        "CONFIRM_APPLICATION",
        {
            "change_event_id": change["event_id"],
            "approval_event_id": approval["event_id"],
            "filter_id": change["payload"]["filter_id"],
            "status": change["payload"]["target_status"],
            "effective_at_utc": "2030-02-04T02:00:00Z",
            "settings": change["payload"]["after_settings"],
            "evidence_reference": "operator confirmed external applied settings",
        },
        "2030-02-04T02:01:00Z",
    )
    c["applied"] = applied
    return applied


def observe(c, monkeypatch, key="obs-1", **extra):
    a = c["applied"]
    return event(
        c,
        monkeypatch,
        "OBSERVE",
        {
            "application_id": a["event_id"],
            "configuration_sha256": fingerprint(a["payload"]["settings"]),
            "observation_key": key,
            "observed_at_utc": "2030-02-04T02:02:00Z",
            "event_start_utc": "2030-02-04T18:00:00Z",
            "event_namespace": "synthetic",
            "event_id": "game-1",
            "execution_mode": "paper",
            "stake_usd": 1,
            "american_odds": 120,
            "selection": "Away",
            "line": "not_applicable",
            "availability_status": "not_checked",
            "features": {"advertised_ev_percent": 5},
            "evidence_reference": "explicit monitoring capture",
            **extra,
        },
        "2030-02-04T02:03:00Z",
    )


def test_decisions_do_not_turn_losing_subgroups_or_missing_evidence_into_passes(
    reviewed, monkeypatch
):
    c = reviewed
    assert len(read_lifecycle(c["journal"])) == 1
    decision = decide(c, monkeypatch)
    assert decision["payload"]["source"]["blockers"]
    assert decision["payload"]["production_authorized"] is False
    assert replay_lifecycle(read_lifecycle(c["journal"]))["current"] == {}
    with pytest.raises(ValueError, match="continue/modify"):
        propose(c, monkeypatch)
    with pytest.raises(ValueError, match="supersession"):
        decide(c, monkeypatch, "stop")
    decide(c, monkeypatch, "stop", supersedes_decision_id=decision["event_id"])


def test_full_change_application_monitoring_and_review_cycle(reviewed, tmp_path, monkeypatch):
    c = reviewed
    decision = decide(c, monkeypatch, "modify", **followup(c, tmp_path, monkeypatch))
    change = propose(c, monkeypatch)
    assert not change["payload"]["approval_blockers"]
    assert not replay_lifecycle(read_lifecycle(c["journal"]))["current"]
    application = approve_apply(c, monkeypatch, change)
    obs = observe(c, monkeypatch)
    event(
        c,
        monkeypatch,
        "SETTLE",
        {
            "observation_event_id": obs["event_id"],
            "result": "won",
            "outcome_available_at_utc": "2030-02-04T20:00:00Z",
            "evidence_reference": "explicit paper result",
        },
        "2030-02-04T20:01:00Z",
    )
    report = publish_lifecycle_report(
        c["journal"], tmp_path / "lifecycle-reports", "2030-02-04T20:01:00Z"
    )
    assert (
        publish_lifecycle_report(
            c["journal"], tmp_path / "lifecycle-reports", "2030-02-04T20:01:00Z"
        )
        == report
    )
    result = load_lifecycle_report(report)
    metric = (
        result["tables"]["metrics"]
        .query("execution_mode == 'paper' and availability_cohort == 'all_captured'")
        .iloc[0]
    )
    assert metric.stake_weighted_roi == pytest.approx(1.2)
    assert metric.equal_stake_mean_return == pytest.approx(1.2)
    assert result["tables"]["applications"].iloc[0].application_id == application["event_id"]
    assert (
        result["tables"]["drift"]
        .query("execution_mode == 'paper' and feature == 'advertised_ev_percent'")
        .iloc[0]
        .status
        == "review_alert"
    )
    cfg = DashboardConfig(
        evidence_root=str(tmp_path), lifecycle_report_publication=str(report.relative_to(tmp_path))
    )
    attachment = load_lifecycle_attachment(cfg, {"events": [c["definition"]]})
    assert len(attachment["decisions"]) == 1
    assert load_lifecycle_attachment(cfg, {"events": []})["tables"]["metrics"].empty
    assert load_lifecycle_attachment(replace(cfg, lifecycle_report_publication=None), None) is None
    review = event(
        c,
        monkeypatch,
        "MONITOR_REVIEW",
        {
            "application_id": application["event_id"],
            "action": "investigate",
            "evidence_reference": "human sees drift, does not automatically change settings",
            "monitoring_report": str(report),
        },
        "2030-02-04T20:02:00Z",
    )
    assert not review["payload"]["automatic_action"]
    assert load_lifecycle_report(report)["events"][-1]["event_id"] != review["event_id"]
    assert decision["payload"]["followup"]["definition"]["version_id"] == c["followup_id"]


def test_production_policy_and_unsupported_settings_fail_closed(reviewed, tmp_path, monkeypatch):
    c = reviewed
    decide(c, monkeypatch, "modify", **followup(c, tmp_path, monkeypatch))
    after = settings()
    after.pop("market_width")
    change = propose(c, monkeypatch, after, change_kind="new_test")
    assert "incomplete_target_settings" in change["payload"]["approval_blockers"]
    with pytest.raises(ValueError, match="Blocked"):
        event(
            c,
            monkeypatch,
            "APPROVE_CHANGE",
            {
                "change_event_id": change["event_id"],
                "evidence_reference": "cannot approve unknown controls",
            },
            "2030-02-04T01:02:00Z",
        )
    policy = copy.deepcopy(c["policy"]["payload"])
    policy["scope"] = "production_approval"
    with pytest.raises(ValueError, match="separate prospective"):
        event(c, monkeypatch, "POLICY", policy, "2030-02-04T01:03:00Z")


@pytest.mark.parametrize(
    "extra",
    [
        {"observed_at_utc": "2030-02-04T01:59:00Z"},
        {"configuration_sha256": "wrong"},
        {"event_start_utc": "2030-02-04T02:02:00Z"},
        {
            "execution_mode": "operator_reported_real_money",
            "fill_reference": "cannot turn TEST into funded",
        },
        {"availability_status": "checked_available"},
    ],
)
def test_monitoring_rejects_retrospective_wrong_version_or_funded_test(
    reviewed, tmp_path, monkeypatch, extra
):
    c = reviewed
    decide(c, monkeypatch, "continue", **followup(c, tmp_path, monkeypatch))
    approve_apply(c, monkeypatch, propose(c, monkeypatch))
    with pytest.raises(ValueError):
        observe(c, monkeypatch, **extra)
    assert not replay_lifecycle(read_lifecycle(c["journal"]))["observations"]


def test_approval_is_not_application_and_actual_settings_must_match(
    reviewed, tmp_path, monkeypatch
):
    c = reviewed
    decide(c, monkeypatch, "continue", **followup(c, tmp_path, monkeypatch))
    change = propose(c, monkeypatch)
    with pytest.raises(ValueError, match="approval"):
        event(
            c,
            monkeypatch,
            "CONFIRM_APPLICATION",
            {
                "change_event_id": change["event_id"],
                "approval_event_id": "wrong",
                "settings": settings(),
                "effective_at_utc": "2030-02-04T02:00:00Z",
                "evidence_reference": "not an approved application",
            },
            "2030-02-04T02:01:00Z",
        )
    approval = event(
        c,
        monkeypatch,
        "APPROVE_CHANGE",
        {"change_event_id": change["event_id"], "evidence_reference": "explicit approval"},
        "2030-02-04T02:02:00Z",
    )
    assert not replay_lifecycle(read_lifecycle(c["journal"]))["current"]
    with pytest.raises(ValueError, match="differ"):
        event(
            c,
            monkeypatch,
            "CONFIRM_APPLICATION",
            {
                "change_event_id": change["event_id"],
                "approval_event_id": approval["event_id"],
                "filter_id": "test-filter",
                "status": "TEST",
                "settings": settings("Another"),
                "effective_at_utc": "2030-02-04T02:03:00Z",
                "evidence_reference": "wrong scanner settings",
            },
            "2030-02-04T02:04:00Z",
        )


def test_journal_and_publication_corruption_are_visible(reviewed, tmp_path, monkeypatch):
    c = reviewed
    decide(c, monkeypatch)
    report = publish_lifecycle_report(c["journal"], tmp_path / "reports", "2030-02-04T01:00:00Z")
    (report / "metrics.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_lifecycle_report(report)
    (c["journal"] / "00000002.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        read_lifecycle(c["journal"])


def test_legacy_prod_is_not_promotion_and_new_production_changes_are_blocked(
    reviewed, tmp_path, monkeypatch
):
    c = reviewed
    baseline = event(
        c,
        monkeypatch,
        "BASELINE",
        {
            "filter_id": "test-filter",
            "policy_event_id": c["policy"]["event_id"],
            "status": "PROD",
            "effective_at_utc": "2029-12-01T00:00:00Z",
            "settings": settings(),
            "legacy_context": "preexisting production before this study; not promoted by its paper profit",
            "evidence_reference": "operator confirms current real scanner lifecycle",
        },
        "2030-02-04T00:55:00Z",
    )
    decide(c, monkeypatch, "modify", **followup(c, tmp_path, monkeypatch))
    change = propose(
        c,
        monkeypatch,
        change_kind="promote",
        target_status="PROD",
        previous_application_id=baseline["event_id"],
    )
    assert "production_policy_not_approved" in change["payload"]["approval_blockers"]
    with pytest.raises(ValueError, match="Blocked"):
        event(
            c,
            monkeypatch,
            "APPROVE_CHANGE",
            {
                "change_event_id": change["event_id"],
                "evidence_reference": "paper results cannot authorize production",
            },
            "2030-02-04T01:02:00Z",
        )
    state = replay_lifecycle(read_lifecycle(c["journal"]))
    assert state["current"]["test-filter"] == baseline["event_id"]
    assert baseline["payload"]["production_approval"] == "legacy_unknown_not_a_promotion"


def test_rollback_creates_new_period_and_late_settlement_stays_with_old_version(
    reviewed, tmp_path, monkeypatch
):
    c = reviewed
    baseline = event(
        c,
        monkeypatch,
        "BASELINE",
        {
            "filter_id": "test-filter",
            "policy_event_id": c["policy"]["event_id"],
            "status": "TEST",
            "effective_at_utc": "2030-02-01T00:00:00Z",
            "settings": settings("Old Book"),
            "legacy_context": "preexisting paper settings",
            "evidence_reference": "operator confirms baseline",
        },
        "2030-02-04T00:55:00Z",
    )
    decide(c, monkeypatch, "modify", **followup(c, tmp_path, monkeypatch))
    change = propose(
        c, monkeypatch, change_kind="modify", previous_application_id=baseline["event_id"]
    )
    application = approve_apply(c, monkeypatch, change)
    observation = observe(c, monkeypatch)
    rollback = event(
        c,
        monkeypatch,
        "CHANGE",
        {
            "filter_id": "test-filter",
            "decision_event_id": c["decision"]["event_id"],
            "evidence_reference": "explicit human rollback request; not an automatic alert action",
            "change_kind": "rollback",
            "target_status": "TEST",
            "previous_application_id": application["event_id"],
            "rollback_application_id": baseline["event_id"],
            "planned_effective_at_utc": "2030-02-05T00:00:00Z",
            "after_settings": settings("Old Book"),
            "ui_evidence": translations(settings("Old Book")),
        },
        "2030-02-04T03:00:00Z",
    )
    approval = event(
        c,
        monkeypatch,
        "APPROVE_CHANGE",
        {
            "change_event_id": rollback["event_id"],
            "evidence_reference": "review exact rollback diff",
        },
        "2030-02-04T03:01:00Z",
    )
    c["applied"] = event(
        c,
        monkeypatch,
        "CONFIRM_APPLICATION",
        {
            "change_event_id": rollback["event_id"],
            "approval_event_id": approval["event_id"],
            "filter_id": "test-filter",
            "status": "TEST",
            "effective_at_utc": "2030-02-05T00:00:00Z",
            "settings": settings("Old Book"),
            "evidence_reference": "operator confirmed actual rollback",
        },
        "2030-02-05T00:01:00Z",
    )
    event(
        c,
        monkeypatch,
        "SETTLE",
        {
            "observation_event_id": observation["event_id"],
            "result": "lost",
            "outcome_available_at_utc": "2030-02-05T00:02:00Z",
            "evidence_reference": "late outcome still belongs to the prior version",
        },
        "2030-02-05T00:03:00Z",
    )
    tables = monitoring_tables(read_lifecycle(c["journal"]), "2030-02-05T00:03:00Z")
    assert len(tables["applications"]) == 3
    assert tables["applications"].current.sum() == 1
    assert tables["observations"].iloc[0].application_id == application["event_id"]
    assert (
        tables["metrics"]
        .query(
            "application_id == @application['event_id'] and execution_mode == 'paper' and availability_cohort == 'all_captured'"
        )
        .iloc[0]
        .stake_weighted_roi
        == -1
    )
    early = monitoring_tables(read_lifecycle(c["journal"]), "2030-02-04T02:03:00Z")
    assert len(early["applications"]) == 2
    assert early["observations"].iloc[0].result == "pending"


def test_closure_requires_published_current_review_decision(reviewed, tmp_path, monkeypatch):
    c = reviewed
    decision = decide(c, monkeypatch, "stop")
    report = publish_lifecycle_report(
        c["journal"], tmp_path / "lifecycle-reports", "2030-02-04T01:00:00Z"
    )
    closed = command(
        c,
        monkeypatch,
        "TRANSITION",
        {
            "to_status": "CLOSED",
            "review_decision_reference": str(report),
            "review_decision_event_id": decision["event_id"],
            "evidence_reference": "human closes this paper version, not scanner retirement",
        },
        "2030-02-04T01:01:00Z",
    )
    assert closed["payload"]["review_outcome"] == "stop"
    assert closed["payload"]["lifecycle_publication_id"] == report.name


def test_dashboard_shows_decisions_changes_and_monitoring_without_mutation(
    reviewed, tmp_path, monkeypatch
):
    import test_filter_lifecycle as module
    from streamlit.testing.v1 import AppTest

    c = reviewed
    decide(c, monkeypatch, "continue", **followup(c, tmp_path, monkeypatch))
    approve_apply(c, monkeypatch, propose(c, monkeypatch))
    report = publish_lifecycle_report(
        c["journal"], tmp_path / "lifecycle-reports", "2030-02-04T02:01:00Z"
    )
    cfg = DashboardConfig(
        evidence_root=str(tmp_path), lifecycle_report_publication=str(report.relative_to(tmp_path))
    )
    attachment = load_lifecycle_attachment(cfg, {"events": [c["definition"]]})
    monkeypatch.setattr(module, "UI_FIXTURE", (attachment, c["definition"]["version_id"]))
    before = read_lifecycle(c["journal"])
    app = AppTest.from_string(
        "from test_filter_lifecycle import render_fixture\nrender_fixture()"
    ).run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.dataframe) == 4
    assert read_lifecycle(c["journal"]) == before
    (report / "drift.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_lifecycle_attachment(cfg, {"events": [c["definition"]]})


def render_fixture():
    """Render immutable attachment evidence without a live journal interaction."""
    import streamlit as st

    from oddsjam_ev.dashboard.pages.filter_lifecycle import lifecycle_evidence

    attachment, version = UI_FIXTURE
    st.session_state.lifecycle_report_snapshot = attachment
    lifecycle_evidence(version)


def test_new_outcome_invalidates_prior_decision_and_change_approval(
    reviewed, tmp_path, monkeypatch
):
    from oddsjam_ev.analysis.experiments import replay_experiments
    from oddsjam_ev.io.experiment_register import read_experiment_events

    c = reviewed
    decide(c, monkeypatch, "modify", **followup(c, tmp_path, monkeypatch))
    change = propose(c, monkeypatch)
    item = replay_experiments(read_experiment_events(c["root"]))[c["last"]["experiment_id"]]
    prior = item["settlements"]["one"]
    command(
        c,
        monkeypatch,
        "SETTLE",
        {
            "enrollment_id": "one",
            "result": "won",
            "supersedes_event_id": prior["event_id"],
            "correction_reason": "explicitly corrected outcome changes source evidence",
            "outcome_available_at_utc": "2030-02-04T01:02:00Z",
            "evidence_reference": "corrected selection outcome",
        },
        "2030-02-04T01:02:00Z",
    )
    with pytest.raises(ValueError, match="evidence changed"):
        event(
            c,
            monkeypatch,
            "APPROVE_CHANGE",
            {
                "change_event_id": change["event_id"],
                "evidence_reference": "must refresh review first",
            },
            "2030-02-04T01:03:00Z",
        )


def test_existing_prod_monitoring_keeps_reported_economics_separate(reviewed, monkeypatch):
    c = reviewed
    c["applied"] = event(
        c,
        monkeypatch,
        "BASELINE",
        {
            "filter_id": "existing-prod",
            "policy_event_id": c["policy"]["event_id"],
            "status": "PROD",
            "effective_at_utc": "2029-12-01T00:00:00Z",
            "settings": settings(),
            "legacy_context": "preexisting funded scanner, no new authorization",
            "evidence_reference": "operator confirms baseline",
        },
        "2030-02-04T01:00:00Z",
    )
    obs = observe(
        c,
        monkeypatch,
        execution_mode="operator_reported_real_money",
        stake_usd=20,
        fill_reference="operator-supplied accepted wager reference",
    )
    with pytest.raises(ValueError, match="net profit"):
        event(
            c,
            monkeypatch,
            "SETTLE",
            {
                "observation_event_id": obs["event_id"],
                "result": "won",
                "outcome_available_at_utc": "2030-02-04T20:00:00Z",
                "evidence_reference": "reported outcome is not enough to infer net profit",
            },
            "2030-02-04T20:01:00Z",
        )
    event(
        c,
        monkeypatch,
        "SETTLE",
        {
            "observation_event_id": obs["event_id"],
            "result": "won",
            "outcome_available_at_utc": "2030-02-04T20:00:00Z",
            "reported_net_profit_usd": 2,
            "net_profit_evidence_reference": "operator reported actual net economics",
            "evidence_reference": "explicit reported settlement",
        },
        "2030-02-04T20:02:00Z",
    )
    tables = monitoring_tables(read_lifecycle(c["journal"]), "2030-02-04T20:02:00Z")
    metric = (
        tables["metrics"]
        .query(
            "execution_mode == 'operator_reported_real_money' and availability_cohort == 'all_captured'"
        )
        .iloc[0]
    )
    assert metric.stake_weighted_roi == pytest.approx(0.1)
    assert tables["metrics"].query("execution_mode == 'paper'").settled.sum() == 0


def test_unknown_reference_is_not_no_drift_and_stale_heads_fail(reviewed, monkeypatch):
    c = reviewed
    decide(c, monkeypatch)
    before = read_lifecycle(c["journal"])
    with pytest.raises(ValueError, match="Stale"):
        record_lifecycle_event(
            c["journal"],
            "DECISION",
            c["decision_payload"],
            "actor",
            "stale command",
            c["policy"]["event_id"],
        )
    assert read_lifecycle(c["journal"]) == before
    policy = copy.deepcopy(c["policy"]["payload"])
    policy["version"] = 2
    policy["monitoring"]["reference_means"] = {}
    policy["monitoring"]["absolute_alert_deltas"] = {}
    updated = event(c, monkeypatch, "POLICY", policy, "2030-02-04T01:01:00Z")
    c["applied"] = event(
        c,
        monkeypatch,
        "BASELINE",
        {
            "filter_id": "paper-unknown-reference",
            "policy_event_id": updated["event_id"],
            "status": "TEST",
            "effective_at_utc": "2030-02-04T01:02:00Z",
            "settings": settings(),
            "legacy_context": "preexisting paper scanner",
            "evidence_reference": "operator confirmed baseline",
        },
        "2030-02-04T01:03:00Z",
    )
    observe(c, monkeypatch)
    drift = monitoring_tables(read_lifecycle(c["journal"]), "2030-02-04T02:03:00Z")["drift"]
    assert set(drift.status) == {"unknown_reference_or_no_observations"}
    assert not drift.automatic_action.any()


def test_corrected_outcome_needs_explicit_review_amendment(reviewed, tmp_path, monkeypatch):
    import oddsjam_ev.io.experiment_register as register
    from oddsjam_ev.analysis.experiments import replay_experiments
    from oddsjam_ev.io.experiment_register import read_experiment_events

    c = reviewed
    original = replay_experiments(read_experiment_events(c["root"]))[c["last"]["experiment_id"]][
        "settlements"
    ]["one"]
    command(
        c,
        monkeypatch,
        "SETTLE",
        {
            "enrollment_id": "one",
            "result": "won",
            "supersedes_event_id": original["event_id"],
            "correction_reason": "late operator correction",
            "outcome_available_at_utc": "2030-02-04T00:04:00Z",
            "evidence_reference": "corrected result",
        },
        "2030-02-04T00:04:00Z",
    )
    monkeypatch.setattr(register, "utc_now", lambda: "2030-02-04T00:05:00Z")
    changed = publish_experiment_report(
        c["root"], c["canonical"], tmp_path / "outcomes", "2030-02-04T00:04:00Z"
    )
    c["decision_payload"]["review_report"] = str(changed)
    with pytest.raises(ValueError, match="amendment"):
        decide(c, monkeypatch)
    amendment = command(
        c,
        monkeypatch,
        "REVIEW",
        {
            "scheduled_review_utc": "2030-02-04T00:00:00Z",
            "supersedes_review_event_id": c["review"]["event_id"],
            "correction_reason": "human re-reviewed corrected evidence",
            "review_assessment_reference": str(changed),
            "notes": "Updated descriptive assessment, not another statistical look",
            "evidence_reference": "explicit amendment",
        },
        "2030-02-04T00:06:00Z",
    )
    latest = publish_experiment_report(
        c["root"], c["canonical"], tmp_path / "outcomes", "2030-02-04T00:06:00Z"
    )
    c["decision_payload"].update(review_event_id=amendment["event_id"], review_report=str(latest))
    decision = decide(c, monkeypatch)
    assert decision["payload"]["source"]["review_event_id"] == amendment["event_id"]
