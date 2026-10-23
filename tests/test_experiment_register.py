"""Approval, timing, protocol, availability and immutable lifecycle gates."""

import json

import pytest
from experiment_fixtures import approved, enroll, launch

from oddsjam_ev.analysis.experiment_config import ExperimentPolicy, aware, validate_protocol
from oddsjam_ev.io.decision_register import record_decision
from oddsjam_ev.io.experiment_register import (
    read_experiment_events,
    record_experiment_event,
    register_experiment,
)


def test_register_launch_enroll_without_availability_check(tmp_path, monkeypatch):
    root, registered, *_ = approved(tmp_path, monkeypatch)
    assert registered["kind"] == "REGISTER"
    started = launch(root, registered, monkeypatch)
    observation = enroll(root, started, monkeypatch)
    assert observation["payload"]["timing_status"] == "prospective"
    assert observation["payload"]["availability_status"] == "not_checked"
    with pytest.raises(ValueError, match="Duplicate"):
        enroll(root, observation, monkeypatch)
    with pytest.raises(ValueError, match="Stale"):
        record_experiment_event(
            root,
            started["experiment_id"],
            "TRANSITION",
            {"to_status": "PAUSED"},
            "actor",
            "reason",
            started["event_id"],
        )
    assert len(read_experiment_events(root)) == 3


def test_approval_and_protocol_changes_fail(tmp_path, monkeypatch):
    root, registered, bundle, decisions, approval, c, protocol = approved(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="new approval"):
        register_experiment(
            root,
            bundle,
            decisions,
            registered["payload"]["proposal_id"],
            approval["event_id"],
            dict(protocol, hypothesis="changed"),
            "actor",
            "reason",
        )
    record_decision(
        decisions,
        bundle,
        registered["payload"]["proposal_id"],
        "SUPERSEDED",
        "actor",
        "supersede this scope",
        "APPROVED",
        approval["event_id"],
    )
    with pytest.raises(ValueError, match="superseded"):
        launch(root, registered, monkeypatch)


def test_launch_not_inferred_and_retrospective_is_not_prospective(tmp_path, monkeypatch):
    root, registered, *_ = approved(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="closed or paused"):
        enroll(root, registered, monkeypatch)
    started = launch(root, registered, monkeypatch)
    later = enroll(root, started, monkeypatch, event_start="2030-01-01T12:01:00Z")
    assert later["payload"]["timing_status"] == "retrospective"
    with pytest.raises(ValueError):
        aware("2030-01-01")
    with pytest.raises(ValueError):
        ExperimentPolicy(funding="real_money")


def test_failed_checks_and_configuration_deviation_retained(tmp_path, monkeypatch):
    root, registered, *_ = approved(tmp_path, monkeypatch)
    observation = enroll(root, launch(root, registered, monkeypatch), monkeypatch)
    checked = record_experiment_event(
        root,
        registered["experiment_id"],
        "CHECK",
        {
            "enrollment_id": "one",
            "status": "checked_unavailable",
            "reason": "Quoted selection not offered",
            "checked_at_utc": "2030-01-01T12:03:00Z",
            "evidence_reference": "synthetic failed check",
        },
        "actor",
        "record failure",
        observation["event_id"],
    )
    assert checked["payload"]["check_delay_seconds"] == 60
    deviation = record_experiment_event(
        root,
        registered["experiment_id"],
        "DEVIATION",
        {"details": "Filter settings differ", "evidence_reference": "synthetic deviation"},
        "actor",
        "pause on deviation",
        checked["event_id"],
    )
    with pytest.raises(ValueError, match="closed or paused"):
        enroll(root, deviation, monkeypatch, enrollment_id="two")
    path = root / "00000001.json"
    data = json.loads(path.read_text())
    data["actor"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity"):
        read_experiment_events(root)


def test_changed_quote_is_not_original_quote_confirmation(tmp_path, monkeypatch):
    root, registered, *_ = approved(tmp_path, monkeypatch)
    observation = enroll(root, launch(root, registered, monkeypatch), monkeypatch)
    checked = record_experiment_event(
        root,
        registered["experiment_id"],
        "CHECK",
        {
            "enrollment_id": "one",
            "status": "checked_available",
            "reason": "Different price observed",
            "checked_at_utc": "2030-01-01T12:03:00Z",
            "evidence_reference": "synthetic check",
            "selection": "Away",
            "line": "not_applicable",
            "odds": 110,
            "available_stake_usd": 10,
        },
        "actor",
        "retain changed quote",
        observation["event_id"],
    )
    assert not checked["payload"]["original_quote_matched"]


def test_verification_requires_sampling_rule_and_failed_sample_evidence(tmp_path, monkeypatch):
    root, registered, bundle, decisions, approval, c, protocol = approved(tmp_path, monkeypatch)
    other = dict(
        protocol, experiment_type="execution_verification", name="Synthetic sampled verification"
    )
    with pytest.raises(ValueError, match="sampling"):
        validate_protocol(other, ExperimentPolicy())
    other["sampling_rule"] = "Check every second prospectively enrolled signal"
    registration = register_experiment(
        root,
        bundle,
        decisions,
        registered["payload"]["proposal_id"],
        approval["event_id"],
        other,
        "actor",
        "register sampled protocol",
    )
    observation = enroll(root, launch(root, registration, monkeypatch), monkeypatch)
    with pytest.raises(ValueError, match="selection evidence"):
        record_experiment_event(
            root,
            registration["experiment_id"],
            "CHECK",
            {
                "enrollment_id": "one",
                "status": "checked_unavailable",
                "reason": "Not found",
                "checked_at_utc": "2030-01-01T12:03:00Z",
                "evidence_reference": "synthetic",
            },
            "actor",
            "failed check",
            observation["event_id"],
        )


def test_register_lock_and_failed_append_leave_history_intact(tmp_path, monkeypatch):
    import oddsjam_ev.io.experiment_register as module

    root, registered, *_ = approved(tmp_path, monkeypatch)
    (root / ".writer.lock").write_text("synthetic lock")
    with pytest.raises(ValueError, match="locked"):
        launch(root, registered, monkeypatch)
    (root / ".writer.lock").unlink()

    def fail(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(module.os, "link", fail)
    with pytest.raises(OSError):
        launch(root, registered, monkeypatch)
    assert len(read_experiment_events(root)) == 1


def test_unknown_timing_retained_and_direct_approval_wrong_scope_rejected(tmp_path, monkeypatch):
    root, registered, bundle, decisions, approval, c, protocol = approved(tmp_path, monkeypatch)
    started = launch(root, registered, monkeypatch)
    observation = enroll(root, started, monkeypatch, observed=None, event_start=None)
    assert observation["payload"]["timing_status"] == "timing_unverified"
    from oddsjam_ev.io.recommendations import load_recommendations

    proposal = (
        load_recommendations(bundle)["proposals"].query("action == 'COLLECT_MORE_DATA'").iloc[0]
    )
    with pytest.raises(ValueError, match="Only approvable"):
        register_experiment(
            root,
            bundle,
            decisions,
            proposal.proposal_id,
            approval["event_id"],
            protocol,
            "actor",
            "no bypass",
        )


def test_protocol_config_changes_and_late_registration_rejected(tmp_path, monkeypatch):
    root, registered, bundle, decisions, approval, c, protocol = approved(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="UI translation"):
        register_experiment(
            root,
            bundle,
            decisions,
            registered["payload"]["proposal_id"],
            approval["event_id"],
            dict(protocol, filter_config={"book": ["Other"]}),
            "actor",
            "changed book",
        )
    import oddsjam_ev.io.experiment_register as module

    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-02T00:00:00Z")
    with pytest.raises(ValueError, match="precede enrollment"):
        register_experiment(
            root,
            bundle,
            decisions,
            registered["payload"]["proposal_id"],
            approval["event_id"],
            protocol,
            "actor",
            "late registration",
        )


def test_exchange_policy_records_taker_workflow_without_fill_claim():
    from oddsjam_ev.io.reconcile import POLICY

    policy = ExperimentPolicy()
    assert policy.exchange_order_behavior == "take_existing_orders_only"
    assert POLICY["exchange_order_behavior"] == policy.exchange_order_behavior
    assert POLICY["exchange_order_behavior_provenance"] == "user_confirmed_2026-10-22"
    with pytest.raises(ValueError):
        ExperimentPolicy(exchange_order_behavior="post_limit_orders")
