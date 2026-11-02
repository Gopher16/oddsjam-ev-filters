"""Verified configuration → prospective fixed-look family → explicit bounded pilot."""

# ruff: noqa: F811

import copy
import json
from pathlib import Path

import pytest
from test_decision_records import configuration
from test_filter_lifecycle import event, translations
from test_paper_definitions import complete, definition_sources  # noqa: F401
from test_prospective_integration import command, launch

from oddsjam_ev.analysis.experiments import replay_experiments
from oddsjam_ev.analysis.prospective import prepare_event
from oddsjam_ev.io.decision_records import load_decisions, publish_decisions
from oddsjam_ev.io.decision_register import record_decision
from oddsjam_ev.io.experiment_register import _append, read_experiment_events, register_experiment
from oddsjam_ev.io.experiment_reports import publish_experiment_report
from oddsjam_ev.io.filter_lifecycle import load_lifecycle_report, publish_lifecycle_report
from oddsjam_ev.io.paper_definitions import publish_definitions, register_definition
from oddsjam_ev.io.prospective import publish_definition_proposal
from oddsjam_ev.io.recommendations import load_recommendations
from oddsjam_ev.io.source_csv import fingerprint


# A batch fixture avoids hundreds of redundant disk replays. It uses the same pure
# capture/settlement validator and immutable writer, then verifies the entire chain.
def complete_terminal_capture(c):
    events = read_experiment_events(c["root"])
    states = replay_experiments(events)
    item = states[c["last"]["experiment_id"]]

    def append(kind, raw, now):
        payload = prepare_event(item, states, kind, raw, now)
        e = _append(
            c["root"],
            events,
            c["last"]["experiment_id"],
            kind,
            payload,
            "synthetic",
            "synthetic prospective evidence",
            now,
        )
        events.append(e)
        item["events"].append(e)
        c["last"] = e
        return e

    for i in range(200):
        key = str(i)
        e = append(
            "ENROLL",
            {
                "enrollment_id": key,
                "observation_key": "synthetic-" + key,
                "observed_at_utc": "2030-01-02T00:02:00Z",
                "event_start_utc": "2030-01-02T18:00:00Z",
                "event_namespace": "synthetic",
                "event_id": "event-" + key,
                "feature_values": {"sportsbook": "Book" if i < 100 else "Other"},
                "feature_available_at_utc": {"sportsbook": "2030-01-02T00:02:00Z"},
                "scanner_quote": {"selection": "Away", "line": "not_applicable", "odds": 100},
                "baseline_config_sha256": fingerprint(item["protocol"]["filter_config"]),
                "parent_membership_reference": "Both books explicitly belong to the parent",
                "evidence_reference": "decision-time quote",
            },
            "2030-01-02T00:03:00Z",
        )
        item["enrollments"][key] = e
    for i in range(200):
        key = str(i)
        e = append(
            "CHECK",
            {
                "enrollment_id": key,
                "status": "checked_available",
                "checked_at_utc": "2030-01-02T00:03:00Z",
                "selection": "Away",
                "line": "not_applicable",
                "odds": 100,
                "available_stake_usd": 1,
                "original_quote_matched": True,
                "check_delay_seconds": 60,
                "reason": "original quote and capacity available",
                "evidence_reference": "synthetic contemporaneous check",
            },
            "2030-01-02T00:04:00Z",
        )
        item["checks"][key] = [e]
    for i in range(200):
        key = str(i)
        won = i < 90 if i < 100 else i < 150
        e = append(
            "SETTLE",
            {
                "enrollment_id": key,
                "result": "won" if won else "lost",
                "outcome_available_at_utc": "2030-01-02T20:00:00Z",
                "evidence_reference": "synthetic selection outcome",
            },
            "2030-01-02T20:01:00Z",
        )
        item["settlements"][key] = e
    assert read_experiment_events(c["root"]) == events


def test_complete_verified_prospective_pilot_path_and_recorded_cap_breaches(
    definition_sources,
    tmp_path,
    monkeypatch,
):
    import oddsjam_ev.io.experiment_register as experiments
    import oddsjam_ev.io.paper_definitions as definitions

    family, priority, _, _, definitions_root = definition_sources
    from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery

    candidate = next(
        r
        for r in load_within_filter_discovery(family)["signal_candidates"].to_dict("records")
        if json.loads(r["predicate_json"])[0]["dimension"] == "sportsbook"
    )
    document = complete(candidate)
    evidence = configuration(candidate)
    evidence["baselines"][0]["settings"]["sportsbooks"]["value"] = ["Book", "Other"]
    sheets = publish_decisions(
        family,
        priority,
        evidence,
        tmp_path / "sheets",
        "2030-01-01T00:00:00Z",
        "2030-01-01T00:00:00Z",
    )
    sheet_pub = load_decisions(sheets)
    sheet = next(
        r
        for r in sheet_pub["records"]
        if r["candidate_id"] == candidate["candidate_id"] and r["design"] == "subgroup_tracking"
    )
    assert sheet["setup_status"] == "ready_for_operator_configuration_review"
    p = document["protocol"]
    p["baseline"]["filter_config"] = {r["control"]: r["baseline_value"] for r in sheet["settings"]}
    p["baseline"]["evidence_reference"] = sheet["baseline"]["evidence_reference"]
    p["configuration_reference"] = {
        "contract_version": 1,
        "publication_id": sheet_pub["manifest"]["publication_id"],
        "decision_id": sheet["decision_id"],
        "record_sha256": sheet["record_sha256"],
    }
    definition = register_definition(
        definitions_root,
        family,
        priority,
        candidate["candidate_id"],
        "pilot-candidate",
        p,
        "synthetic",
        "frozen complete settings",
        decision_publication=sheets,
    )
    publication = publish_definitions(
        definitions_root, tmp_path / "definitions", "2030-01-01T00:00:00Z"
    )
    bundle = publish_definition_proposal(
        publication,
        definition["version_id"],
        tmp_path / "proposals",
        "paper-scanner",
        "All parent observations from both books",
    )
    row = load_recommendations(bundle)["proposals"].iloc[0]
    review = record_decision(
        tmp_path / "approvals",
        bundle,
        row.proposal_id,
        "UNDER_REVIEW",
        "synthetic",
        "review exact paper protocol",
        "PROPOSED",
    )
    approval = record_decision(
        tmp_path / "approvals",
        bundle,
        row.proposal_id,
        "APPROVED",
        "synthetic",
        "paper only",
        "UNDER_REVIEW",
        review["event_id"],
    )
    monkeypatch.setattr(experiments, "utc_now", lambda: "2030-01-01T12:00:00Z")
    root = tmp_path / "experiments"
    reg = register_experiment(
        root,
        bundle,
        tmp_path / "approvals",
        row.proposal_id,
        approval["event_id"],
        json.loads(row.prospective_protocol_json),
        "synthetic",
        "register exact definition",
    )
    c = {
        "root": root,
        "last": reg,
        "protocol": json.loads(row.prospective_protocol_json),
        "journal": tmp_path / "lifecycle",
    }
    policy = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/promotion_policy.example.json").read_text()
    )
    policy["promotion"]["family_definitions"] = [
        {
            "publication": str(publication),
            "version_id": definition["version_id"],
            "trial_arm": "subgroup",
        }
    ]
    policy["evidence_reference"] = "synthetic review of approved prospective policy before launch"
    pe = event(c, monkeypatch, "POLICY", policy, "2030-01-01T12:01:00Z")
    launch(c, monkeypatch)
    complete_terminal_capture(c)
    command(
        c,
        monkeypatch,
        "TRANSITION",
        {
            "to_status": "AWAITING_OUTCOMES",
            "ended_at_utc": "2030-02-02T00:00:00Z",
            "evidence_reference": "confirmed capture end",
        },
        "2030-02-02T00:01:00Z",
    )
    # Reuse the canonical bundle that actually produced the exact configuration source.
    canonical = next((tmp_path / "canonical").glob("*/manifest.json")).parent
    monkeypatch.setattr(experiments, "utc_now", lambda: "2030-02-04T00:01:00Z")
    report = publish_experiment_report(
        root, canonical, tmp_path / "outcomes", "2030-02-04T00:00:00Z"
    )
    re = command(
        c,
        monkeypatch,
        "REVIEW",
        {
            "scheduled_review_utc": "2030-02-04T00:00:00Z",
            "review_assessment_reference": str(report),
            "notes": "terminal review, no automatic production approval",
            "evidence_reference": "explicit completed review",
        },
        "2030-02-04T00:02:00Z",
    )
    report = publish_experiment_report(
        root, canonical, tmp_path / "outcomes", "2030-02-04T00:02:00Z"
    )
    source = {
        "experiment_register": str(root),
        "experiment_id": reg["experiment_id"],
        "review_report": str(report),
        "review_event_id": re["event_id"],
    }
    assessment = event(
        c,
        monkeypatch,
        "EVALUATE_PROMOTION",
        {"policy_event_id": pe["event_id"], "sources": [source]},
        "2030-02-04T00:20:00Z",
    )
    assert assessment["payload"]["assessment"]["results"][0]["eligible_for_human_pilot_review"]
    assert set(assessment["payload"]["runtime_versions"]) == {"numpy", "pandas", "scipy"}
    assert "analysis/promotion.py" in assessment["payload"]["evaluation_code_sha256"]
    with pytest.raises(ValueError, match="already evaluated"):
        event(
            c,
            monkeypatch,
            "EVALUATE_PROMOTION",
            {"policy_event_id": pe["event_id"], "sources": [source]},
            "2030-02-04T00:20:00Z",
        )
    # Continue paper capture from the unchanged parent (both books), while the
    # funded pilot applies only the independently verified trial settings (Book).
    after = {}
    for r in sheet["settings"]:
        state = r["baseline_state"] if r["state"] == "inherited" else "observed"
        after[r["control"]] = {
            "state": state,
            "value": r["final_value"],
            "basis": "prospective_design",
            "reason": "Exact tested scanner setting",
        }
    next_plan = copy.deepcopy(p)
    next_plan.update(
        planned_start_utc="2030-02-05T00:00:00Z",
        planned_end_utc="2030-03-05T00:00:00Z",
        review_dates_utc=["2030-03-07T00:00:00Z"],
    )
    monkeypatch.setattr(definitions, "utc_now", lambda: "2030-02-04T00:21:00Z")
    next_definition = register_definition(
        definitions_root,
        family,
        priority,
        candidate["candidate_id"],
        "pilot-candidate",
        next_plan,
        "synthetic",
        "new prospective version",
        expected_version_id=definition["version_id"],
        decision_publication=sheets,
    )
    next_pub = publish_definitions(
        definitions_root, tmp_path / "definitions", "2030-02-04T00:21:00Z"
    )
    decision = event(
        c,
        monkeypatch,
        "DECISION",
        {
            **source,
            "policy_event_id": pe["event_id"],
            "action": "continue",
            "evidence_reference": "human review, not automatic pass",
            "followup_publication": str(next_pub),
            "followup_version_id": next_definition["version_id"],
        },
        "2030-02-04T00:22:00Z",
    )
    change = event(
        c,
        monkeypatch,
        "CHANGE",
        {
            "filter_id": "pilot-scanner",
            "decision_event_id": decision["event_id"],
            "change_kind": "promote",
            "target_status": "PROD",
            "previous_application_id": None,
            "planned_effective_at_utc": "2030-02-04T02:00:00Z",
            "after_settings": after,
            "ui_evidence": translations(after),
            "evidence_reference": "proposed exact trial configuration",
            "configuration_publication": str(sheets),
            "promotion_assessment_event_id": assessment["event_id"],
            "pilot_budget_attestation": "Synthetic human confirms $1 per wager, $100 program total, 14 days and one active pilot; no real money approved",
        },
        "2030-02-04T01:01:00Z",
    )
    assert not change["payload"]["approval_blockers"]
    assert decision["payload"]["followup"]["definition"]["protocol"]["baseline"]["filter_config"][
        "sportsbooks"
    ] == ["Book", "Other"]
    assert change["payload"]["after_settings"]["sportsbooks"]["value"] == ["Book"]
    approved = event(
        c,
        monkeypatch,
        "APPROVE_CHANGE",
        {
            "change_event_id": change["event_id"],
            "evidence_reference": "explicit bounded pilot approval",
        },
        "2030-02-04T01:02:00Z",
    )
    applied = event(
        c,
        monkeypatch,
        "CONFIRM_APPLICATION",
        {
            "change_event_id": change["event_id"],
            "approval_event_id": approved["event_id"],
            "filter_id": "pilot-scanner",
            "status": "PROD",
            "settings": after,
            "effective_at_utc": "2030-02-04T02:00:00Z",
            "evidence_reference": "synthetic actual application confirmation",
        },
        "2030-02-04T02:01:00Z",
    )
    second_request = copy.deepcopy(change["payload"])
    second_request.update(
        filter_id="second-pilot",
        previous_application_id=None,
        planned_effective_at_utc="2030-02-04T03:00:00Z",
    )
    blocked = event(c, monkeypatch, "CHANGE", second_request, "2030-02-04T02:02:00Z")
    assert (
        "another_confirmed_pilot_requires_explicit_retirement"
        in blocked["payload"]["approval_blockers"]
    )
    assert (
        "definition_already_used_for_a_pilot_new_prospective_evidence_required"
        in blocked["payload"]["approval_blockers"]
    )
    with pytest.raises(ValueError, match="Blocked production"):
        event(
            c,
            monkeypatch,
            "APPROVE_CHANGE",
            {
                "change_event_id": blocked["event_id"],
                "evidence_reference": "cannot stack or repeat approved pilots",
            },
            "2030-02-04T02:03:00Z",
        )
    # Retain an explicitly recorded external breach instead of hiding an actual trade.
    breached = event(
        c,
        monkeypatch,
        "OBSERVE",
        {
            "application_id": applied["event_id"],
            "configuration_sha256": fingerprint(applied["payload"]["settings"]),
            "observation_key": "late-pilot-breach",
            "observed_at_utc": "2030-02-20T00:02:00Z",
            "event_start_utc": "2030-02-20T18:00:00Z",
            "event_namespace": "synthetic",
            "event_id": "later-event",
            "execution_mode": "operator_reported_real_money",
            "stake_usd": 101,
            "american_odds": 100,
            "selection": "Away",
            "line": "not_applicable",
            "availability_status": "not_checked",
            "fill_reference": "synthetic operator-reported wager",
            "evidence_reference": "record a breach, do not authorize it",
        },
        "2030-02-20T00:03:00Z",
    )
    assert set(breached["payload"]["pilot_breaches"]) == {
        "stake_per_wager_cap_exceeded",
        "pilot_duration_cap_exceeded",
        "program_deployed_stake_cap_exceeded",
    }
    output = publish_lifecycle_report(
        c["journal"], tmp_path / "lifecycle-reports", "2030-02-20T00:03:00Z"
    )
    tables = load_lifecycle_report(output)["tables"]
    assert tables["applications"].iloc[0].production_approval == "explicit_human_bounded_pilot_only"
    assert tables["applications"].iloc[0].pilot_expiry_review_due
    assert tables["metrics"].pilot_breach_count.sum() == 1
