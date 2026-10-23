"""Synthetic paper experiment approval and protocol fixtures."""

from test_recommendation_bundle import proposal_bundle

from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
from oddsjam_ev.io.decision_register import record_decision
from oddsjam_ev.io.discovery import load_discovery
from oddsjam_ev.io.experiment_register import record_experiment_event, register_experiment
from oddsjam_ev.io.recommendations import load_recommendations, publish_recommendations


def approved(tmp_path, monkeypatch):
    _, d, a, c, r, _ = proposal_bundle(tmp_path)
    candidate = (
        load_discovery(d)["candidate_results"].query("view == 'sportsbook'").iloc[0].candidate_id
    )
    base = {
        "hypothesis": "Synthetic signal screening",
        "duration_days": 1,
        "min_event_groups": 1,
        "primary_metric": "reported_roi",
        "funding": "paper_only",
        "max_funded_stake_usd": 0,
        "review_criteria": "Review at end, never promote automatically",
    }
    cap = {
        "schema_version": 1,
        "exact_values": {
            "sportsbook": {
                "Book": {
                    "ui_control": "book",
                    "ui_value": ["Book"],
                    "semantics_verified": True,
                    "evidence_reference": "synthetic",
                }
            }
        },
    }
    bundle = publish_recommendations(
        d,
        a,
        c,
        tmp_path / "proposals",
        RecommendationConfig(paper_protocols={candidate: base}),
        r,
        cap,
    )
    proposal = (
        load_recommendations(bundle)["proposals"]
        .query("action == 'CREATE_TEST'")
        .iloc[0]
        .proposal_id
    )
    decisions = tmp_path / "decisions"
    review = record_decision(
        decisions, bundle, proposal, "UNDER_REVIEW", "synthetic actor", "review", "PROPOSED"
    )
    approval = record_decision(
        decisions,
        bundle,
        proposal,
        "APPROVED",
        "synthetic actor",
        "paper only",
        "UNDER_REVIEW",
        review["event_id"],
    )
    protocol = {
        **base,
        "name": "Synthetic experiment",
        "experiment_type": "signal_screening",
        "filter_id": "TEST - Synthetic",
        "filter_config": {"book": ["Book"]},
        "predicate": {"sportsbook": "Book"},
        "planned_start_utc": "2030-01-01T12:00:00Z",
        "planned_end_utc": "2030-01-02T12:00:00Z",
        "comparison_plan": "No causal control; descriptive only",
        "overlap_policy": "Allow explicit overlap and report it",
        "expected_liquidity": "unknown",
        "stop_rules": "Manual pause for configuration deviation",
        "settlement_followup_days": 2,
    }
    import oddsjam_ev.io.experiment_register as module

    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-01T11:00:00Z")
    root = tmp_path / "experiments"
    registration = register_experiment(
        root,
        bundle,
        decisions,
        proposal,
        approval["event_id"],
        protocol,
        "synthetic actor",
        "register before launch",
    )
    return root, registration, bundle, decisions, approval, c, protocol


def launch(root, registration, monkeypatch):
    import oddsjam_ev.io.experiment_register as module

    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-01T12:01:00Z")
    protocol = registration["payload"]["protocol"]
    return record_experiment_event(
        root,
        registration["experiment_id"],
        "TRANSITION",
        {
            "to_status": "RUNNING",
            "launched_at_utc": "2030-01-01T12:00:00Z",
            "filter_id": protocol["filter_id"],
            "filter_config": protocol["filter_config"],
            "evidence_reference": "synthetic launch evidence",
        },
        "synthetic actor",
        "confirm external launch",
        registration["event_id"],
    )


def enroll(
    root,
    last,
    monkeypatch,
    enrollment_id="one",
    observed="2030-01-01T12:02:00Z",
    event_start="2030-01-01T18:00:00Z",
):
    import oddsjam_ev.io.experiment_register as module

    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-01T12:03:00Z")
    return record_experiment_event(
        root,
        last["experiment_id"],
        "ENROLL",
        {
            "enrollment_id": enrollment_id,
            "observation_key": "signal-" + enrollment_id,
            "observed_at_utc": observed,
            "event_start_utc": event_start,
            "feature_values": {"sportsbook": "Book"},
            "scanner_quote": {"selection": "Away", "line": "not_applicable", "odds": 120},
            "evidence_reference": "synthetic decision-time capture",
        },
        "synthetic actor",
        "enroll unchecked signal",
        last["event_id"],
    )
