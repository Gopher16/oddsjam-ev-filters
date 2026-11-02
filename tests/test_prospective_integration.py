"""Synthetic definition → approval → launch → capture → settlement → review cycles."""

import copy
import json
from dataclasses import replace

import pytest
from test_paper_definitions import definition_sources, save  # noqa: F401, F811

from oddsjam_ev.analysis.experiment_config import ExperimentPolicy, validate_protocol
from oddsjam_ev.analysis.prospective import build_prospective_tables, matches
from oddsjam_ev.dashboard.artifacts import DashboardConfig
from oddsjam_ev.dashboard.prospective_artifacts import load_prospective_report
from oddsjam_ev.io.decision_register import record_decision
from oddsjam_ev.io.experiment_register import (
    read_experiment_events,
    record_experiment_event,
    register_experiment,
)
from oddsjam_ev.io.experiment_reports import load_experiment_report, publish_experiment_report
from oddsjam_ev.io.paper_definitions import publish_definitions
from oddsjam_ev.io.prospective import publish_definition_proposal
from oddsjam_ev.io.recommendations import load_recommendations
from oddsjam_ev.io.source_csv import fingerprint

UI_FIXTURE = None


@pytest.fixture
def prospective(definition_sources, tmp_path, monkeypatch):  # noqa: F811
    import oddsjam_ev.io.experiment_register as module

    # Small support thresholds are fixture choices, never production policy changes.
    definition_sources[3]["protocol"]["min_wagers_per_arm"] = 1
    definition_sources[3]["protocol"]["min_event_groups_per_arm"] = 1
    definition = save(definition_sources)
    publication = publish_definitions(
        definition_sources[4], tmp_path / "definitions", "2030-01-01T00:00:00Z"
    )
    proposal = publish_definition_proposal(
        publication,
        definition["version_id"],
        tmp_path / "proposals",
        "TEST - Synthetic",
        "Capture every distinct parent observation, including complement",
    )
    row = load_recommendations(proposal)["proposals"].iloc[0]
    protocol = json.loads(row.prospective_protocol_json)
    decisions = tmp_path / "decisions"
    review = record_decision(
        decisions,
        proposal,
        row.proposal_id,
        "UNDER_REVIEW",
        "actor",
        "review exact frozen protocol",
        "PROPOSED",
    )
    approval = record_decision(
        decisions,
        proposal,
        row.proposal_id,
        "APPROVED",
        "actor",
        "paper only",
        "UNDER_REVIEW",
        review["event_id"],
    )
    monkeypatch.setattr(module, "utc_now", lambda: "2030-01-01T12:00:00Z")
    root = tmp_path / "experiments"
    registration = register_experiment(
        root,
        proposal,
        decisions,
        row.proposal_id,
        approval["event_id"],
        protocol,
        "actor",
        "register frozen version",
    )
    return {
        "root": root,
        "last": registration,
        "definition": definition,
        "publication": publication,
        "proposal": proposal,
        "protocol": protocol,
        "decisions": decisions,
        "approval": approval,
        "proposal_id": row.proposal_id,
        "definition_inputs": definition_sources,
    }


def command(context, monkeypatch, kind, payload, now):
    import oddsjam_ev.io.experiment_register as module

    monkeypatch.setattr(module, "utc_now", lambda: now)
    event = record_experiment_event(
        context["root"],
        context["last"]["experiment_id"],
        kind,
        payload,
        "actor",
        "explicit synthetic evidence",
        context["last"]["event_id"],
    )
    context["last"] = event
    return event


def launch(context, monkeypatch):
    return command(
        context,
        monkeypatch,
        "TRANSITION",
        {
            "to_status": "RUNNING",
            "launched_at_utc": "2030-01-02T00:00:00Z",
            "filter_id": context["protocol"]["filter_id"],
            "filter_config": context["protocol"]["filter_config"],
            "evidence_reference": "operator confirms exact baseline applied",
        },
        "2030-01-02T00:01:00Z",
    )


def capture(context, key="one", subgroup=True):
    terms = context["protocol"]["predicate"]
    values = {}
    for term in terms:
        values[term["dimension"]] = (
            term["value"]
            if term["operator"] == "=="
            else str(float(term["endpoint"]) + (1 if term["operator"] in {">", ">="} else -1))
        )
    if not subgroup:
        term = terms[0]
        values[term["dimension"]] = (
            "not matching"
            if term["operator"] == "=="
            else str(float(term["endpoint"]) + (-1 if term["operator"] in {">", ">="} else 1))
        )
    return {
        "enrollment_id": key,
        "observation_key": "observation-" + key,
        "observed_at_utc": "2030-01-02T00:02:00Z",
        "event_start_utc": "2030-01-02T18:00:00Z",
        "event_namespace": "synthetic",
        "event_id": "game-" + key,
        "feature_values": values,
        "feature_available_at_utc": {k: "2030-01-02T00:02:00Z" for k in values},
        "scanner_quote": {"selection": "Away", "line": "not_applicable", "odds": 120},
        "baseline_config_sha256": fingerprint(context["protocol"]["filter_config"]),
        "parent_membership_reference": "operator confirms original parent matched",
        "evidence_reference": "decision-time synthetic quote",
    }


def enroll(context, monkeypatch, key="one", subgroup=True):
    return command(
        context, monkeypatch, "ENROLL", capture(context, key, subgroup), "2030-01-02T00:03:00Z"
    )


def settle(context, monkeypatch, key="one", result="won", **extra):
    return command(
        context,
        monkeypatch,
        "SETTLE",
        {
            "enrollment_id": key,
            "result": result,
            "outcome_available_at_utc": "2030-01-02T20:00:00Z",
            "evidence_reference": "operator captured published game result",
            **extra,
        },
        "2030-01-02T20:01:00Z",
    )


def test_exact_numeric_predicates_and_unknown_complement():
    term = [
        {
            "dimension": "advertised_ev_percent",
            "operator": ">",
            "endpoint": "3",
            "units": "percentage_points",
        }
    ]
    assert not matches(term, {"advertised_ev_percent": "3.000"})
    assert matches(term, {"advertised_ev_percent": "3.00000000000000001"})
    for value in (None, float("nan"), True, "n/a"):
        with pytest.raises(ValueError):
            matches(term, {"advertised_ev_percent": value})
    with pytest.raises(ValueError, match="Missing"):
        matches(term, {})


def test_full_version_bound_cycle(prospective, tmp_path, monkeypatch):
    c = prospective
    assert c["last"]["kind"] == "REGISTER"
    assert not read_experiment_events(c["root"])[0]["external_execution_performed"]
    with pytest.raises(ValueError, match="closed or paused"):
        enroll(c, monkeypatch)
    launch(c, monkeypatch)
    one = enroll(c, monkeypatch)
    assert one["payload"]["arm"] == "subgroup"
    two = enroll(c, monkeypatch, "two", False)
    assert two["payload"]["arm"] == "complement"
    for key in ("one", "two"):
        command(
            c,
            monkeypatch,
            "CHECK",
            {
                "enrollment_id": key,
                "status": "checked_available",
                "reason": "same quote still offered",
                "checked_at_utc": "2030-01-02T00:03:00Z",
                "selection": "Away",
                "line": "not_applicable",
                "odds": 120,
                "available_stake_usd": 1,
                "evidence_reference": "paper quote check",
            },
            "2030-01-02T00:04:00Z",
        )
    command(
        c,
        monkeypatch,
        "LINK",
        {
            "enrollment_id": "one",
            "source_record_id": "source:one",
            "evidence_reference": "manual source identity, not a fill",
        },
        "2030-01-02T00:05:00Z",
    )
    settle(c, monkeypatch)
    settle(c, monkeypatch, "two", "lost")
    command(
        c,
        monkeypatch,
        "TRANSITION",
        {
            "to_status": "AWAITING_OUTCOMES",
            "ended_at_utc": "2030-02-02T00:00:00Z",
            "evidence_reference": "operator ended capture",
        },
        "2030-02-02T00:01:00Z",
    )
    # Canonical fixture is independent historical evidence, not automatic enrollment.
    from test_atlas_bundle import canonical as canonical_fixture

    (tmp_path / "canonical-source").mkdir()
    canonical = canonical_fixture(tmp_path / "canonical-source")
    import oddsjam_ev.io.experiment_register as register

    monkeypatch.setattr(register, "utc_now", lambda: "2030-02-04T00:01:00Z")
    report = publish_experiment_report(
        c["root"], canonical, tmp_path / "reports", "2030-02-04T00:00:00Z"
    )
    assert (
        publish_experiment_report(
            c["root"], canonical, tmp_path / "reports", "2030-02-04T00:00:00Z"
        )
        == report
    )
    tables = load_experiment_report(report)
    metrics = (
        tables["prospective_metrics"]
        .query("cohort == 'all_captured_hypothetical'")
        .set_index("arm")
    )
    assert metrics.loc["parent", "stake_weighted_roi"] == pytest.approx(0.1)
    assert metrics.loc["parent", "settled_wagers"] == 2
    assert tables["prospective_reviews"].iloc[0].review_due
    assert not tables["prospective_reviews"].iloc[0].review_recorded
    assert tables["prospective_reviews"].iloc[0].readiness == "eligible_for_human_readiness_review"
    cfg = DashboardConfig(
        evidence_root=str(tmp_path),
        prospective_report_publication=str(report.relative_to(tmp_path)),
    )
    attached = load_prospective_report(cfg, {"events": [c["definition"]]})
    assert len(attached["tables"]["prospective_observations"]) == 2
    import test_prospective_integration as module
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        module,
        "UI_FIXTURE",
        (
            {
                "events": [c["definition"]],
                "manifest": {"specification": {"as_of": "2030-01-01T00:00:00Z"}},
                "latest": {c["definition"]["test_key"]: c["definition"]["version_id"]},
            },
            attached,
        ),
        raising=False,
    )
    app = AppTest.from_string(
        "from test_prospective_integration import render_prospective_fixture\nrender_prospective_fixture()"
    ).run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.dataframe) == 4
    assert app.dataframe[0].value.iloc[0]["Status"] == "AWAITING_OUTCOMES"
    assert any("Version-specific prospective evidence" in h.value for h in app.subheader)
    assert load_prospective_report(cfg, {"events": []})["tables"]["prospective_metrics"].empty
    assert load_prospective_report(replace(cfg, prospective_report_publication=None), None) is None
    command(
        c,
        monkeypatch,
        "REVIEW",
        {
            "scheduled_review_utc": "2030-02-04T00:00:00Z",
            "review_assessment_reference": str(report),
            "evidence_reference": "operator reviewed both arms",
            "notes": "Descriptive paper evidence only; no production promotion",
        },
        "2030-02-04T00:02:00Z",
    )
    result = build_prospective_tables(read_experiment_events(c["root"]), "2030-02-04T00:03:00Z")
    assert result["prospective_reviews"].iloc[0].review_recorded
    assert not result["prospective_reviews"].promotion_eligible.any()
    # Earlier report is immutable after the human review event.
    assert not load_experiment_report(report)["prospective_reviews"].iloc[0].review_recorded


@pytest.mark.parametrize(
    "change",
    [
        {"observed_at_utc": "2030-01-01T23:59:00Z"},
        {"event_start_utc": "2030-01-02T00:02:00Z"},
        {"definition_version_id": "wrong"},
        {"feature_available_at_utc": {}},
        {"baseline_config_sha256": "wrong"},
        {"feature_values": {}},
        {"event_id": ""},
    ],
)
def test_capture_rejects_unsupported_evidence(prospective, monkeypatch, change):
    launch(prospective, monkeypatch)
    with pytest.raises(ValueError):
        command(
            prospective,
            monkeypatch,
            "ENROLL",
            {**capture(prospective), **change},
            "2030-01-02T00:03:00Z",
        )
    assert len(read_experiment_events(prospective["root"])) == 2


def test_pending_late_revision_and_asof(prospective, monkeypatch):
    c = prospective
    launch(c, monkeypatch)
    enroll(c, monkeypatch)
    pending = settle(c, monkeypatch, result="pending")
    before = build_prospective_tables(read_experiment_events(c["root"]), "2030-01-03T00:00:00Z")
    assert before["prospective_metrics"].settled_wagers.sum() == 0
    with pytest.raises(ValueError, match="revisions"):
        settle(c, monkeypatch)
    won = settle(
        c,
        monkeypatch,
        supersedes_event_id=pending["event_id"],
        correction_reason="Result now published",
    )
    assert won["payload"]["paper_profit_usd"] == 1.2
    historical = read_experiment_events(c["root"], "2030-01-02T20:00:00Z")
    assert (
        build_prospective_tables(historical, "2030-01-02T20:00:00Z")["prospective_observations"]
        .iloc[0]
        .result
        == "pending"
    )
    with pytest.raises(ValueError, match="Cross-version"):
        command(
            c,
            monkeypatch,
            "SETTLE",
            {**won["payload"], "definition_version_id": "another"},
            "2030-01-02T20:02:00Z",
        )


def test_changed_protocol_pauses_duplicates_and_unknown_launch(prospective, monkeypatch):
    c = prospective
    altered = copy.deepcopy(c["protocol"])
    altered["filter_config"]["book"] = ["Another"]
    with pytest.raises(ValueError, match="differs"):
        validate_protocol(altered, ExperimentPolicy())
    launch(c, monkeypatch)
    enroll(c, monkeypatch)
    with pytest.raises(ValueError, match="Duplicate"):
        enroll(c, monkeypatch)
    command(
        c,
        monkeypatch,
        "TRANSITION",
        {"to_status": "PAUSED", "evidence_reference": "explicit pause"},
        "2030-01-02T00:05:00Z",
    )
    with pytest.raises(ValueError, match="closed or paused"):
        enroll(c, monkeypatch, "new")
    with pytest.raises(ValueError, match="Actual"):
        command(
            c,
            monkeypatch,
            "TRANSITION",
            {"to_status": "AWAITING_OUTCOMES", "evidence_reference": "missing actual end"},
            "2030-01-02T00:06:00Z",
        )


def test_cross_version_capture_and_economics_never_pool(prospective, tmp_path, monkeypatch):
    c = prospective
    launch(c, monkeypatch)
    enroll(c, monkeypatch)
    import oddsjam_ev.io.experiment_register as register
    import oddsjam_ev.io.paper_definitions as definitions

    inputs = c["definition_inputs"]
    inputs[3]["protocol"]["planned_start_utc"] = "2030-01-03T00:00:00Z"
    monkeypatch.setattr(definitions, "utc_now", lambda: "2030-01-02T00:10:00Z")
    second = save(inputs, expected_version_id=c["definition"]["version_id"])
    publication = publish_definitions(inputs[4], tmp_path / "definitions", "2030-01-02T00:10:00Z")
    with pytest.raises(ValueError, match="Superseded"):
        publish_definition_proposal(
            publication, c["definition"]["version_id"], tmp_path / "proposals", "TEST", "all parent"
        )
    bundle = publish_definition_proposal(
        publication, second["version_id"], tmp_path / "proposals", "TEST", "all parent"
    )
    row = load_recommendations(bundle)["proposals"].iloc[0]
    review = record_decision(
        c["decisions"], bundle, row.proposal_id, "UNDER_REVIEW", "actor", "new version", "PROPOSED"
    )
    approval = record_decision(
        c["decisions"],
        bundle,
        row.proposal_id,
        "APPROVED",
        "actor",
        "exact new version",
        "UNDER_REVIEW",
        review["event_id"],
    )
    monkeypatch.setattr(register, "utc_now", lambda: "2030-01-02T00:11:00Z")
    protocol = json.loads(row.prospective_protocol_json)
    reg = register_experiment(
        c["root"],
        bundle,
        c["decisions"],
        row.proposal_id,
        approval["event_id"],
        protocol,
        "actor",
        "new version",
    )
    other = {**c, "last": reg, "protocol": protocol}
    command(
        other,
        monkeypatch,
        "TRANSITION",
        {
            "to_status": "RUNNING",
            "launched_at_utc": "2030-01-03T00:00:00Z",
            "filter_id": "TEST",
            "filter_config": protocol["filter_config"],
            "evidence_reference": "confirmed new version launch",
        },
        "2030-01-03T00:01:00Z",
    )
    repeated = capture(other)
    repeated["observed_at_utc"] = "2030-01-03T00:02:00Z"
    repeated["event_start_utc"] = "2030-01-03T18:00:00Z"
    with pytest.raises(ValueError, match="across this test's versions"):
        command(other, monkeypatch, "ENROLL", repeated, "2030-01-03T00:03:00Z")
    repeated["observation_key"] = "genuinely-new-observation"
    command(other, monkeypatch, "ENROLL", repeated, "2030-01-03T00:03:00Z")
    tables = build_prospective_tables(read_experiment_events(c["root"]), "2030-01-04T00:00:00Z")
    assert tables["prospective_observations"].groupby("definition_version_id").size().tolist() == [
        1,
        1,
    ]
    assert tables["prospective_metrics"].definition_version_id.nunique() == 2
    # Same underlying event can appear in different tests/versions: disclose dependence.
    assert tables["prospective_observations"].shared_event_experiments.tolist() == [2, 2]


def test_failed_changed_and_late_checks_are_not_fills(prospective, monkeypatch):
    c = prospective
    launch(c, monkeypatch)
    enroll(c, monkeypatch)
    for status, extra in [
        ("checked_unavailable", {}),
        ("inconclusive", {}),
        (
            "checked_available",
            {"selection": "Away", "line": "not_applicable", "odds": 110, "available_stake_usd": 10},
        ),
    ]:
        command(
            c,
            monkeypatch,
            "CHECK",
            {
                "enrollment_id": "one",
                "status": status,
                "reason": "retain all failures and changed price",
                "checked_at_utc": "2030-01-02T00:03:00Z",
                "evidence_reference": "synthetic quote check",
                **extra,
            },
            "2030-01-02T00:04:00Z",
        )
    settle(c, monkeypatch)
    command(
        c,
        monkeypatch,
        "CHECK",
        {
            "enrollment_id": "one",
            "status": "checked_available",
            "reason": "late attestation is not decision-time verification",
            "checked_at_utc": "2030-01-02T00:03:00Z",
            "selection": "Away",
            "line": "not_applicable",
            "odds": 120,
            "available_stake_usd": 10,
            "evidence_reference": "late evidence",
        },
        "2030-01-02T20:02:00Z",
    )
    result = build_prospective_tables(read_experiment_events(c["root"]), "2030-01-03T00:00:00Z")
    assert not result["prospective_observations"].original_quote_available.any()
    verified = result["prospective_metrics"].query("cohort == 'original_quote_checked_available'")
    assert verified.settled_wagers.sum() == 0
    assert (
        result["prospective_metrics"]
        .query("arm == 'subgroup' and cohort == 'all_captured_hypothetical'")
        .iloc[0]
        .settled_wagers
        == 1
    )


def test_superseded_approval_blocks_new_capture(prospective, monkeypatch):
    c = prospective
    launch(c, monkeypatch)
    record_decision(
        c["decisions"],
        c["proposal"],
        c["proposal_id"],
        "SUPERSEDED",
        "actor",
        "withdrawn approval",
        "APPROVED",
        c["approval"]["event_id"],
    )
    with pytest.raises(ValueError, match="superseded"):
        enroll(c, monkeypatch)


def test_review_requires_due_date_and_current_report(prospective, tmp_path, monkeypatch):
    c = prospective
    launch(c, monkeypatch)
    enroll(c, monkeypatch)
    from test_atlas_bundle import canonical

    import oddsjam_ev.io.experiment_register as register

    source = tmp_path / "canonical-source"
    source.mkdir()
    snapshot = canonical(source)
    monkeypatch.setattr(register, "utc_now", lambda: "2030-02-04T00:00:00Z")
    report = publish_experiment_report(
        c["root"], snapshot, tmp_path / "reports", "2030-02-04T00:00:00Z"
    )
    payload = {
        "scheduled_review_utc": "2030-02-04T00:00:00Z",
        "review_assessment_reference": str(report),
        "evidence_reference": "review missing data",
        "notes": "Inconclusive",
    }
    with pytest.raises(ValueError, match="due predeclared"):
        command(c, monkeypatch, "REVIEW", payload, "2030-02-03T23:59:00Z")
    command(
        c,
        monkeypatch,
        "CHECK",
        {
            "enrollment_id": "one",
            "status": "inconclusive",
            "reason": "unavailable evidence",
            "checked_at_utc": "2030-02-04T00:00:00Z",
            "evidence_reference": "failure retained",
        },
        "2030-02-04T00:01:00Z",
    )
    with pytest.raises(ValueError, match="Stale"):
        command(c, monkeypatch, "REVIEW", payload, "2030-02-04T00:02:00Z")
    # Rehashing cached dashboard attachments catches changed Parquet bytes.
    (report / "prospective_metrics.parquet").write_bytes(b"corrupt")
    cfg = DashboardConfig(
        evidence_root=str(tmp_path),
        prospective_report_publication=str(report.relative_to(tmp_path)),
    )
    with pytest.raises(ValueError, match="integrity"):
        load_prospective_report(cfg, {"events": [c["definition"]]})


def render_prospective_fixture():
    """Render immutable evidence without reading a live register or writing events."""
    import streamlit as st

    from oddsjam_ev.dashboard.pages.paper_tests import paper_tests

    definitions, report = UI_FIXTURE
    st.session_state.paper_definition_snapshot = definitions
    st.session_state.prospective_report_snapshot = report
    paper_tests(None, "all", 100)
