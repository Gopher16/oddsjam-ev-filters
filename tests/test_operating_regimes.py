"""Operating evidence must never invent chronology, fills or canonical economics."""

import copy

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_sizing_config import evidence
from test_sizing_config import history as sizing_history

from oddsjam_ev.analysis.operating_regimes import (
    build_regime_linkage,
    migrate_evidence,
    regime_record,
    validate_regimes,
)
from oddsjam_ev.io.source_csv import fingerprint


def registry():
    return {
        "schema_version": 1,
        "snapshot_date": "2026-11-24",
        "filters": [{"id": "core", "name": "TEST - Core", "former_names": ["PROD - Core"]}],
    }


def history():
    row = regime_record("r1", "synthetic", "2026-11-24T00:00:00Z", "synthetic evidence")
    row.update(
        filter_id="core",
        precision="timestamp",
        effective_from="2026-11-10T12:00:00Z",
        effective_to="2026-11-20T12:00:00Z",
        effective_interval_verified=True,
        evidence_type="user_confirmed",
        confidence="confirmed",
        funding="paper",
        selection_mode="discretionary",
        paper_sizing="kelly_recommendation",
    )
    return dict(
        contract_version=1,
        knowledge_as_of="2026-11-24T00:00:00Z",
        previous_history_id=None,
        records=[row],
        wager_evidence=[],
    )


def inputs(times):
    bets = pd.DataFrame(
        [
            dict(
                bet_id=f"b{i}",
                namespace="synthetic",
                product="exchange",
                execution_status="paper",
                reported_stake_usd=100,
                reported_profit_usd=-100,
                created_at_utc=t,
            )
            for i, t in enumerate(times)
        ]
    )
    attrs = pd.DataFrame(
        [dict(bet_id=b, filter_id="core", attribution_status="resolved") for b in bets.bet_id]
    )
    return bets, attrs


def placed(h, times):
    h["wager_evidence"] = [
        dict(
            evidence_id=f"w{i}",
            bet_id=f"b{i}",
            placement_at=t,
            placement_verified=True,
            recorded_at=h["knowledge_as_of"],
            source_reference="receipt",
        )
        for i, t in enumerate(times)
    ]


def test_exact_boundary_and_no_created_fallback_or_money_mutation():
    times = ["2026-11-10T11:59:59Z", "2026-11-10T12:00:00Z", "2026-11-20T12:00:00Z"]
    b, a = inputs(times)
    before = b.copy(deep=True)
    h = history()
    unknown, _ = build_regime_linkage(b, a, h, registry())
    assert unknown.linkage_status.tolist() == ["unknown_placement"] * 3
    placed(h, times)
    frame, audit = build_regime_linkage(b, a, h, registry())
    assert frame.linkage_status.tolist() == ["unknown_regime", "matched", "unknown_regime"]
    assert frame.paper_sizing.tolist() == ["unknown", "kelly_recommendation", "unknown"]
    assert "reported_stake_usd" not in frame and "accepted_stake" not in frame
    assert audit["accepted_fills_not_inferred"]
    assert_frame_equal(b, before)


def test_date_only_unknown_end_and_overlapping_contradictions():
    b, a = inputs(["2026-11-11T12:00:00Z"])
    h = history()
    placed(h, [b.created_at_utc.iloc[0]])
    r = h["records"][0]
    r.update(
        precision="date",
        effective_from="2026-11-10",
        effective_to=None,
        effective_interval_verified=False,
    )
    validate_regimes(h, registry())
    frame, audit = build_regime_linkage(b, a, h, registry())
    assert frame.linkage_status.iloc[0] == "unknown_regime" and audit["unlinked_record_ids"] == [
        "r1"
    ]
    r["effective_interval_verified"] = True
    with pytest.raises(ValueError, match="[Vv]erified"):
        validate_regimes(h, registry())
    h = history()
    placed(h, [b.created_at_utc.iloc[0]])
    other = copy.deepcopy(h["records"][0])
    other.update(evidence_id="r2", funding="real_money", paper_sizing="unknown")
    h["records"].append(other)
    frame, _ = build_regime_linkage(b, a, h, registry())
    assert frame.linkage_status.iloc[0] == "ambiguous_regime"
    assert frame.funding_evidence.iloc[0] == "unknown"
    h["records"] = [other]
    assert build_regime_linkage(b, a, h, registry())[0].linkage_status.iloc[0] == "funding_conflict"


def test_scope_placement_conflict_revision_and_knowledge():
    b, a = inputs(["2026-11-11T12:00:00Z"])
    h = history()
    placed(h, [b.created_at_utc.iloc[0]])
    h["records"][0]["namespace"] = "different"
    assert build_regime_linkage(b, a, h, registry())[0].linkage_status.iloc[0] == "unknown_regime"
    h["records"][0]["namespace"] = "synthetic"
    a["filter_id"] = None
    a["attribution_status"] = "unresolved"
    assert build_regime_linkage(b, a, h, registry())[0].linkage_status.iloc[0] == "unknown_regime"
    w = copy.deepcopy(h["wager_evidence"][0])
    w.update(evidence_id="w2", placement_at="2026-11-12T00:00:00Z")
    h["wager_evidence"].append(w)
    assert (
        build_regime_linkage(b, a, h, registry())[0].linkage_status.iloc[0]
        == "conflicting_placement"
    )
    newer = copy.deepcopy(h)
    newer["previous_history_id"] = fingerprint(h)
    validate_regimes(newer, registry(), h)
    newer["records"][0]["selection_mode"] = "broad_capture"
    with pytest.raises(ValueError, match="append-only"):
        validate_regimes(newer, registry(), h)
    h["knowledge_as_of"] = "2026-11-01T00:00:00Z"
    with pytest.raises(ValueError, match="cutoff"):
        validate_regimes(h, registry())


@pytest.mark.parametrize(
    "update",
    [
        {"effective_from": "2026-11-10T00:00:00"},
        {"filter_id": "PROD - Core"},
        {"paper_sizing": "kelly_recommendation", "funding": "real_money"},
        {"venue_access": ["Book", "Book"]},
        {"settings": {"bankroll": {"state": "unset", "value": 0, "reason": "inactive"}}},
        {"open_ended": 1},
        {"accepted_stake": "50"},
    ],
)
def test_malformed_evidence_fails_closed(update):
    h = history()
    h["records"][0].update(update)
    with pytest.raises(ValueError):
        validate_regimes(h, registry())


def test_legacy_sizing_adapter_preserves_unknown_process_and_input():
    legacy = sizing_history()
    legacy["wager_evidence"] = [evidence()]
    before = copy.deepcopy(legacy)
    migrated = migrate_evidence(registry(), "2026-11-24T00:00:00Z", sizing=legacy)
    assert legacy == before
    assert migrated["records"][0]["selection_mode"] == "unknown"
    assert migrated["records"][0]["funding"] == "unknown"
    assert migrated["records"][0]["paper_sizing"] == "unknown"
    assert "legacy-sizing:" + fingerprint(legacy) in migrated["records"][0]["source_reference"]


def test_prospective_alias_migration_stays_planned_and_unknown_namespace():
    baseline = dict(
        baseline_id="old",
        filter_id="PROD - Core",
        kind="prospective_parent",
        recorded_at="2026-11-24T00:00:00Z",
        effective_from="2026-11-24T00:00:00Z",
        effective_until=None,
        evidence_reference="synthetic",
        settings={},
    )
    old = dict(contract_version=1, baselines=[baseline], lifecycle=[], translations=[])
    migrated = migrate_evidence(registry(), "2026-11-24T00:00:00Z", decision=old)
    assert migrated["records"][0]["filter_id"] == "core"
    assert migrated["records"][0]["evidence_type"] == "planned"
    assert migrated["records"][0]["effective_interval_verified"] is False
    baseline["filter_id"] = "unknown label"
    with pytest.raises(ValueError, match="alias"):
        migrate_evidence(registry(), "2026-11-24T00:00:00Z", decision=old)


def test_record_kinds_cannot_be_swapped_between_arrays():
    h = history()
    placed(h, ["2026-11-11T12:00:00Z"])
    h["records"], h["wager_evidence"] = h["wager_evidence"], h["records"]
    with pytest.raises(ValueError, match="Exact"):
        validate_regimes(h, registry())


@pytest.mark.parametrize(
    "change", [{"product": "fixed_odds"}, {"evidence_type": "planned"}, {"confidence": "assumed"}]
)
def test_ineligible_scope_or_assumptions_never_link(change):
    b, a = inputs(["2026-11-11T12:00:00Z"])
    h = history()
    placed(h, [b.created_at_utc.iloc[0]])
    h["records"][0].update(change)
    assert build_regime_linkage(b, a, h, registry())[0].linkage_status.iloc[0] == "unknown_regime"


def test_unknown_funding_stays_canonical_unknown_and_empty_candidates_work():
    b, a = inputs(["2026-11-11T12:00:00Z"])
    b["execution_status"] = "unknown"
    h = history()
    placed(h, [b.created_at_utc.iloc[0]])
    before = b.copy(deep=True)
    frame, _ = build_regime_linkage(b, a, h, registry())
    assert frame.funding_evidence.iloc[0] == "paper"
    assert_frame_equal(b, before)
    h["wager_evidence"] = []
    frame, audit = build_regime_linkage(b.iloc[:0], a.iloc[:0], h, registry())
    assert frame.empty and audit["candidate_count"] == 0
    placed(h, ["2026-11-11T12:00:00Z"])
    with pytest.raises(ValueError, match="unknown canonical"):
        build_regime_linkage(b.iloc[:0], a.iloc[:0], h, registry())
