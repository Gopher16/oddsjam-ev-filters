"""Information-availability gates, event purging and outcome-isolated families."""

import json

import pandas as pd
from discovery_fixtures import wagers
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.walk_forward import EVIDENCE_FLAGS, build_walk_forward


def temporal_fixture():
    x = wagers(6)
    x["game_id"] = [f"g{i}" for i in range(6)]
    x["created_at_utc"] = pd.to_datetime(
        ["2026-01-02", "2026-01-03", "2026-02-02", "2026-02-03", "2026-03-02", "2026-03-03"],
        utc=True,
    )
    x["features_available_at_utc"] = x.created_at_utc - pd.Timedelta(hours=1)
    x["outcome_available_at_utc"] = x.created_at_utc + pd.Timedelta(hours=2)
    x["information_end_utc"] = x.outcome_available_at_utc
    cfg = DiscoveryConfig(
        views=(("sportsbook",),),
        embargo_days=1,
        temporal_windows=(
            {
                "fold_id": "one",
                "development_start": "2026-01-01T00:00Z",
                "selection_start": "2026-02-01T00:00Z",
                "test_start": "2026-03-01T00:00Z",
                "test_end": "2026-04-01T00:00Z",
            },
        ),
    )
    evidence = {
        **dict.fromkeys(EVIDENCE_FLAGS, True),
        "source_reference": "synthetic_fixture_only",
        "observed_as_of_utc": "2026-05-01T00:00Z",
    }
    return x, cfg, evidence


def test_complete_gate_and_no_outcome_leakage():
    x, cfg, evidence = temporal_fixture()
    a = build_walk_forward(x, cfg, evidence)
    assert a["temporal_gates"].iloc[0].status == "available_conditional_on_evidence"
    assert a["temporal_partitions"].partition.value_counts().to_dict() == {
        "development": 2,
        "selection": 2,
        "test": 2,
    }
    y = x.copy()
    y.loc[2:, "reported_profit_usd"] = 99999
    b = build_walk_forward(y, cfg, evidence)
    assert_frame_equal(a["frozen_candidates"], b["frozen_candidates"])
    assert_frame_equal(a["temporal_partitions"], b["temporal_partitions"])
    assert_frame_equal(
        a["temporal_metrics"].query("partition == 'development'").reset_index(drop=True),
        b["temporal_metrics"].query("partition == 'development'").reset_index(drop=True),
    )
    # New categories in the holdout do not become a discovered candidate.
    y.loc[4:, "sportsbook"] = "New holdout category"
    c = build_walk_forward(y, cfg, evidence)
    assert_frame_equal(a["frozen_candidates"], c["frozen_candidates"])


def test_event_crossing_embargo_and_delayed_outcome():
    x, cfg, evidence = temporal_fixture()
    x.loc[0, "game_id"] = x.loc[2, "game_id"]
    x.loc[1, "information_end_utc"] = pd.Timestamp("2026-02-01T00:00Z")
    x.loc[3, "outcome_available_at_utc"] = pd.Timestamp("2026-03-02T00:00Z")
    x.loc[3, "information_end_utc"] = x.loc[3, "outcome_available_at_utc"]
    result = build_walk_forward(x, cfg, evidence)
    reasons = result["temporal_partitions"].reason.value_counts().to_dict()
    assert reasons["event_spans_partition_boundary"] == 2
    assert reasons["information_overlap_or_embargo"] == 2
    assert result["frozen_candidates"].empty


def test_invalid_naive_missing_and_unverified_time():
    x, cfg, evidence = temporal_fixture()
    for modified in [
        x.drop(columns="outcome_available_at_utc"),
        x.assign(created_at_utc=pd.NaT),
        x.assign(created_at_utc="2026-01-02"),
    ]:
        result = build_walk_forward(modified, cfg, evidence)
        assert result["temporal_gates"].iloc[0].status == "blocked"
        assert result["temporal_partitions"].empty
    result = build_walk_forward(x, cfg, {})
    assert "placement_time_unverified" in json.loads(result["temporal_gates"].iloc[0].reasons_json)


def test_whole_event_quarantine_on_unavailable_feature():
    x, cfg, evidence = temporal_fixture()
    x.loc[1, "game_id"] = x.loc[0, "game_id"]
    x.loc[0, "features_available_at_utc"] = x.loc[0, "created_at_utc"] + pd.Timedelta(hours=1)
    result = build_walk_forward(x, cfg, evidence)
    assert (
        result["temporal_partitions"].iloc[:2].reason.eq("invalid_or_unavailable_information").all()
    )


def test_multiple_expanding_origins_and_stability_accounting():
    x, cfg, evidence = temporal_fixture()
    first = cfg.temporal_windows[0]
    second = {
        "fold_id": "two",
        "development_start": first["development_start"],
        "selection_start": "2026-03-01T00:00Z",
        "test_start": "2026-04-01T00:00Z",
        "test_end": "2026-05-01T00:00Z",
    }
    cfg = DiscoveryConfig(views=cfg.views, embargo_days=1, temporal_windows=(first, second))
    result = build_walk_forward(x, cfg, evidence)
    assert len(result["temporal_partitions"]) == 2 * len(x)
    assert result["temporal_stability"].groupby("fold_id").eligible_count.sum().to_dict() == {
        "one": 6,
        "two": 6,
    }
    assert set(result["temporal_stability"].calendar_month_utc) == {"2026-01", "2026-02", "2026-03"}
    overlap = dict(second, test_start="2026-03-15T00:00Z")
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        DiscoveryConfig(temporal_windows=(first, overlap))


def test_incomplete_test_window_never_presents_full_validation():
    x, cfg, evidence = temporal_fixture()
    evidence["observed_as_of_utc"] = "2026-03-15T00:00Z"
    result = build_walk_forward(x, cfg, evidence)
    audit = result["temporal_fold_audit"].iloc[0]
    assert audit.fold_status == "blocked_incomplete_or_empty_partition"
    assert audit.excluded_count == len(x) and audit.reconciles
    assert result["temporal_partitions"].reason.eq("incomplete_test_window").all()
    assert result["temporal_metrics"].empty
