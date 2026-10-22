"""Stable targets, conservative translation and unique-wager coverage accounting."""

import pandas as pd
import pytest

from oddsjam_ev.analysis.filter_actions import coverage_impact, resolve_target, translate_predicate


def registry():
    return {
        "schema_version": 1,
        "snapshot_date": "2026-10-22",
        "filters": [
            {
                "id": "test_id",
                "name": "PROD - Named differently",
                "status": "PROD",
                "lifecycle": "existing",
            },
            {
                "id": "planned",
                "name": "TEST - Planned",
                "status": "TEST",
                "lifecycle": "planned_not_yet_confirmed_created",
            },
        ],
    }


def test_stable_ids_and_planned_targets():
    assert resolve_target("test_id", registry())["target_status"] == "PROD"
    assert resolve_target("test_id", registry())["target_existing"]
    assert not resolve_target("planned", registry())["target_existing"]
    with pytest.raises(ValueError):
        resolve_target("missing", registry())


def test_unmapped_and_half_open_boundaries_do_not_translate():
    cap = {"schema_version": 1, "exact_values": {}}
    assert (
        translate_predicate({"odds_band": "[100,150)"}, cap)["translation_status"]
        == "unavailable_or_partial"
    )
    assert (
        "diagnostic_category"
        in translate_predicate({"market_family": "<unmapped>"}, cap)["translation_blockers_json"]
    )


def test_overlap_never_multiplies_wagers_or_stake():
    bets = pd.DataFrame(
        {
            "bet_id": ["a", "b", "c"],
            "reported_roi_eligible": [True, True, False],
            "reported_stake_usd": [10.0, 30.0, 100.0],
        }
    )
    bridge = pd.DataFrame(
        [
            {"bet_id": b, "filter_id": f, "attribution_status": "resolved"}
            for b, f in [("a", "one"), ("a", "one"), ("a", "two"), ("b", "one"), ("c", "one")]
        ]
    )
    result = coverage_impact(bets, bridge, {"a"}, "one")
    assert result["target_observed_wagers"] == 3
    assert result["matching_target_wagers"] == 1
    assert result["nonmatching_target_wagers"] == 2
    assert result["target_eligible_stake_usd"] == 40
    assert result["matching_target_stake_share"] == 0.25
    assert result["other_filter_overlap_wagers"] == 1
    assert not result["future_scanner_coverage_verified"]
