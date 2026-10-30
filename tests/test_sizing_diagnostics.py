"""Paired economic estimands, concentration and conditional group uncertainty."""

import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig, build_sizing_diagnostics


def inputs(stakes=(10.0, 100.0), profits=(10.0, -20.0)):
    n = len(stakes)
    bets = pd.DataFrame(
        {
            "bet_id": [f"b{i}" for i in range(n)],
            "product": ["exchange"] * n,
            "execution_status": ["real_money"] * n,
            "reported_roi_eligible": [True] * n,
            "reported_stake_usd": stakes,
            "reported_profit_usd": profits,
            "economic_exclusions_json": ["[]"] * n,
            "namespace": ["synthetic"] * n,
            "game_id": [f"g{i}" for i in range(n)],
        }
    )
    bridge = pd.DataFrame(
        {"bet_id": bets.bet_id, "filter_id": ["a"] * n, "attribution_status": ["resolved"] * n}
    )
    linkage = pd.DataFrame(
        {
            "bet_id": bets.bet_id,
            "product": bets["product"],
            "execution_status": bets.execution_status,
            "source_reported_stake_usd": stakes,
            "linkage_status": ["unknown_placement"] * n,
            "applied_policy_id": [None] * n,
            "bankroll_amount": [None] * n,
        }
    )
    return {"bet_facts": bets, "filter_attribution": bridge}, linkage


def overall(tables, linkage, cfg=None):
    result = build_sizing_diagnostics(
        tables, linkage, cfg or SizingDiagnosticsConfig(min_event_groups=2, bootstrap_resamples=100)
    )
    return result, result["cohort_metrics"].query("view == 'overall'").iloc[0]


def test_sign_reversal_paired_cohort_concentration_and_scaling():
    tables, linkage = inputs()
    original = copy.deepcopy(tables)
    result, row = overall(tables, linkage)
    assert row.stake_weighted_roi == pytest.approx(-10 / 110)
    assert row.canonical_reported_roi == row.stake_weighted_roi
    assert row.equal_stake_mean_return == pytest.approx(0.4)
    assert row.fixed_unit_notional_profit_usd == pytest.approx(0.8)
    assert row.weighted_equal_sign_disagreement
    assert row.largest_stake_share == pytest.approx(100 / 110)
    assert row.stake_hhi == pytest.approx((10 / 110) ** 2 + (100 / 110) ** 2)
    assert row.largest_absolute_profit_share == pytest.approx(20 / 30)
    for key in tables:
        assert_frame_equal(tables[key], original[key])
    scaled = copy.deepcopy(tables)
    scaled["bet_facts"][["reported_stake_usd", "reported_profit_usd"]] *= 10
    link = linkage.copy()
    link.source_reported_stake_usd *= 10
    _, other = overall(scaled, link)
    assert other.stake_weighted_roi == row.stake_weighted_roi
    assert other.equal_stake_mean_return == row.equal_stake_mean_return
    assert other.reported_profit_usd == row.reported_profit_usd * 10
    same, link = inputs((10.0, 10.0), (5.0, -10.0))
    _, row = overall(same, link)
    assert row.stake_weighted_roi == row.equal_stake_mean_return


def test_regimes_only_verified_application_and_filter_views_nonadditive():
    tables, link = inputs()
    link.loc[0, ["applied_policy_id", "linkage_status", "bankroll_amount"]] = [
        "p1",
        "application_verified",
        "1000",
    ]
    result, row = overall(tables, link)
    regimes = result["cohort_metrics"].query("view == 'regime'")
    assert set(regimes.sizing_regime) == {"p1", "<unknown>"}
    assert regimes.eligible_stake_usd.sum() == row.eligible_stake_usd
    assert row.bankroll_relative_available_count == 1
    assert row.bankroll_relative_stake_mean == 0.01
    assert result["candidate_diagnostics"].bankroll_relative_stake.isna().sum() == 1
    extra = tables["filter_attribution"].copy()
    extra.filter_id = "b"
    tables["filter_attribution"] = pd.concat([tables["filter_attribution"], extra])
    result, row = overall(tables, link)
    assert row.candidate_count == 2
    assert result["cohort_metrics"].query("view == 'filter'").candidate_count.sum() == 4
    link.loc[0, "linkage_status"] = "matched_setting_interval"
    with pytest.raises(ValueError, match="not verified"):
        overall(tables, link)


def test_exclusions_missing_events_and_supported_products():
    tables, link = inputs((0.0, 10.0, 10.0, 10.0), (0.0, np.nan, 5.0, -5.0))
    tables["bet_facts"].loc[2, "game_id"] = ""
    tables["bet_facts"].loc[3, "reported_roi_eligible"] = False
    tables["bet_facts"].loc[3, "economic_exclusions_json"] = '["unsupported_settlement"]'
    result, row = overall(tables, link)
    assert row.diagnostic_eligible_count == 1
    assert row.additional_excluded_count == 2
    assert row.event_missing_count == 1
    assert row.event_proxy_count == 0
    assert np.isnan(row.weighted_roi_ci_low)
    assert row.uncertainty_status == "insufficient_event_proxy_groups"
    assert row.equal_stake_mean_return == 0.5
    assert (
        result["candidate_diagnostics"]
        .diagnostic_exclusions_json.str.contains("canonical_ineligible")
        .sum()
        == 1
    )


def test_event_intervals_use_same_covered_subset_and_are_deterministic():
    tables, link = inputs((10.0, 20.0, 30.0, 40.0), (5.0, -20.0, 15.0, -4.0))
    tables["bet_facts"].loc[3, "game_id"] = ""
    result, row = overall(tables, link)
    assert row.event_covered_count == 3
    assert row.event_missing_count == 1
    assert row.event_covered_equal_stake_mean_return == pytest.approx(0)
    assert row.event_covered_stake_weighted_roi == pytest.approx(0)
    assert row.interval_cohort_fingerprint != row.cohort_fingerprint
    assert row.uncertainty_status == "conditional_event_proxy_intervals"
    repeat, _ = overall({key: frame.iloc[::-1] for key, frame in tables.items()}, link.iloc[::-1])
    assert_frame_equal(result["cohort_metrics"], repeat["cohort_metrics"])
    # Repeated candidates in one event are resampled as one cluster.
    tables["bet_facts"]["game_id"] = "same"
    _, row = overall(tables, link)
    assert row.event_proxy_count == 1 and np.isnan(row.weighted_roi_ci_low)


@pytest.mark.parametrize("change", ["duplicate", "stake", "scope", "population", "bridge"])
def test_invalid_provenance(change):
    tables, link = inputs()
    if change == "duplicate":
        tables["bet_facts"] = pd.concat([tables["bet_facts"], tables["bet_facts"]])
    elif change == "stake":
        link.loc[0, "source_reported_stake_usd"] = 11
    elif change == "scope":
        link.loc[0, "product"] = "fixed_odds"
    elif change == "population":
        link = link.iloc[:1]
    else:
        tables["filter_attribution"] = pd.concat([tables["filter_attribution"]] * 2)
    with pytest.raises(ValueError):
        overall(tables, link)


def test_empty_and_invalid_config():
    tables, link = inputs((), ())
    result = build_sizing_diagnostics(tables, link, SizingDiagnosticsConfig())
    assert result["cohort_metrics"].empty
    assert "equal_stake_mean_return" in result["cohort_metrics"]
    for kwargs in (
        {"fixed_unit_usd": 0},
        {"seed": -1},
        {"bootstrap_resamples": 10},
        {"confidence_level": 1},
        {"min_event_groups": 1},
    ):
        with pytest.raises(ValueError):
            SizingDiagnosticsConfig(**kwargs)
