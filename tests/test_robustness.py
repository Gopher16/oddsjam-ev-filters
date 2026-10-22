"""Synthetic discovery contracts; no private data or network required."""

import numpy as np
import pandas as pd
import pytest
from discovery_fixtures import policy, wagers

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.robustness import (
    cluster_bootstrap,
    evaluate_candidate,
    event_profit_pvalue,
    event_totals,
)


def test_missing_events_and_exact_economics():
    x = wagers()
    x.loc[0, "game_id"] = ""
    x.loc[1, "reported_roi_eligible"] = False
    r = evaluate_candidate(x, policy(), 42)
    e = x[x.reported_roi_eligible]
    assert r["eligible_count"] == 119 and r["missing_event_wagers"] == 1
    assert r["eligible_reported_profit_usd"] == pytest.approx(e.reported_profit_usd.sum())
    assert r["event_covered_profit_usd"] == pytest.approx(e.iloc[1:].reported_profit_usd.sum())
    assert r["interval_status"] == "conditional_exploratory"
    assert r == evaluate_candidate(x, policy(), 42)
    for k in ["cluster_roi_ci_low", "cluster_roi_ci_high", "leave_one_event_out_roi_min"]:
        assert r[k] == evaluate_candidate(x.iloc[::-1], policy(), 42)[k]


def test_group_pairing_degeneracy_and_leave_one_out():
    x = wagers(4)
    x["game_id"] = ["a", "a", "b", "b"]
    x["reported_stake_usd"] = [10, 20, 30, 40]
    x["reported_profit_usd"] = [1, 2, 3, 4]
    groups = event_totals(x)
    assert groups.stake.tolist() == [30, 70]
    assert groups.profit.tolist() == [3, 7]
    assert np.isnan(
        cluster_bootstrap(groups, DiscoveryConfig(min_event_groups=2, bootstrap_resamples=100), 4)[
            0
        ]
    )
    assert np.isnan(event_profit_pvalue(pd.DataFrame({"profit": [1, 1]})))
    x.loc[2:, "reported_profit_usd"] = -10
    r = evaluate_candidate(x, DiscoveryConfig(min_event_groups=2, bootstrap_resamples=100), 2)
    assert r["without_largest_stake_event_roi"] == pytest.approx(0.1)
    assert r["without_largest_positive_profit_event_roi"] == pytest.approx(-20 / 70)
    assert r["leave_one_event_out_roi_min"] == pytest.approx(-20 / 70)


def test_conditional_cluster_interval_coverage_on_synthetic_events():
    rng = np.random.default_rng(21)
    cfg = DiscoveryConfig(min_event_groups=100, bootstrap_resamples=500)
    covered = 0
    for _ in range(100):
        stakes = rng.uniform(5, 50, 150)
        groups = pd.DataFrame({"stake": stakes, "profit": stakes * rng.normal(0, 1, 150)})
        lo, hi = cluster_bootstrap(groups, cfg, int(rng.integers(0, 1000000)))
        covered += int(lo <= 0 <= hi)
    assert 0.85 <= covered / 100 <= 1
