"""Synthetic discovery contracts; no private data or network required."""

import numpy as np
from discovery_fixtures import policy, wagers
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.segment_search import enumerate_segments
from oddsjam_ev.io.discovery import build_discovery


def test_complete_ledger_independent_of_outcomes():
    x = wagers()
    x.loc[:9, "reported_roi_eligible"] = False
    a, links = enumerate_segments(x, policy())
    y = x.iloc[::-1].copy()
    y["reported_profit_usd"] = 999
    y["reported_roi_eligible"] = False
    b, other = enumerate_segments(y, policy())
    assert_frame_equal(a, b)
    assert_frame_equal(links, other)
    result = build_discovery(x, policy())
    assert len(result["candidate_results"]) == len(a) == 2
    assert result["candidate_results"].raw_pvalue.isna().all()
    assert result["candidate_results"].test_status.str.contains("stake_threshold_unset").all()
    assert not result["candidate_results"].promotion_allowed.any()
    assert result["temporal_gates"].iloc[0].status == "blocked"
    assert result["temporal_metrics"].empty


def test_empty_and_unsupported_populations():
    x = wagers(0)
    result = build_discovery(x, policy())
    assert result["candidate_ledger"].empty
    x = wagers()
    x["reported_roi_eligible"] = False
    result = build_discovery(x, policy())
    assert result["candidate_results"].eligible_count.eq(0).all()
    assert result["candidate_results"].reported_roi.isna().all()


def test_overlapping_full_search_nulls_and_planted_segment():
    cfg = DiscoveryConfig(
        min_event_groups=20,
        min_eligible_stake_usd=0,
        bootstrap_resamples=100,
        views=(("sportsbook",), ("market_family",), ("sportsbook", "market_family")),
    )
    rng = np.random.default_rng(901)
    rejections = 0
    for _ in range(40):
        x = wagers(240)
        group_index = np.arange(240) // 2
        x["sportsbook"] = np.where(group_index % 2, "A", "B")
        x["market_family"] = np.where(group_index % 3, "spread", "moneyline")
        # Shared event outcomes and unequal stakes; all views reuse the same pool.
        stake = rng.uniform(5, 50, 120)
        x["reported_stake_usd"] = np.repeat(stake / 2, 2)
        x["reported_profit_usd"] = np.repeat(rng.normal(0, 1, 120) * stake / 2, 2)
        results = build_discovery(x, cfg)["candidate_results"]
        assert results.family_size.eq(len(results)).all()
        rejections += int(results.exploratory_holm_reject.any())
    assert rejections <= 8
    x.loc[x.sportsbook.eq("A"), "reported_profit_usd"] += 100
    results = build_discovery(x, cfg)["candidate_results"]
    planted = results.loc[results.predicate_json.eq('{"sportsbook": "A"}')].iloc[0]
    assert planted.exploratory_holm_reject
    assert not results.promotion_allowed.any()
