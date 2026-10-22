"""Synthetic grains, overlap, missingness, ticket and economic accounting tests."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.analysis.opportunity_atlas import build_atlas, summarize_wagers


def fixture():
    bets = pd.DataFrame(
        [
            dict(
                bet_id="b1",
                sportsbook="Book",
                sport="basketball",
                league="NBA",
                market_name="Moneyline",
                odds="120",
                percentage="5",
                execution_status="paper",
                product="fixed_odds",
                reported_roi_eligible=True,
                reported_stake_usd=10.0,
                reported_profit_usd=12.0,
                fact_grain="source_single_candidate",
                game_id="g",
            ),
            dict(
                bet_id="b2",
                sportsbook="Book",
                sport="basketball",
                league="NBA",
                market_name="Moneyline",
                odds="120",
                percentage="",
                execution_status="unknown",
                product="fixed_odds",
                reported_roi_eligible=False,
                reported_stake_usd=90.0,
                reported_profit_usd=99.0,
                fact_grain="ticket_candidate",
                game_id="",
            ),
        ]
    )
    sources = pd.DataFrame(
        [
            dict(
                _ingest_observation_id=f"o{i}",
                sportsbook="Book",
                sport="basketball",
                league="NBA",
                market_name="Moneyline",
                odds="120",
                percentage="5",
            )
            for i in range(3)
        ]
    )
    links = pd.DataFrame(
        [
            dict(bet_id=b, source_observation_id=f"o{i}", opportunity_proxy_id=p)
            for i, (b, p) in enumerate([("b1", "p1"), ("b2", "p1"), ("b2", "p2")])
        ]
    )
    opportunities = pd.DataFrame(
        [
            dict(
                opportunity_proxy_id=p,
                selection_proxy_id="s",
                identity_quality="event_selection_proxy",
            )
            for p in ["p1", "p2"]
        ]
    )
    attribution = pd.DataFrame(
        [
            dict(bet_id=b, filter_id=f, attribution_status="resolved")
            for b, f in [("b1", "a"), ("b1", "a"), ("b1", "b"), ("b2", "a")]
        ]
    )
    return dict(
        bet_facts=bets,
        source_features=sources,
        bet_opportunity_links=links,
        opportunity_facts=opportunities,
        filter_attribution=attribution,
    )


def test_accounting_and_overlap():
    tables = build_atlas(fixture(), AtlasConfig())
    overall = tables["wager_segments"].query("view == 'overall'")
    assert overall.eligible_count.sum() == 1
    assert overall.eligible_stake_usd.sum() == 10
    assert overall.eligible_reported_profit_usd.sum() == 12
    empty = overall.query("execution_status == 'unknown'").iloc[0]
    assert pd.isna(empty.reported_roi) and pd.isna(empty.eligible_stake_usd)
    assert empty.ticket_candidate_count == 1
    assert len(tables["opportunity_dimensions"]) == 2
    assert not any("profit" in c or "stake" in c for c in tables["opportunity_segments"])
    f = tables["filter_segment_diagnostics"].query(
        "view == 'filter_id__attribution_status' and execution_status == 'paper'"
    )
    assert f.eligible_stake_usd.tolist() == [10.0, 10.0]
    assert not f.additive_across_filters.any()
    p = tables["opportunity_segments"].query("view == 'overall'")
    assert p.opportunity_proxy_count.sum() == 3  # same proxy in two execution modes
    for view in ["sportsbook", "market_family", "advertised_ev_band"]:
        assert (
            tables["wager_segments"]
            .loc[tables["wager_segments"].view.eq(view)]
            .eligible_stake_usd.sum()
            == 10
        )


def test_conflicts_reorder_and_orphans():
    t = fixture()
    t["source_features"].loc[1, "league"] = "Other"
    result = build_atlas(t, AtlasConfig())
    assert "<conflict>" in set(result["opportunity_dimensions"].league)
    shuffled = {k: v.iloc[::-1].reset_index(drop=True) for k, v in t.items()}
    rebuilt = build_atlas(shuffled, AtlasConfig())
    for name in result:
        assert_frame_equal(result[name], rebuilt[name])
    t["bet_opportunity_links"].loc[0, "bet_id"] = "orphan"
    with pytest.raises(ValueError, match="orphan"):
        build_atlas(t, AtlasConfig())


def test_missing_source_rejected():
    t = fixture()
    t["source_features"] = t["source_features"].iloc[:1]
    with pytest.raises(ValueError, match="reconcile"):
        build_atlas(t, AtlasConfig())


def test_empty_input():
    t = {k: v.iloc[:0] for k, v in fixture().items()}
    result = build_atlas(t, AtlasConfig())
    assert all(v.empty for v in result.values())


def test_weighted_roi_and_invalid_drilldown():
    t = fixture()
    bets = t["bet_facts"]
    bets.loc[1, "execution_status"] = "paper"
    bets.loc[1, "fact_grain"] = "source_single_candidate"
    bets.loc[1, "reported_roi_eligible"] = True
    bets.loc[1, "reported_profit_usd"] = -90.0
    result = build_atlas(t, AtlasConfig())
    overall = result["wager_segments"].query("view == 'overall'").iloc[0]
    assert overall.eligible_stake_usd == 100
    assert overall.eligible_reported_profit_usd == -78
    assert overall.reported_roi == pytest.approx(-0.78)
    assert overall.largest_stake_share == pytest.approx(0.9)
    with pytest.raises(ValueError, match="dimensions"):
        summarize_wagers(result["wager_dimensions"], ["reported_profit_usd"])
