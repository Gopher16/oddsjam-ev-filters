"""Synthetic chart economics, coverage and source semantics; no notebook execution."""

import json
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from oddsjam_ev.analysis.visual_config import VisualConfig
from oddsjam_ev.analysis.visual_diagnostics import CHARTS, build_visual_diagnostics
from oddsjam_ev.analysis.visual_features import _date, _time, band, prepare_visuals


def inputs():
    facts = pd.DataFrame(
        {
            "bet_id": ["a", "b", "c", "d"],
            "namespace": ["source"] * 4,
            "game_id": ["1", "1", "2", "3"],
            "execution_status": ["real_money", "real_money", "paper", "real_money"],
            "product": ["exchange"] * 4,
            "sportsbook": ["A", "A", "B", "A"],
            "sport": ["basketball"] * 4,
            "league": ["NBA"] * 4,
            "percentage": ["3.0000000000000001", "", "2.5", "5"],
            "odds": [100, -200, 150, 100],
            "created_at": ["2026-01-05T23:00:00-06:00", "2026-01-07T12:00:00", "2026-01-05", ""],
            "event_start_date": ["2026-01-06T07:00:00Z"] * 4,
            "timestamp_timezone": ["unknown"] * 4,
            "liquidity": [500, None, 0, -1],
            "clv": [-200, 100, None, 100],
            "clv_source": ["fair", "untrusted", "", "fair"],
            "reported_stake_usd": [10.0, 90.0, 20.0, 0.0],
            "reported_profit_usd": [10.0, -45.0, 10.0, 100.0],
        }
    )
    dimensions = facts.assign(market_family="spread")
    diagnostics = facts.assign(
        diagnostic_eligible=[True, True, True, False],
        diagnostic_exclusions_json=["[]", "[]", "[]", '["stake"]'],
        stake=facts.reported_stake_usd,
        profit=facts.reported_profit_usd,
        unit_return=[1.0, -0.5, 0.5, np.nan],
        sizing_regime="<unknown>",
    )
    bridge = pd.DataFrame(
        {
            "bet_id": ["a", "a", "b", "c", "d"],
            "filter_id": ["first", "second", "first", None, "first"],
            "attribution_status": ["resolved", "resolved", "resolved", "missing", "resolved"],
        }
    )
    return {"bet_facts": facts, "filter_attribution": bridge}, dimensions, diagnostics


def test_matched_economics_population_separation_overlap_and_missingness():
    tables, audit = build_visual_diagnostics(*inputs(), VisualConfig())
    assert set(tables) == {
        "visual_candidates",
        "visual_memberships",
        "visual_cohorts",
        "visual_periods",
        "visual_activity",
        "visual_coverage",
        "visual_readiness",
    }
    overall = tables["visual_cohorts"].query("scope_kind == 'population' and view == 'overall'")
    assert overall.candidate_count.sum() == 4
    assert overall.eligible_count.sum() == 3
    real = overall.loc[overall.execution_status.eq("real_money")].iloc[0]
    assert real.reported_profit_usd == -35
    assert real.stake_weighted_roi == -0.35
    assert real.equal_stake_mean_return == 0.25
    assert real.fixed_unit_notional_profit_usd == 0.5
    assert real.ev_matched_count == 1
    assert real.ev_matched_roi == 1
    assert real.ev_mean_advertised_ev_fraction == pytest.approx(0.03)
    assert real.closing_matched_count == 0
    assert real.event_proxy_groups == 1
    assert not real.performance_display_eligible
    assert audit["candidate_count"] == 4
    assert audit["unresolved_or_missing_attribution_candidates"] == 1
    assert not audit["hypothesis_tests_performed"]
    readiness = tables["visual_readiness"]
    assert set(readiness.chart_id) == set(CHARTS)
    assert len(readiness) == audit["scope_count"] * len(CHARTS)
    blocked = readiness.loc[
        readiness.chart_id.isin(
            ["liquidity_summary", "liquidity_ev_summary", "time_to_event_summary"]
        )
    ]
    assert blocked.status.eq("blocked").all()
    assert blocked.remediation.str.len().gt(0).all()
    assert tables["visual_candidates"].liquidity_usd.isna().all()
    assert tables["visual_candidates"].closing_price_ev_fraction.isna().all()


def test_calendar_gaps_are_unknown_not_zero_and_activity_is_active_days():
    tables, _ = build_visual_diagnostics(*inputs(), VisualConfig())
    periods = tables["visual_periods"].query(
        "scope_kind == 'population' and execution_status == 'real_money' and resolution == 'D'"
    )
    periods = periods.loc[periods.platform.isna()].reset_index(drop=True)
    assert len(periods) == 3
    assert pd.isna(periods.iloc[1].candidate_count)
    assert pd.isna(periods.iloc[1].reported_profit_usd)
    assert periods.iloc[1].period_status == "unobserved_coverage_unknown"
    assert periods.iloc[-1].cumulative_attributed_reported_profit_usd == -35
    assert periods.iloc[0].period_start.startswith("2026-01-05")  # no UTC date shift
    activity = tables["visual_activity"]
    assert activity.active_source_day_denominator.eq(1).all()
    assert activity.denominator_semantics.str.contains("not_all_covered_days").all()
    assert set(tables["visual_periods"].resolution) == {"D", "W-SUN", "M"}


def test_verified_optional_features_and_joint_matched_economics():
    cfg = VisualConfig(
        placement_time_verified=True,
        event_time_verified=True,
        chronology_evidence="synthetic placement/event provenance",
        liquidity_usd_verified=True,
        liquidity_evidence="synthetic USD quotes",
        closing_fair_american_verified=True,
        closing_sources=("fair",),
        closing_evidence="synthetic fair closing prices",
    )
    work = prepare_visuals(*inputs(), cfg).set_index("bet_id")
    assert work.loc["a", "time_to_event_hours"] == 2
    assert pd.isna(work.loc["b", "time_to_event_hours"])
    assert work.loc["a", "liquidity_usd"] == 500
    assert work.loc["c", "liquidity_usd"] == 0
    assert pd.isna(work.loc["d", "liquidity_usd"])
    assert work.loc["a", "closing_price_ev_fraction"] == pytest.approx(1 / 3)
    assert pd.isna(work.loc["b", "closing_price_ev_fraction"])
    tables, _ = build_visual_diagnostics(*inputs(), cfg)
    cohort = (
        tables["visual_cohorts"]
        .query(
            "scope_kind == 'population' and execution_status == 'real_money' and view == 'overall'"
        )
        .iloc[0]
    )
    assert cohort.joint_matched_count == 1
    assert cohort.joint_matched_roi == 1
    assert cohort.joint_weighted_closing_price_ev_fraction == pytest.approx(1 / 3)


def test_exact_bands_invalid_dates_dst_and_after_start():
    assert band(Decimal("3.0000000000000001"), (0, 3, 5)) == "[3, 5)"
    assert band(Decimal("2.9999999999999999"), (0, 3, 5)) == "[0, 3)"
    assert band(-1, (0, 3)) == "[-inf, 0)"
    assert band(3, (0, 3)) == "[3, inf)"
    assert _date("10/30/2026")[2] == "unsupported_source_format"
    assert _date("2026-02-30")[2] == "invalid"
    assert _time("2026-11-01T01:30:00", "America/Chicago") is None
    assert _time("2026-03-08T02:30:00", "America/Chicago") is None
    assert _time("2026-01-01T12:00:00", None) is None
    source, dim, diag = inputs()
    for frame in (source["bet_facts"], dim, diag):
        frame.loc[0, "event_start_date"] = "2026-01-06T04:00:00Z"
    cfg = VisualConfig(
        placement_time_verified=True, event_time_verified=True, chronology_evidence="synthetic"
    )
    assert prepare_visuals(source, dim, diag, cfg).iloc[0].time_to_event_bucket == "after_start"


def test_input_drift_and_economic_tampering_fail():
    source, dim, diag = inputs()
    diag.loc[0, "stake"] = 999
    with pytest.raises(ValueError, match="economics"):
        prepare_visuals(source, dim, diag, VisualConfig())
    source, dim, diag = inputs()
    dim.loc[0, "bet_id"] = "b"
    with pytest.raises(ValueError, match="identities"):
        prepare_visuals(source, dim, diag, VisualConfig())
    source, dim, diag = inputs()
    diag.loc[0, "unit_return"] = 999
    with pytest.raises(ValueError, match="economics"):
        prepare_visuals(source, dim, diag, VisualConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"liquidity_usd_verified": True},
        {"closing_fair_american_verified": True},
        {"placement_time_verified": True},
        {"ev_edges": (3, 3)},
        {"max_rows": 0},
    ],
)
def test_unsupported_policy_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        VisualConfig(**kwargs)


def test_no_dated_rows_and_no_eligible_economics():
    source, dim, diag = inputs()
    for frame in (source["bet_facts"], dim, diag):
        frame["created_at"] = ""
    diag["diagnostic_eligible"] = False
    tables, _ = build_visual_diagnostics(
        source, dim, diag, replace(VisualConfig(), min_performance_wagers=1)
    )
    assert tables["visual_periods"].empty
    assert tables["visual_activity"].empty
    assert tables["visual_readiness"].status.eq("blocked").all()
    assert tables["visual_cohorts"].reported_profit_usd.isna().all()
    for value in tables["visual_cohorts"].category_json:
        json.loads(value)


def test_empty_source_drift_and_calendar_budget_are_explicit():
    source, dim, diag = inputs()
    source["bet_facts"] = source["bet_facts"].iloc[:0]
    with pytest.raises(ValueError, match="at least one"):
        build_visual_diagnostics(source, dim.iloc[:0], diag.iloc[:0], VisualConfig())
    source, dim, diag = inputs()
    dim.loc[0, "reported_profit_usd"] = 42
    with pytest.raises(ValueError, match="source field drift"):
        prepare_visuals(source, dim, diag, VisualConfig())
    from oddsjam_ev.analysis.visual_diagnostics import VisualBudgetExceeded

    with pytest.raises(VisualBudgetExceeded, match="calendar span"):
        build_visual_diagnostics(*inputs(), VisualConfig(max_calendar_days=1))
    with pytest.raises(VisualBudgetExceeded, match="membership budget"):
        build_visual_diagnostics(*inputs(), VisualConfig(max_scope_links=1))
    assert _date("9999-01-01")[2] == "invalid"
    assert _time("2026-01-01T12:00:00", "unknown-zone") is None
