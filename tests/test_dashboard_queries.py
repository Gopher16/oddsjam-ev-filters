"""Published cohort equality, execution separation and null/overlap semantics."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from oddsjam_ev.dashboard import queries
from oddsjam_ev.dashboard.artifacts import load_published_run


def test_query_values_equal_published_artifacts(dashboard_run):
    run = load_published_run(*dashboard_run)
    queries.validate_display_contract(run.tables)
    for mode in queries.MODES:
        expected = run.tables["canonical"]["economic_summary"]
        expected = expected.loc[expected.execution_status.eq(mode), queries.ECONOMICS]
        assert_frame_equal(queries.economics(run.tables, mode), expected)
        for view in run.tables["atlas"]["wager_segments"].view.unique():
            result = queries.atlas_view(run.tables, mode, view)
            source = run.tables["atlas"]["wager_segments"]
            source = source.loc[
                source.execution_status.eq(mode) & source.view.eq(view), result.columns
            ]
            assert_frame_equal(result, source)
        p = queries.proposals(run.tables, mode)
        assert p.proposal_id.is_monotonic_increasing
        assert not set(p.columns) & {"notes", "tags"}
    with pytest.raises(ValueError):
        queries.economics(run.tables, None)
    with pytest.raises(ValueError):
        queries.atlas_view(run.tables, "paper", "invented")


def test_excluded_cohort_null_roi_and_no_cross_mode_sum():
    frame = pd.DataFrame(
        [
            {
                "product": "fixed_odds",
                "execution_status": mode,
                "candidate_count": 1,
                "eligible_count": 0,
                "eligible_stake_usd": None,
                "eligible_reported_profit_usd": None,
                "reported_roi": None,
                "currency": "USD",
                "fee_inclusion": "unverified",
            }
            for mode in queries.MODES
        ]
    )
    tables = {"canonical": {"economic_summary": frame}}
    for mode in queries.MODES:
        selected = queries.economics(tables, mode)
        assert len(selected) == 1
        assert selected.reported_roi.isna().all()
        assert selected.execution_status.eq(mode).all()


def test_nonadditive_views_and_display_queries_do_not_mutate(dashboard_run):
    run = load_published_run(*dashboard_run)
    original = run.tables["atlas"]["wager_segments"].copy(deep=True)
    for mode in queries.MODES:
        assert not queries.filter_diagnostics(run.tables, mode).additive_across_filters.any()
        result = queries.atlas_view(run.tables, mode, "sportsbook")
        result.loc[:, "candidate_count"] = 999
        proxies = queries.atlas_view(run.tables, mode, "sportsbook", proxies=True)
        assert not any(
            "profit" in col or "stake_usd" in col or "reported_roi" in col for col in proxies
        )
    assert_frame_equal(original, run.tables["atlas"]["wager_segments"])
    assert queries.experiment_metrics(run.tables, "paper").empty
    run.tables["canonical"]["economic_summary"].drop(columns="reported_roi", inplace=True)
    with pytest.raises(ValueError, match="schema"):
        queries.validate_display_contract(run.tables)
