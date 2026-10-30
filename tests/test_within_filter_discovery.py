"""Bounded outcome-independent families and exact within-filter partitions."""

from dataclasses import asdict

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_sizing_diagnostics import inputs

from oddsjam_ev.analysis.multidimensional_config import MultidimensionalConfig
from oddsjam_ev.analysis.multidimensional_discovery import (
    DiscoveryBudgetExceeded,
    build_within_filter_discovery,
)
from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig, build_sizing_diagnostics


def fixture():
    source, link = inputs((10.0, 20.0, 30.0, 40.0), (10.0, -10.0, 10.0, -20.0))
    source["bet_facts"]["percentage"] = ["2.999", "3", "3.001", "4"]
    source["filter_attribution"] = pd.DataFrame(
        {
            "bet_id": ["b0", "b1", "b2", "b3", "b3"],
            "filter_id": ["a", "a", "a", "b", "a"],
            "attribution_status": ["resolved"] * 5,
        }
    )
    dims = source["bet_facts"][["bet_id"]].assign(
        sportsbook=["Novig", "Novig", "Other", "Other"],
        sport=["basketball"] * 4,
        market_family=["player_prop"] * 4,
    )
    cfg = SizingDiagnosticsConfig(min_event_groups=2, bootstrap_resamples=100)
    data = build_sizing_diagnostics(source, link, cfg)["candidate_diagnostics"]
    return source, dims, data, cfg


def run(cfg=None):
    source, dims, data, sizing_cfg = fixture()
    policy = cfg or MultidimensionalConfig(views=(("sportsbook", "advertised_ev_percent"),))
    return build_within_filter_discovery(
        source, dims, data, policy, sizing_cfg, {"canonical": "synthetic"}
    )


def test_exact_parent_subgroup_complement_and_empty_candidates():
    tables, audit = run()
    assert len(tables["signal_candidates"]) == 3
    assert tables["signal_candidates"].observed_subgroup_count.eq(0).sum() == 1
    assert audit["all_partitions_reconcile"] and not audit["outcome_screening_performed"]
    comparisons = tables["signal_comparisons"]
    assert (
        comparisons.subgroup_count + comparisons.complement_count == comparisons.parent_count
    ).all()
    assert set(comparisons.support_status) == {
        "empty_subgroup",
        "observed_descriptive",
        "empty_complement",
    }
    assert set(tables["signal_evidence"].uncertainty_status) == {"comparison_inference_deferred"}
    assert tables["signal_evidence"].weighted_roi_ci_low.isna().all()
    b = comparisons.loc[comparisons.filter_id.eq("b")].iloc[0]
    assert b.parent_count == 1 and b.subgroup_count == 1
    parent = tables["signal_evidence"].query("role == 'parent' and sizing_regime.isna()")
    assert sorted(parent.candidate_count) == [1, 4]


def test_semantic_order_threshold_normalization_and_row_order_stable():
    a = MultidimensionalConfig(
        views=[["sportsbook", "advertised_ev_percent"]],
        ev_thresholds=[{"operator": ">", "endpoint": "3.00", "units": "percentage_points"}],
    )
    b = MultidimensionalConfig(views=[["advertised_ev_percent", "sportsbook"]])
    assert asdict(a) == asdict(b)
    first, audit = run(a)
    second, other = run(b)
    assert audit == other
    for name in first:
        assert_frame_equal(first[name], second[name])
    source, dims, data, cfg = fixture()
    shuffled, _ = build_within_filter_discovery(
        {k: f.iloc[::-1] for k, f in source.items()},
        dims.iloc[::-1],
        data.iloc[::-1],
        a,
        cfg,
        {"canonical": "synthetic"},
    )
    for name in first:
        assert_frame_equal(first[name], shuffled[name])


@pytest.mark.parametrize(
    "kwargs",
    [{"max_candidates": 1}, {"max_parents": 1}, {"max_memberships": 1}, {"max_scan_rows": 1}],
)
def test_budget_fails_closed(kwargs):
    with pytest.raises(DiscoveryBudgetExceeded) as error:
        run(MultidimensionalConfig(views=[["sportsbook"]], **kwargs))
    assert error.value.audit["status"] == "aborted_budget_exceeded_not_truncated"


def test_outcomes_do_not_change_enumeration():
    source, dims, data, cfg = fixture()
    policy = MultidimensionalConfig(views=[["sportsbook"]])
    first, audit = build_within_filter_discovery(
        source, dims, data, policy, cfg, {"canonical": "same"}
    )
    source["bet_facts"]["reported_profit_usd"] *= 2
    data["reported_profit_usd"] *= 2
    data["profit"] *= 2
    data["unit_return"] *= 2
    second, other = build_within_filter_discovery(
        source, dims, data, policy, cfg, {"canonical": "same"}
    )
    assert audit == other
    assert_frame_equal(first["signal_candidates"], second["signal_candidates"])
    assert_frame_equal(first["signal_membership"], second["signal_membership"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"views": [["profit"]]},
        {"views": [["sportsbook"]] * 2},
        {"max_depth": 5},
        {"views": [["sportsbook", "sport"]], "max_depth": 1},
        {"ev_thresholds": [{"operator": ">", "endpoint": "3", "units": "fraction"}]},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        MultidimensionalConfig(**kwargs)


def test_missing_categories_remain_in_complement_and_regimes_retained():
    source, dims, data, cfg = fixture()
    dims.loc[0, "sportsbook"] = "<missing>"
    data.loc[1, "sizing_regime"] = "verified_policy"
    tables, audit = build_within_filter_discovery(
        source, dims, data, MultidimensionalConfig(views=[["sportsbook"]]), cfg, {"source": "test"}
    )
    assert audit["usage"]["excluded_missing_category_groups"] == 1
    assert "verified_policy" in set(tables["signal_evidence"].sizing_regime.dropna())
    assert tables["signal_comparisons"].partition_reconciles.all()


def test_greater_equal_boundary_four_dimensions_and_source_preservation():
    source, dims, data, cfg = fixture()
    originals = {k: v.copy(deep=True) for k, v in source.items()}
    dimensions_before = dims.copy(deep=True)
    diagnostics_before = data.copy(deep=True)
    policy = MultidimensionalConfig(
        views=[["sportsbook", "sport", "market_family", "advertised_ev_percent"]],
        ev_thresholds=[{"operator": ">=", "endpoint": "3", "units": "percentage_points"}],
    )
    tables, _ = build_within_filter_discovery(
        source, dims, data, policy, cfg, {"source": "synthetic"}
    )
    assert tables["signal_candidates"].predicate_depth.eq(4).all()
    a = (
        tables["signal_candidates"]
        .loc[lambda f: f.filter_id.eq("a") & f.predicate_json.str.contains("Novig")]
        .iloc[0]
    )
    assert a.observed_subgroup_count == 1  # exactly 3 matches >=3, unlike >3
    for key in source:
        assert_frame_equal(source[key], originals[key])
    assert_frame_equal(dims, dimensions_before)
    assert_frame_equal(data, diagnostics_before)


def test_missing_ev_and_unknown_only_categories_never_create_matches():
    source, dims, data, cfg = fixture()
    source["bet_facts"]["percentage"] = ["", "<conflict>", "bad", "4"]
    data["percentage"] = source["bet_facts"].percentage
    tables, audit = build_within_filter_discovery(
        source,
        dims,
        data,
        MultidimensionalConfig(views=[["advertised_ev_percent"]]),
        cfg,
        {"source": "missing-ev"},
    )
    assert set(audit["ev_input_status_counts_all_candidates"]) == {
        "valid",
        "missing",
        "conflict",
        "invalid",
    }
    a = tables["signal_comparisons"].loc[lambda f: f.filter_id.eq("a")].iloc[0]
    assert a.subgroup_count == 1 and a.complement_count == 3
    dims["sportsbook"] = "<missing>"
    with pytest.raises(ValueError, match="No observed predicates"):
        build_within_filter_discovery(
            source,
            dims,
            data,
            MultidimensionalConfig(views=[["sportsbook"]]),
            cfg,
            {"source": "empty-category"},
        )
