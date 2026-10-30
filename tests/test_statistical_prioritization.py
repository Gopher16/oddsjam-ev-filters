"""Approved gates, paired clusters, complete correction and presentation-only overlap."""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_sizing_diagnostics import inputs

from oddsjam_ev.analysis.multidimensional_config import MultidimensionalConfig
from oddsjam_ev.analysis.multidimensional_discovery import build_within_filter_discovery
from oddsjam_ev.analysis.paired_comparison import PrioritizationBudgetExceeded, paired_comparison
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.analysis.signal_prioritization import build_prioritization
from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig, build_sizing_diagnostics


def prepared():
    # Each event includes both platforms/arms; strong but nondegenerate contrast.
    count = 12
    stakes = tuple([10.0] * (2 * count))
    profits = tuple(v for i in range(count) for v in (4.0 + i / 20, -2.0 + i / 30))
    source, link = inputs(stakes, profits)
    bets = source["bet_facts"]
    bets["percentage"] = "4"
    bets["game_id"] = [f"g{i // 2}" for i in range(2 * count)]
    dims = bets[["bet_id"]].assign(
        sportsbook=["A", "B"] * count, sport=["sportA", "sportB"] * count
    )
    sizing_cfg = SizingDiagnosticsConfig(min_event_groups=2, bootstrap_resamples=100)
    data = build_sizing_diagnostics(source, link, sizing_cfg)["candidate_diagnostics"]
    tables, _ = build_within_filter_discovery(
        source,
        dims,
        data,
        MultidimensionalConfig(views=[["sportsbook"], ["sport"]]),
        sizing_cfg,
        {"source": "synthetic"},
    )
    cfg = PrioritizationConfig(
        min_eligible_wagers=2,
        min_event_groups=2,
        max_wager_stake_share=0.9,
        max_event_stake_share=0.9,
        bootstrap_resamples=200,
    )
    return tables, data, cfg


def test_paired_contrast_and_all_family_slots_preserved_after_suppression():
    family, data, cfg = prepared()
    before = copy.deepcopy(family)
    tables, audit = build_prioritization(family, data, cfg)
    assert audit["hypothesis_count"] == 4 * len(family["signal_candidates"])
    assert audit["family_preserved_after_suppression"]
    assert audit["suppressed_equivalent_candidates"] == 2
    assert tables["signal_hypotheses"].family_size.eq(16).all()
    assert tables["signal_statistics"].test_status.eq("approximate_conditional_exploratory").all()
    assert set(tables["signal_priorities"].category) == {
        "exploratory_positive_signal",
        "exploratory_loss_pocket",
    }
    assert len(tables["signal_shortlist"]) == 2
    assert not tables["signal_statistics"].validated.any()
    assert not tables["signal_statistics"].production_authorized.any()
    assert "equivalent" in set(tables["signal_overlap"].relationship)
    for key in family:
        assert_frame_equal(family[key], before[key])
    # Permuting sources cannot alter Monte Carlo draws or representative identity.
    shuffled = {key: value.iloc[::-1] for key, value in family.items()}
    again, other = build_prioritization(shuffled, data.iloc[::-1], cfg)
    assert audit == other
    for key in tables:
        assert_frame_equal(tables[key], again[key])


def test_missing_coverage_and_all_blocked_hypotheses_reserved():
    family, data, cfg = prepared()
    data.loc[0, "game_id"] = ""
    tables, audit = build_prioritization(family, data, cfg)
    assert audit["blocked_hypotheses"] == audit["hypothesis_count"] == 16
    assert tables["signal_hypotheses"].raw_pvalue.isna().all()
    assert tables["signal_hypotheses"].holm_pvalue.isna().all()
    assert tables["signal_hypotheses"].family_size.eq(16).all()
    assert tables["signal_shortlist"].empty
    assert (
        tables["signal_statistics"]
        .gate_reasons_json.str.contains("incomplete_event_coverage")
        .all()
    )


def test_arm_gates_and_scale_independence():
    family, data, cfg = prepared()
    subgroup = data.iloc[::2]
    complement = data.iloc[1::2]
    result, p = paired_comparison(subgroup, complement, cfg, 42)
    assert result["test_status"] == "approximate_conditional_exploratory"
    scaled = data.copy()
    scaled[["stake", "profit"]] *= 100
    other, q = paired_comparison(scaled.iloc[::2], scaled.iloc[1::2], cfg, 42)
    assert result["weighted_difference"] == pytest.approx(other["weighted_difference"])
    assert p["weighted"]["positive"] == pytest.approx(q["weighted"]["positive"])
    gate, _ = paired_comparison(subgroup, complement, replace(cfg, min_eligible_wagers=100), 42)
    assert "insufficient_wagers" in gate["gate_reasons_json"]
    concentrated = subgroup.copy()
    concentrated.loc[concentrated.index[0], "stake"] = 10000
    gate, _ = paired_comparison(concentrated, complement, cfg, 42)
    assert "concentration" in gate["gate_reasons_json"]


def test_shared_event_not_execution_iid_and_degenerate_inputs():
    family, data, cfg = prepared()
    sub = data.iloc[::2].copy()
    comp = data.iloc[1::2].copy()
    result, _ = paired_comparison(sub, comp, cfg, 42)
    assert result["joint_event_count"] == 12
    assert result["bootstrap_cells"] == 12 * cfg.bootstrap_resamples
    sub["unit_return"] = 0.5
    sub["profit"] = 5.0
    comp["unit_return"] = 0.1
    comp["profit"] = 1.0
    result, p = paired_comparison(sub, comp, cfg, 42)
    assert result["test_status"] == "blocked_degenerate_comparison"
    assert all(np.isnan(v) for measure in p.values() for v in measure.values())
    with pytest.raises(ValueError, match="disjoint"):
        paired_comparison(sub, sub, cfg, 42)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_event_coverage": 0.9},
        {"alpha": 0},
        {"bootstrap_resamples": 1},
        {"max_event_stake_share": 1},
        {"min_eligible_stake_usd": 10000},
    ],
)
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        PrioritizationConfig(**kwargs)


def test_budgets_fail_before_family_publication():
    family, data, cfg = prepared()
    with pytest.raises(PrioritizationBudgetExceeded):
        build_prioritization(family, data, replace(cfg, max_pair_comparisons=1))
    with pytest.raises(PrioritizationBudgetExceeded):
        build_prioritization(family, data, replace(cfg, max_bootstrap_cells=1))


def test_shared_event_shocks_cancel_in_paired_contrast():
    family, data, cfg = prepared()
    arm_a = data.bet_id.str[1:].astype(int).mod(2).eq(0)
    baseline, _ = paired_comparison(data.loc[arm_a], data.loc[~arm_a], cfg, 42)
    noisy = data.copy()
    shocks = np.where(data.game_id.str[1:].astype(int).mod(2), 1.0, -1.0)
    noisy["unit_return"] += shocks
    noisy["profit"] += noisy.stake * shocks
    paired, _ = paired_comparison(noisy.loc[arm_a], noisy.loc[~arm_a], cfg, 42)
    assert paired["weighted_standard_error"] == pytest.approx(baseline["weighted_standard_error"])
    assert paired["weighted_ci_low"] == pytest.approx(baseline["weighted_ci_low"])
    assert paired["weighted_ci_high"] == pytest.approx(baseline["weighted_ci_high"])


def test_cross_parent_equivalence_is_audited_not_suppressed_together():
    family, data, cfg = prepared()
    family = copy.deepcopy(family)
    for key in ("signal_candidates", "signal_comparisons"):
        clone = family[key].copy()
        clone["candidate_id"] += ":other"
        clone["parent_id"] += ":other"
        clone["filter_id"] = "other"
        family[key] = pd.concat([family[key], clone], ignore_index=True)
    clone = family["signal_membership"].copy()
    clone["scope_id"] += ":other"
    family["signal_membership"] = pd.concat([family["signal_membership"], clone], ignore_index=True)
    tables, audit = build_prioritization(family, data, cfg)
    assert audit["hypothesis_count"] == 32
    assert tables["signal_overlap"].same_parent.eq(False).any()
    lookup = tables["signal_priorities"].set_index("candidate_id")
    for row in tables["signal_priorities"].itertuples():
        assert lookup.loc[row.representative_id, "parent_id"] == row.parent_id
