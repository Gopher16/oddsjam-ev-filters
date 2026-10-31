"""Full evidence families, transparent priorities, distinct ideas and PROD safeguards."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from test_decision_records import STAMP, configuration
from test_statistical_prioritization import prepared

from oddsjam_ev.analysis.decision_config import CONTROLS, validate_evidence
from oddsjam_ev.analysis.decision_records import resolve_decision
from oddsjam_ev.analysis.review_catalogue import ReviewBudgetExceeded, enumerate_global
from oddsjam_ev.analysis.review_config import ReviewConfig
from oddsjam_ev.analysis.review_queues import build_review_queues
from oddsjam_ev.analysis.review_ranking import _fronts, classify
from oddsjam_ev.analysis.signal_prioritization import build_prioritization


def review_inputs(prod=False):
    family, data, cfg = prepared()
    priority, _ = build_prioritization(family, data, cfg)
    evidence = {"contract_version": 1, "baselines": [], "lifecycle": [], "translations": []}
    if prod:
        candidate = family["signal_candidates"].iloc[0].to_dict()
        evidence = configuration(candidate)
        evidence["baselines"][0]["settings"]["sportsbooks"]["value"] = ["A", "B"]
        evidence["lifecycle"] = [
            {
                "filter_id": candidate["filter_id"],
                "status": "PROD",
                "recorded_at": STAMP,
                "effective_from": STAMP,
                "effective_until": None,
                "evidence_reference": "Synthetic confirmation",
                "actor": "Synthetic operator",
                "confirmation": "operator_confirmed",
            }
        ]
    normalized = validate_evidence(evidence)
    records = [
        resolve_decision(row, normalized, design, STAMP, STAMP)
        for row in family["signal_candidates"].to_dict("records")
        for design in ("subgroup_tracking", "exclusion_comparison")
    ]
    decisions = {
        "manifest": {
            "specification": {"evidence": evidence, "as_of": STAMP, "effective_at": STAMP}
        },
        "records": records,
    }
    visual = {
        "visual_candidates": data.assign(
            sportsbook=["A", "B"] * 12,
            sport=["sportA", "sportB"] * 12,
            league="League",
            market_family="spread",
            advertised_ev_percent_exact="4",
            odds=100,
        )
    }
    return family, priority, decisions, visual


def test_corrected_family_preserved_and_unknown_prod_does_not_become_leak():
    sources = review_inputs()
    before = copy.deepcopy(sources)
    result, audit = build_review_queues(*sources, ReviewConfig())
    old = [r for r in result["records"] if r["source_kind"] == "within_filter_corrected_family"]
    assert len(old) == len(sources[0]["signal_candidates"])
    assert audit["source_hypothesis_count"] == 16
    assert all(r["evidence_tier"] == "corrected_historical_comparison" for r in old)
    assert audit["prospectively_validated_count"] == 0
    assert audit["production_proposal_count"] == 0
    assert all(not r["validated"] and not r["production_authorized"] for r in result["records"])
    assert any(r["direction"] == "loss_association" and r["queue"] == "investigations" for r in old)
    assert sum(r["headline"] for r in old) == 1  # shared events do not fill four cards
    assert any(
        r["suppression_reason"] == "exact_parent_subgroup_equivalence_presentation_only"
        for r in old
    )
    assert all(
        len(r["configuration_sheet"]["settings"]) == len(CONTROLS) for r in result["records"]
    )
    for a, b in zip(sources[:2], before[:2], strict=True):
        for name in a:
            assert_frame_equal(a[name], b[name])
    assert sources[2] == before[2]
    assert_frame_equal(sources[3]["visual_candidates"], before[3]["visual_candidates"])
    json.dumps(result, allow_nan=False)
    global_rows = [
        r for r in result["records"] if r["source_kind"] == "global_descriptive_catalogue"
    ]
    assert global_rows
    assert all(
        r["source_correction_family_id"] is None and not r["original_hypotheses"]
        for r in global_rows
    )
    assert all(r["evidence_tier"] == "insufficient_or_conflicting" for r in global_rows)
    assert all(r["configuration_sheet"]["baseline"] is None for r in global_rows)


def test_confirmed_production_requires_pinned_baseline_and_keeps_not_change_sheet():
    sources = review_inputs(prod=True)
    result, audit = build_review_queues(*sources, ReviewConfig())
    changes = [r for r in result["records"] if r["queue"] == "production_changes"]
    assert changes and audit["production_proposal_count"] == len(changes)
    for row in changes:
        proposal = row["production_proposal"]
        assert proposal["baseline_sha256"]
        assert row["configuration_sheet"]["design"] == "exclusion_comparison"
        assert row["setup_status"] == proposal["implementation_status"]
        assert "compound_exclusion_requires_verified_implementation" in row["setup_blockers"]
        assert proposal["logical_change"]["operator"] == "NOT"
        assert (
            proposal["removed_candidate_count"] + proposal["retained_candidate_count"]
            == row["historical_economics"]["parent"]["candidate_count"]
        )
        assert proposal["implementation_status"] == "blocked"
        assert not proposal["production_change_authorized"]
        assert proposal["profit_saved_or_forecast"] is None
    # Confirmed lifecycle alone cannot create a pinned configuration proposal.
    sources[2]["records"] = copy.deepcopy(sources[2]["records"])
    for row in sources[2]["records"]:
        row["baseline"] = None
    result, _ = build_review_queues(*sources, ReviewConfig())
    assert not any(r["queue"] == "production_changes" for r in result["records"])


def test_unconfirmed_comparison_stays_exploratory_and_no_forced_winners():
    sources = review_inputs()
    h = sources[1]["signal_hypotheses"]
    h["exploratory_holm_reject"] = False
    result, _ = build_review_queues(*sources, ReviewConfig())
    old = [r for r in result["records"] if r["source_kind"] == "within_filter_corrected_family"]
    assert all(r["evidence_tier"] == "support_qualified_exploratory" for r in old)
    assert all(r["future_profit_forecast"] is None for r in old)
    sources[1]["signal_statistics"]["test_status"] = "blocked_policy_gates"
    result, _ = build_review_queues(*sources, ReviewConfig())
    assert not any(r["headline"] for r in result["records"])


def test_shuffle_is_reproducible_and_global_enumeration_is_outcome_independent():
    sources = review_inputs()
    cfg = ReviewConfig()
    expected, audit = build_review_queues(*sources, cfg)
    family, priority, decisions, visual = sources
    again, other = build_review_queues(
        {k: v.iloc[::-1] for k, v in family.items()},
        {k: v.iloc[::-1] for k, v in priority.items()},
        {**decisions, "records": decisions["records"][::-1]},
        {"visual_candidates": visual["visual_candidates"].iloc[::-1]},
        cfg,
    )
    assert again == expected and audit == other
    original = enumerate_global(visual["visual_candidates"], cfg)
    changed = visual["visual_candidates"].copy()
    changed["profit"] = np.arange(len(changed)) * 10000
    assert enumerate_global(changed, cfg) == original


def test_pareto_components_and_sizing_conflict_policy():
    rows = [
        {"candidate_id": name, "v": value}
        for name, value in [
            ("a", (1, 100, -0.1)),
            ("b", (2, 90, -0.1)),
            ("c", (0, 80, -0.2)),
            ("d", (1, 100, -0.1)),
        ]
    ]
    assert _fronts(rows, lambda r: r["v"]) == {"a": 1, "b": 1, "d": 1, "c": 2}
    stat = {
        "subgroup_weighted_roi": -0.1,
        "subgroup_equal_stake_return": 0.0,
        "complement_weighted_roi": 0.1,
        "complement_equal_stake_return": 0.1,
        "test_status": "approximate_conditional_exploratory",
    }
    assert classify(stat, True) == ("sizing_or_estimand_conflict", "insufficient_or_conflicting")


def test_complete_family_and_budget_fail_closed():
    sources = review_inputs()
    sources[1]["signal_hypotheses"] = sources[1]["signal_hypotheses"].iloc[:-1]
    with pytest.raises(ValueError, match="four-slot"):
        build_review_queues(*sources, ReviewConfig())
    with pytest.raises(ReviewBudgetExceeded):
        build_review_queues(*review_inputs(), replace(ReviewConfig(), max_pairs=1))
    with pytest.raises(ReviewBudgetExceeded):
        build_review_queues(*review_inputs(), replace(ReviewConfig(), max_candidates=1))
    with pytest.raises(ValueError):
        ReviewConfig(global_views=(("liquidity",),))


def test_global_supported_associations_have_no_invented_tests_and_populations_stay_separate():
    sources = list(review_inputs())
    original = sources[3]["visual_candidates"]
    copies = []
    for i in range(400):
        row = original.iloc[i % 2].copy()
        row["bet_id"] = f"new{i}"
        row["game_id"] = f"new_event{i}"
        row["execution_status"] = "paper"
        row["profit"] = 2.0 if i % 2 == 0 else -1.0
        row["stake"] = 10.0
        row["unit_return"] = row["profit"] / row["stake"]
        copies.append(row)
    import pandas as pd

    sources[3] = {
        "visual_candidates": pd.concat([original, pd.DataFrame(copies)], ignore_index=True)
    }
    result, _ = build_review_queues(*sources, ReviewConfig())
    qualified = [
        r
        for r in result["records"]
        if r["source_kind"] == "global_descriptive_catalogue"
        and r["direction"] == "opportunity"
        and r["execution_status"] == "paper"
    ]
    assert qualified
    for row in qualified:
        assert row["evidence_tier"] == "support_qualified_exploratory"
        assert row["rank_components"]["conservative_pointwise_contrast"] is None
        assert not row["original_hypotheses"]
        assert not any(row["corrected_direction_rejected_both_measures"].values())
        assert row["configuration_sheet"]["logical_change"]["scope"] == "new_prospective_universe"
    assert sum(p["candidate_count"] for p in result["universe"]["populations"]) == 424
    assert {p["execution_status"] for p in result["universe"]["populations"]} == {
        "real_money",
        "paper",
    }
