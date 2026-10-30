"""
signal_prioritization.py
===============================================================================
Purpose
-------
Account for complete directional families before overlap-aware exploratory display.

What this module provides
-------------------------
Paired comparisons, four hypotheses per candidate, Holm accounting and full audits.

Design Notes
------------
Suppression is presentation-only. Approximate marginal assumptions, selection,
chronology and fillability limitations remain; no validated/production labels exist.
===============================================================================
"""

from dataclasses import asdict
from itertools import combinations

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.multiple_testing import adjust_family
from oddsjam_ev.analysis.paired_comparison import PrioritizationBudgetExceeded, paired_comparison
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.io.source_csv import fingerprint


def build_prioritization(
    discovery: dict, diagnostics: pd.DataFrame, cfg: PrioritizationConfig
) -> tuple[dict, dict]:
    """Evaluate every frozen candidate, correct all hypotheses, then curate display.

    Parameters
    ----------
    discovery : dict
        Complete verified within-filter publication with memberships/comparisons.
    diagnostics : pandas.DataFrame
        Matched verified candidate diagnostics with canonical eligibility preserved.
    cfg : PrioritizationConfig
        Explicit user-approved exploratory gates and correction/presentation policy.

    Returns
    -------
    tuple[dict, dict]
        Full evidence/hypotheses/overlap/priority tables, shortlist and family audit.
        Raw unavailable tests stay null while reserving their correction slots.
    """
    candidates = discovery["signal_candidates"].sort_values("candidate_id")
    members = discovery["signal_membership"]
    comparisons = discovery["signal_comparisons"].set_index("candidate_id")
    if (
        candidates.empty
        or candidates.candidate_id.duplicated().any()
        or candidates.family_id.nunique() != 1
    ):
        raise ValueError("One complete frozen candidate family required")
    if diagnostics.bet_id.duplicated().any() or diagnostics.bet_id.isna().any():
        raise ValueError("Unique diagnostic candidates required")
    if members.duplicated(["scope_id", "bet_id"]).any() or not set(members.bet_id) <= set(
        diagnostics.bet_id
    ):
        raise ValueError("Invalid prioritization membership")
    if (
        set(comparisons.index) != set(candidates.candidate_id)
        or comparisons.index.duplicated().any()
    ):
        raise ValueError("Comparison family mismatch")
    pair_count = len(candidates) * (len(candidates) - 1) // 2
    if pair_count > cfg.max_pair_comparisons:
        raise PrioritizationBudgetExceeded(
            {
                "reason": "overlap_pairs",
                "requested": pair_count,
                "maximum": cfg.max_pair_comparisons,
            }
        )
    data = diagnostics.set_index("bet_id", drop=False).rename_axis(None).sort_index()
    groups = {ident: set(part.bet_id) for ident, part in members.groupby("scope_id")}
    rawsets = {}
    eligsets = {}
    plans = []
    cache = {}
    cells = 0
    # Preflight all gates and unique bootstrap work before computing any tests.
    for row in candidates.itertuples():
        parent = groups.get(row.parent_id, set())
        selected = groups.get(row.candidate_id, set())
        complement = parent - selected
        check = comparisons.loc[row.candidate_id]
        if (
            not selected <= parent
            or len(parent) != check.parent_count
            or len(selected) != check.subgroup_count
            or len(complement) != check.complement_count
        ):
            raise ValueError("Prioritization partition mismatch")
        if not parent:
            raise ValueError("Missing frozen filter parent")
        if (
            data.loc[sorted(parent), "execution_status"].ne(row.execution_status).any()
            or data.loc[sorted(parent), "product"].ne(row.product).any()
        ):
            raise ValueError("Prioritization parent scope mismatch")
        rawsets[row.candidate_id] = selected
        sub = data.loc[sorted(selected)]
        comp = data.loc[sorted(complement)]
        eligible_a = sorted(sub.loc[sub.diagnostic_eligible, "bet_id"])
        eligible_b = sorted(comp.loc[comp.diagnostic_eligible, "bet_id"])
        eligsets[row.candidate_id] = set(eligible_a)
        key = fingerprint({"a": eligible_a, "b": eligible_b})
        seed = (cfg.seed + int(key[:8], 16)) % (2**32)
        if key not in cache:
            preflight, _ = paired_comparison(sub, comp, cfg, seed, evaluate=False)
            cells += preflight["bootstrap_cells"]
            if cells > cfg.max_bootstrap_cells:
                raise PrioritizationBudgetExceeded(
                    {
                        "reason": "total_bootstrap_cells",
                        "requested": cells,
                        "maximum": cfg.max_bootstrap_cells,
                    }
                )
            cache[key] = {"sub": sub, "comp": comp, "seed": seed, "preflight": preflight}
        plans.append((row, key))
    for value in cache.values():
        value["evidence"], value["tests"] = paired_comparison(
            value["sub"], value["comp"], cfg, value["seed"]
        )
    evidence = []
    hypotheses = []
    for row, key in plans:
        value = cache[key]
        result = value["evidence"]
        sub = value["sub"].loc[lambda f: f.diagnostic_eligible]
        evidence.append(
            {
                "candidate_id": row.candidate_id,
                "parent_id": row.parent_id,
                "filter_id": row.filter_id,
                "execution_status": row.execution_status,
                "product": row.product,
                "partition_fingerprint": key,
                **result,
                "subgroup_weighted_roi": float(sub.profit.sum() / sub.stake.sum())
                if len(sub)
                else np.nan,
                "subgroup_equal_stake_return": float(sub.unit_return.mean())
                if len(sub)
                else np.nan,
                "validated": False,
                "production_authorized": False,
            }
        )
        for measure in ("weighted", "equal_stake"):
            for direction in ("positive", "negative"):
                ident = "hypothesis:" + fingerprint(
                    {"candidate": row.candidate_id, "measure": measure, "direction": direction}
                )
                hypotheses.append(
                    {
                        "candidate_id": ident,
                        "signal_candidate_id": row.candidate_id,
                        "measure": measure,
                        "direction": direction,
                        "raw_pvalue": value["tests"][measure][direction],
                        "test_status": result["test_status"],
                    }
                )
    correction = "correction:" + fingerprint(
        {
            "frozen_family": candidates.family_id.iloc[0],
            "policy": asdict(cfg),
            "hypotheses_per_candidate": 4,
        }
    )
    corrected = adjust_family(pd.DataFrame(hypotheses), cfg.alpha).rename(
        columns={"candidate_id": "hypothesis_id"}
    )
    corrected["family_id"] = correction
    corrected["placeholder_in_correction"] = corrected.raw_pvalue.isna()
    stats = pd.DataFrame(evidence).set_index("candidate_id", drop=False)
    priorities = []
    for row in candidates.itertuples():
        result = stats.loc[row.candidate_id]
        tests = corrected.loc[corrected.signal_candidate_id.eq(row.candidate_id)]

        def rejects(direction, tests=tests):
            return bool(tests.loc[tests.direction.eq(direction), "exploratory_holm_reject"].all())

        w, u = result.weighted_difference, result.equal_stake_difference
        if result.test_status != "approximate_conditional_exploratory":
            category = "insufficient_evidence"
        elif np.sign(w) != np.sign(u):
            category = "estimand_disagreement"
        elif w > 0 and u > 0 and rejects("positive"):
            category = (
                "exploratory_positive_signal"
                if result.subgroup_weighted_roi > 0 and result.subgroup_equal_stake_return > 0
                else "exploratory_relative_association"
            )
        elif w < 0 and u < 0 and rejects("negative"):
            category = (
                "exploratory_loss_pocket"
                if result.subgroup_weighted_roi < 0 and result.subgroup_equal_stake_return < 0
                else "exploratory_relative_association"
            )
        else:
            category = "exploratory_unconfirmed"
        priorities.append(
            {
                "candidate_id": row.candidate_id,
                "parent_id": row.parent_id,
                "category": category,
                "interval_available": bool(
                    np.isfinite([result.weighted_ci_low, result.equal_stake_ci_low]).all()
                ),
                "min_arm_event_groups": min(result.subgroup_groups, result.complement_groups),
                "min_arm_eligible_wagers": min(
                    result.subgroup_eligible_count, result.complement_eligible_count
                ),
                "worst_event_share": max(
                    result.subgroup_largest_event_share, result.complement_largest_event_share
                ),
                "worst_wager_share": max(
                    result.subgroup_largest_wager_share, result.complement_largest_wager_share
                ),
                "representative_id": row.candidate_id,
                "suppression_reason": None,
                "included_shortlist": False,
                "ordering_reason": "interval availability; event breadth; eligible count; lower event/wager concentration; stable ID; no effect-size or specificity bonus",
                "interpretation": "exploratory_not_holdout_validated_no_filter_edit_authority",
            }
        )
    priority = (
        pd.DataFrame(priorities)
        .sort_values(
            [
                "category",
                "interval_available",
                "min_arm_event_groups",
                "min_arm_eligible_wagers",
                "worst_event_share",
                "worst_wager_share",
                "candidate_id",
            ],
            ascending=[True, False, False, False, True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    lookup = priority.set_index("candidate_id")
    overlap = []
    ids = sorted(rawsets)
    for a, b in combinations(ids, 2):
        sa, sb = rawsets[a], rawsets[b]
        shared = len(sa & sb)
        if not shared:
            continue
        relation = (
            "equivalent"
            if sa == sb
            else "left_nested"
            if sa < sb
            else "right_nested"
            if sb < sa
            else "partial_overlap"
        )
        same_parent = lookup.loc[a, "parent_id"] == lookup.loc[b, "parent_id"]
        overlap.append(
            {
                "left_candidate_id": a,
                "right_candidate_id": b,
                "shared_wagers": shared,
                "union_wagers": len(sa | sb),
                "jaccard": shared / len(sa | sb),
                "shared_eligible_wagers": len(eligsets[a] & eligsets[b]),
                "relationship": relation,
                "same_parent": same_parent,
                "correction_family_unchanged": True,
            }
        )
    # Exact equivalence is transitive; select one representative within parent/category.
    representatives = {}
    for i, row in priority.iterrows():
        key = (row.parent_id, row.category, frozenset(rawsets[row.candidate_id]))
        if key in representatives and rawsets[row.candidate_id]:
            priority.loc[i, "representative_id"] = representatives[key]
            priority.loc[i, "suppression_reason"] = (
                "equivalent_membership_same_parent_category_presentation_only"
            )
        else:
            representatives[key] = row.candidate_id
    approved_categories = {
        "exploratory_positive_signal",
        "exploratory_loss_pocket",
        "exploratory_relative_association",
    }
    counts = {}
    for i, row in priority.iterrows():
        if row.category in approved_categories and pd.isna(row.suppression_reason):
            counts[row.category] = counts.get(row.category, 0) + 1
            if counts[row.category] <= cfg.max_shortlist_per_category:
                priority.loc[i, "included_shortlist"] = True
    priority["presentation_order"] = np.arange(1, len(priority) + 1)
    overlap_columns = [
        "left_candidate_id",
        "right_candidate_id",
        "shared_wagers",
        "union_wagers",
        "jaccard",
        "shared_eligible_wagers",
        "relationship",
        "same_parent",
        "correction_family_unchanged",
    ]
    tables = {
        "signal_statistics": pd.DataFrame(evidence),
        "signal_hypotheses": corrected,
        "signal_overlap": pd.DataFrame(overlap, columns=overlap_columns),
        "signal_priorities": priority,
        "signal_shortlist": priority.loc[priority.included_shortlist].reset_index(drop=True),
    }
    audit = {
        "candidate_count": len(candidates),
        "hypothesis_count": len(corrected),
        "hypotheses_per_candidate": 4,
        "correction_family_id": correction,
        "blocked_hypotheses": int(corrected.placeholder_in_correction.sum()),
        "test_status_counts": {
            str(k): int(v)
            for k, v in pd.Series([r["test_status"] for r in evidence]).value_counts().items()
        },
        "category_counts": {str(k): int(v) for k, v in priority.category.value_counts().items()},
        "overlap_pairs_examined": pair_count,
        "overlap_pairs_recorded": len(overlap),
        "unique_test_partitions": len(cache),
        "bootstrap_cells": cells,
        "suppressed_equivalent_candidates": int(priority.suppression_reason.notna().sum()),
        "shortlist_count": len(tables["signal_shortlist"]),
        "family_preserved_after_suppression": len(corrected) == 4 * len(candidates),
        "validated_candidates": 0,
        "production_authorized_candidates": 0,
        "policy_approval_reference": cfg.approval_reference,
        "limitations": [
            "event_proxies_not_proven_independent",
            "chronology_unverified",
            "conditional_t_approximation_not_exact_randomization",
            "selected_logged_history",
            "fees_and_fillability_unverified",
            "no_causal_filter_edit_claim",
        ],
    }
    return tables, audit
