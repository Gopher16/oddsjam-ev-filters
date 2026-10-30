"""
multidimensional_discovery.py
===============================================================================
Purpose
-------
Enumerate bounded observed predicates and reconcile within-filter comparisons.

What this module provides
-------------------------
Outcome-independent family ledger, parent/subgroup membership and paired economics.

Design Notes
------------
Complement means parent minus subgroup, not a global losing slice. No inference,
ranking, widening or automatic exclusion recommendation is performed.
===============================================================================
"""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.continuous_ev import ev_threshold, parse_ev
from oddsjam_ev.analysis.multidimensional_config import EV_DIMENSION, MultidimensionalConfig
from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig, summarize_sizing_cohort
from oddsjam_ev.io.source_csv import fingerprint

MISSING = {"", "<missing>", "<invalid>", "<conflict>", "<unmapped>"}
BLOCKS = [
    "descriptive_only",
    "chronology_unverified",
    "comparison_inference_deferred",
    "multiplicity_not_applied",
    "inferential_support_policy_unset",
    "fees_and_fillability_unverified",
]


class DiscoveryBudgetExceeded(ValueError):
    """Resource overflow with a serializable audit; no truncated family is valid."""

    def __init__(self, audit):
        self.audit = audit
        super().__init__("Bounded discovery budget exceeded; no partial family published")


def build_within_filter_discovery(
    canonical: dict,
    dimensions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    cfg: MultidimensionalConfig,
    sizing_cfg: SizingDiagnosticsConfig,
    lineage: dict,
) -> tuple[dict, dict]:
    """Freeze observed predicates before reading returns; evaluate exact parent sets.

    Parameters
    ----------
    canonical : dict
        Verified canonical facts/attribution bridge.
    dimensions : pandas.DataFrame
        Verified atlas wager dimensions tied to the same canonical bundle.
    diagnostics : pandas.DataFrame
        Verified prepared sizing diagnostics with paired eligibility/estimands.
    cfg : MultidimensionalConfig
        Predeclared views, numeric cutpoints and fail-closed budgets.
    sizing_cfg : SizingDiagnosticsConfig
        Original diagnostic unit policy. No resampling is performed here.
    lineage : dict
        Pinned input identities/hashes incorporated in family and candidate IDs.

    Returns
    -------
    tuple[dict, dict]
        Complete candidate/metric/comparison/membership tables and a budget audit.
        Empty/sparse subgroups remain in the family; no outcome screening occurs.
    """
    family = "family:" + fingerprint({"lineage": lineage, "policy": asdict(cfg)})
    usage = {
        "parents": 0,
        "candidates": 0,
        "memberships": 0,
        "scan_rows": 0,
        "excluded_missing_category_groups": 0,
    }

    def spend(key, value):
        usage[key] += value
        if usage[key] > getattr(cfg, "max_" + key):
            raise DiscoveryBudgetExceeded(
                {
                    "family_id": family,
                    "status": "aborted_budget_exceeded_not_truncated",
                    "budget": key,
                    "usage": dict(usage),
                    "policy": asdict(cfg),
                }
            )

    source = canonical["bet_facts"]
    bridge = canonical["filter_attribution"]
    categorical = sorted({d for v in cfg.views for d in v if d != EV_DIMENSION})
    for frame in (source, dimensions, diagnostics):
        if (
            "bet_id" not in frame
            or frame.bet_id.isna().any()
            or frame.bet_id.duplicated().any()
            or set(frame.bet_id) != set(source.bet_id)
        ):
            raise ValueError("Within-filter input identities do not reconcile")
    if set(categorical) - set(dimensions) or "percentage" not in diagnostics:
        raise ValueError("Missing exact discovery dimensions")
    check = [
        "reported_stake_usd",
        "reported_profit_usd",
        "reported_roi_eligible",
        "execution_status",
        "product",
        "namespace",
        "game_id",
        "percentage",
    ]
    left = diagnostics.set_index("bet_id").sort_index()
    right = source.set_index("bet_id").sort_index()
    for column in check:
        pd.testing.assert_series_equal(left[column], right[column], check_names=False)
    data = (
        diagnostics.drop(columns=[d for d in categorical if d in diagnostics])
        .merge(dimensions[["bet_id", *categorical]], on="bet_id", validate="one_to_one")
        .set_index("bet_id", drop=False)
        .rename_axis(None)
        .sort_index()
    )
    if not {"bet_id", "filter_id", "attribution_status"} <= set(bridge) or not set(
        bridge.bet_id
    ) <= set(data.bet_id):
        raise ValueError("Invalid within-filter attribution")
    resolved = bridge.loc[bridge.attribution_status.eq("resolved"), ["bet_id", "filter_id"]]
    if resolved.filter_id.isna().any() or resolved.duplicated().any():
        raise ValueError("Duplicate within-filter membership")
    parents = []
    exclusions = []
    members = []
    ledger = []
    plans = []
    expanded = data.reset_index(drop=True).merge(resolved, on="bet_id", validate="one_to_many")
    # Enumeration touches identities/features only, never profit/support outcomes.
    for scope, parent in expanded.groupby(["filter_id", "execution_status", "product"], sort=True):
        spend("parents", 1)
        ids = sorted(parent.bet_id)
        parent = data.loc[ids]
        scope_dict = dict(zip(["filter_id", "execution_status", "product"], scope, strict=True))
        parent_id = "parent:" + fingerprint({"lineage": lineage, **scope_dict, "members": ids})
        parents.append((parent_id, scope_dict, ids))
        spend("memberships", len(ids))
        spend("scan_rows", len(ids))
        members.extend({"scope_id": parent_id, "role": "parent", "bet_id": b} for b in ids)
        for view in cfg.views:
            cats = [d for d in view if d != EV_DIMENSION]
            thresholds = cfg.ev_thresholds if EV_DIMENSION in view else (None,)
            spend("scan_rows", len(parent) * max(1, len(thresholds)))
            masks = {
                json.dumps(t, sort_keys=True): ev_threshold(
                    parent.percentage, t["operator"], t["endpoint"]
                )
                for t in thresholds
                if t is not None
            }
            groups = parent.groupby(cats, sort=True, dropna=False) if cats else [((), parent)]
            for values, group in groups:
                values = (values,) if len(cats) == 1 and not isinstance(values, tuple) else values
                if any(pd.isna(v) or str(v).strip() in MISSING for v in values):
                    usage["excluded_missing_category_groups"] += 1
                    exclusions.append(
                        {
                            "parent_id": parent_id,
                            **scope_dict,
                            "view_json": json.dumps(view),
                            "category_values_json": json.dumps(
                                [None if pd.isna(v) else str(v) for v in values]
                            ),
                            "affected_candidate_count": len(group),
                            "reason": "missing_invalid_or_unmapped_positive_category_retained_in_parent_complement",
                        }
                    )
                    continue
                terms = [
                    {"dimension": d, "operator": "==", "value": str(v)}
                    for d, v in zip(cats, values, strict=True)
                ]
                for threshold in thresholds:
                    predicate = terms + (
                        [{"dimension": EV_DIMENSION, **threshold}] if threshold is not None else []
                    )
                    predicate = sorted(predicate, key=lambda p: p["dimension"])
                    definition = {**scope_dict, "predicate": predicate}
                    ident = "candidate:" + fingerprint({"family_id": family, **definition})
                    spend("candidates", 1)
                    # Account for both partition summaries and regime breakdowns
                    # before evaluating any returns; this is a conservative work bound.
                    spend("scan_rows", len(ids) * 4)
                    selected = (
                        group.bet_id.tolist()
                        if threshold is None
                        else group.loc[
                            masks[json.dumps(threshold, sort_keys=True)].reindex(group.index),
                            "bet_id",
                        ].tolist()
                    )
                    selected = sorted(selected)
                    spend("memberships", len(selected))
                    members.extend(
                        {"scope_id": ident, "role": "subgroup", "bet_id": b} for b in selected
                    )
                    ledger.append(
                        {
                            "candidate_id": ident,
                            "family_id": family,
                            "parent_id": parent_id,
                            **scope_dict,
                            "predicate_json": json.dumps(predicate, sort_keys=True),
                            "predicate_depth": len(predicate),
                            "observed_subgroup_count": len(selected),
                            "coverage": "observed_logged_categories_not_available_universe",
                            "validation_status": "exploratory_only",
                            "blockers_json": json.dumps(BLOCKS),
                        }
                    )
                    plans.append((ident, parent_id, scope_dict, ids, selected))
    # Only a fully enumerated, budget-accounted family proceeds to economic summaries.
    metrics = []
    comparisons = []
    cache = {}

    def summarize(ident, role, scope, ids):
        cohort = data.loc[ids]
        digest = fingerprint(ids)
        if digest not in cache:
            cache[digest] = summarize_sizing_cohort(cohort, sizing_cfg)
        point = cache[digest]
        metrics.append({"cohort_id": ident, "role": role, **scope, "sizing_regime": None, **point})
        for regime, part in cohort.groupby("sizing_regime", sort=True):
            key = fingerprint(sorted(part.bet_id))
            if key not in cache:
                cache[key] = summarize_sizing_cohort(part, sizing_cfg)
            metrics.append(
                {"cohort_id": ident, "role": role, **scope, "sizing_regime": regime, **cache[key]}
            )
        return point

    for ident, scope, ids in parents:
        summarize(ident, "parent", scope, ids)
    for ident, parent_id, scope, ids, selected in plans:
        complement = sorted(set(ids) - set(selected))
        if set(selected) & set(complement) or set(selected) | set(complement) != set(ids):
            raise ValueError("Within-filter membership reconciliation failed")
        sub = summarize(ident, "subgroup", scope, selected)
        comp_id = "complement:" + fingerprint({"parent_id": parent_id, "candidate_id": ident})
        comp = summarize(comp_id, "complement", scope, complement)
        parent = cache[fingerprint(ids)]
        for key in ("candidate_count", "diagnostic_eligible_count"):
            if sub[key] + comp[key] != parent[key]:
                raise ValueError("Within-filter counts do not reconcile")
        for key in ("eligible_stake_usd", "reported_profit_usd", "fixed_unit_notional_profit_usd"):
            total = sum(p[key] if p["diagnostic_eligible_count"] else 0 for p in (sub, comp))
            expected = parent[key] if parent["diagnostic_eligible_count"] else 0
            if not np.isclose(total, expected, rtol=1e-10, atol=1e-8):
                raise ValueError("Within-filter economics do not reconcile")
        comparisons.append(
            {
                "candidate_id": ident,
                "parent_id": parent_id,
                "complement_id": comp_id,
                **scope,
                "subgroup_membership_fingerprint": fingerprint(selected),
                "complement_membership_fingerprint": fingerprint(complement),
                "partition_reconciles": True,
                "parent_count": len(ids),
                "subgroup_count": len(selected),
                "complement_count": len(complement),
                "support_status": "empty_subgroup"
                if not selected
                else "empty_complement"
                if not complement
                else "observed_descriptive",
                "subgroup_minus_complement_weighted_roi": sub["stake_weighted_roi"]
                - comp["stake_weighted_roi"],
                "subgroup_minus_complement_equal_stake_return": sub["equal_stake_mean_return"]
                - comp["equal_stake_mean_return"],
                "complement_minus_parent_weighted_roi": comp["stake_weighted_roi"]
                - parent["stake_weighted_roi"],
                "complement_minus_parent_equal_stake_return": comp["equal_stake_mean_return"]
                - parent["equal_stake_mean_return"],
                "interpretation": "historical_association_not_causal_filter_edit_gain",
                "comparison_uncertainty_status": "not_estimated_inference_deferred",
                "blockers_json": json.dumps(BLOCKS),
            }
        )
    if not ledger:
        raise ValueError("No observed predicates within resolved saved-filter parents")
    audit = {
        "family_id": family,
        "status": "complete_descriptive_family",
        "usage": usage,
        "candidate_count": len(ledger),
        "comparison_count": len(comparisons),
        "all_partitions_reconcile": True,
        "unresolved_or_missing_bridge_rows": int(bridge.attribution_status.ne("resolved").sum()),
        "missing_categories": "excluded_from_positive_predicates_retained_in_parent_complement",
        "complement_membership": "reconstruct_parent_minus_subgroup; digest_verified",
        "formal_validation_blocked": True,
        "blockers": BLOCKS,
        "outcome_screening_performed": False,
        "family_correction_applied": False,
        "ev_input_status_counts_all_candidates": {
            str(k): int(v) for k, v in parse_ev(data.percentage).ev_status.value_counts().items()
        },
    }
    return {
        "signal_candidates": pd.DataFrame(ledger)
        .sort_values("candidate_id")
        .reset_index(drop=True),
        "signal_evidence": pd.DataFrame(metrics),
        "signal_comparisons": pd.DataFrame(comparisons)
        .sort_values("candidate_id")
        .reset_index(drop=True),
        "signal_membership": pd.DataFrame(members)
        .sort_values(["scope_id", "bet_id"])
        .reset_index(drop=True),
        "signal_exclusions": pd.DataFrame(
            exclusions,
            columns=[
                "parent_id",
                "filter_id",
                "execution_status",
                "product",
                "view_json",
                "category_values_json",
                "affected_candidate_count",
                "reason",
            ],
        ),
    }, audit
