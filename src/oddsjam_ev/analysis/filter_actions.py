"""
filter_actions.py
===============================================================================
Purpose
-------
Translate predicates conservatively and audit observed filter coverage.

What this module provides
-------------------------
Stable target resolution, exact UI mappings and unique-wager impact diagnostics.

Design Notes
------------
Historical memberships never establish current live settings or future coverage.
===============================================================================
"""

import json

import pandas as pd

from oddsjam_ev.analysis.filter_attribution import validate_registry
from oddsjam_ev.analysis.recommendation_config import validate_capabilities


def resolve_target(target_id: str | None, registry: dict) -> dict:
    """Describe an explicitly selected historical filter without changing its ID.

    Parameters
    ----------
    target_id : str, optional
        Stable registry ID, not a name prefix or inferred target.
    registry : dict
        Dated filter registry.

    Returns
    -------
    dict
        Target evidence and whether it is confirmed as existing in the registry.
    """
    validate_registry(registry)
    if target_id is None:
        return {
            "target_filter_id": None,
            "target_status": "not_selected",
            "target_lifecycle": None,
            "target_existing": False,
        }
    row = next((x for x in registry["filters"] if x["id"] == target_id), None)
    if row is None:
        raise ValueError("Unknown target filter ID")
    existing = row.get("lifecycle") in {"existing", "existing_corrected_2026_10_22"}
    return {
        "target_filter_id": target_id,
        "target_status": row.get("status", "unknown"),
        "target_lifecycle": row.get("lifecycle", "unknown"),
        "target_existing": existing,
    }


def translate_predicate(predicate: dict, capabilities: dict) -> dict:
    """Translate only explicitly reviewed exact source-value semantics.

    Parameters
    ----------
    predicate : dict
        Atlas predicate, including its exact interval labels.
    capabilities : dict
        Reviewed UI capability mappings.

    Returns
    -------
    dict
        Translation status, source references, and unresolved dimensions.
    """
    validate_capabilities(capabilities)
    controls = []
    unresolved = []
    for dimension, value in sorted(predicate.items()):
        if not isinstance(value, str) or value.startswith("<"):
            unresolved.append(dimension + ":diagnostic_category")
            continue
        exact = capabilities["exact_values"].get(dimension, {}).get(value)
        if exact is None:
            unresolved.append(dimension + ":exact_ui_semantics_unverified")
        else:
            controls.append({"source_dimension": dimension, "source_value": value, **exact})
    return {
        "translation_status": "exact_reviewed"
        if predicate and not unresolved
        else "unavailable_or_partial",
        "ui_controls_json": json.dumps(controls, sort_keys=True),
        "translation_blockers_json": json.dumps(
            unresolved if predicate else ["overall_is_not_a_filter_predicate"]
        ),
    }


def coverage_impact(
    wagers: pd.DataFrame, attribution: pd.DataFrame, member_ids: set, target_id: str | None
) -> dict:
    """Count unique observed target wagers and the subset matching a predicate.

    Parameters
    ----------
    wagers : pandas.DataFrame
        One row per canonical wager in the declared execution/product cohort.
    attribution : pandas.DataFrame
        Canonical filter bridge; no monetary fields are taken from this table.
    member_ids : set
        Unique candidate bet IDs. Matching is not an instruction to retain/remove.
    target_id : str, optional
        Explicit target. Without one, target coverage is unavailable.

    Returns
    -------
    dict
        Observed counts/stake shares, overlap and unresolved membership diagnostics.
    """
    if wagers.bet_id.duplicated().any():
        raise ValueError("Coverage requires unique wagers")
    bridge = attribution.loc[attribution.bet_id.isin(wagers.bet_id)].drop_duplicates(
        ["bet_id", "filter_id", "attribution_status"]
    )
    unresolved = set(bridge.loc[bridge.attribution_status.ne("resolved"), "bet_id"]) | (
        set(wagers.bet_id) - set(bridge.bet_id)
    )
    known = bridge.loc[bridge.attribution_status.eq("resolved")]
    target = set(known.loc[known.filter_id.eq(target_id), "bet_id"]) if target_id else set()
    matched = target & member_ids
    economic = wagers.loc[wagers.reported_roi_eligible & wagers.bet_id.isin(target)].sort_values(
        "bet_id"
    )
    matched_economic = economic.loc[economic.bet_id.isin(matched)]
    stake = economic.reported_stake_usd.sum(min_count=1)
    matched_stake = matched_economic.reported_stake_usd.sum(min_count=1)
    if len(economic) and not len(matched_economic):
        matched_stake = 0.0
    overlaps = (
        set(known.loc[known.filter_id.ne(target_id), "bet_id"]) & target if target_id else set()
    )
    return {
        "coverage_status": "observed_resolved_history_only" if target_id else "target_not_selected",
        "target_observed_wagers": len(target) if target_id else None,
        "matching_target_wagers": len(matched) if target_id else None,
        "nonmatching_target_wagers": len(target - matched) if target_id else None,
        "target_eligible_wagers": len(economic) if target_id else None,
        "matching_target_eligible_wagers": len(matched_economic) if target_id else None,
        "nonmatching_target_eligible_wagers": len(economic) - len(matched_economic)
        if target_id
        else None,
        "target_eligible_stake_usd": stake if target_id else None,
        "matching_target_stake_share": matched_stake / stake
        if target_id and pd.notna(stake) and stake > 0
        else None,
        "nonmatching_target_stake_share": 1 - matched_stake / stake
        if target_id and pd.notna(stake) and stake > 0
        else None,
        "other_filter_overlap_wagers": len(overlaps) if target_id else None,
        "cohort_unresolved_attribution_wagers": len(unresolved),
        "future_scanner_coverage_verified": False,
    }
