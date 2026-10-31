"""
review_queues.py
===============================================================================
Purpose
-------
Join historical comparisons and configuration evidence into auditable review queues.

What this module provides
-------------------------
Full-family review, separate global catalogue, research tiers and controlled proposals.

Design Notes
------------
Setup readiness is not statistical validation. Lifecycle and baseline evidence are
required for production proposals; no source inference or configuration is mutated.
===============================================================================
"""

import json
from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.decision_config import validate_evidence
from oddsjam_ev.analysis.decision_records import resolve_decision
from oddsjam_ev.analysis.paired_comparison import paired_comparison
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.analysis.review_catalogue import ReviewBudgetExceeded, enumerate_global
from oddsjam_ev.analysis.review_ranking import classify, components, rank_records
from oddsjam_ev.io.source_csv import fingerprint


def clean(value):
    """Normalize finite JSON scalars without rounding source inference values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def economics(frame):
    """Summarize eligible economics without treating empty returns as zero."""
    e = frame.loc[frame.diagnostic_eligible].sort_values("bet_id")
    return {
        "candidate_count": len(frame),
        "eligible_count": len(e),
        "stake_usd": float(e.stake.sum()) if len(e) else None,
        "reported_profit_usd": float(e.profit.sum()) if len(e) else None,
        "weighted_roi": float(e.profit.sum() / e.stake.sum()) if len(e) else None,
        "equal_stake_return": float(e.unit_return.mean()) if len(e) else None,
        "fixed_unit_profit_at_one_usd": float(e.unit_return.sum()) if len(e) else None,
        "sizing_regimes": [
            {
                "regime": str(key),
                "eligible_count": len(part),
                "weighted_roi": float(part.profit.sum() / part.stake.sum()),
                "equal_stake_return": float(part.unit_return.mean()),
            }
            for key, part in e.groupby("sizing_regime", sort=True)
        ],
    }


def build_review_queues(discovery, priority, decisions, visual, cfg):
    """Publish complete all-history review records without modifying correction results.

    Parameters
    ----------
    discovery, priority : dict
        Matching verified full candidate family and its original inference tables.
    decisions : dict
        Verified complete settings/lifecycle records and their pinned evidence spec.
    visual : dict
        Verified unique visual candidates and coverage of the same canonical universe.
    cfg : ReviewConfig
        Frozen presentation and separate descriptive catalogue policy.

    Returns
    -------
    tuple[dict, dict]
        JSON-compatible records/groups/overlap/coverage and reconciliation audit.
        No prospective-evidence tier is fabricated from historical inputs.
    """
    raw = visual["visual_candidates"]
    if raw.bet_id.isna().any() or raw.bet_id.duplicated().any():
        raise ValueError("Unique review universe required")
    data = raw.set_index("bet_id", drop=False).rename_axis(None).sort_index()
    source_candidates = discovery["signal_candidates"]
    ids = set(source_candidates.candidate_id)
    if source_candidates.candidate_id.duplicated().any() or not ids:
        raise ValueError("Complete nonempty corrected source family required")
    statistics = priority["signal_statistics"]
    ranks = priority["signal_priorities"]
    hypotheses = priority["signal_hypotheses"]
    if any(
        frame.candidate_id.duplicated().any() or set(frame.candidate_id) != ids
        for frame in (statistics, ranks)
    ):
        raise ValueError("Complete source statistics and priorities required")
    if len(hypotheses) != 4 * len(ids) or set(hypotheses.signal_candidate_id) != ids:
        raise ValueError("Full four-slot correction family required")
    for _, part in hypotheses.groupby("signal_candidate_id"):
        if set(zip(part.measure, part.direction, strict=True)) != {
            (m, d) for m in ("weighted", "equal_stake") for d in ("positive", "negative")
        }:
            raise ValueError("Directional correction slots do not reconcile")
    if (
        hypotheses.hypothesis_id.duplicated().any()
        or hypotheses.family_id.nunique() != 1
        or not hypotheses.family_size.eq(len(hypotheses)).all()
    ):
        raise ValueError("Correction family accounting mismatch")
    statistics = statistics.set_index("candidate_id")
    sheets = {}
    for record in decisions["records"]:
        key = record["candidate_id"], record["design"]
        if key in sheets:
            raise ValueError("Duplicate review configuration sheet")
        sheets[key] = record
    if set(sheets) != {
        (ident, design) for ident in ids for design in ("subgroup_tracking", "exclusion_comparison")
    }:
        raise ValueError("Complete decision sheets required")
    members = {}
    bridge = discovery["signal_membership"]
    if bridge.duplicated(["scope_id", "bet_id"]).any() or not set(bridge.bet_id) <= set(
        data.bet_id
    ):
        raise ValueError("Review membership mismatch")
    members.update({key: set(group.bet_id) for key, group in bridge.groupby("scope_id")})
    source = source_candidates.assign(source_kind="within_filter_corrected_family").to_dict(
        "records"
    )
    global_candidates, global_members, coverage = enumerate_global(raw, cfg)
    members.update(global_members)
    candidates = source + global_candidates
    if (
        len(candidates) > cfg.max_candidates
        or sum(map(len, members.values())) > cfg.max_memberships
    ):
        raise ReviewBudgetExceeded("Complete review family budget exceeded")
    if len(candidates) * (len(candidates) - 1) // 2 > cfg.max_pairs:
        raise ReviewBudgetExceeded("Complete review overlap budget exceeded")
    records = []
    ds = decisions["manifest"]["specification"]
    evidence = validate_evidence(ds["evidence"])
    for candidate in sorted(candidates, key=lambda r: r["candidate_id"]):
        ident, parent = candidate["candidate_id"], candidate["parent_id"]
        selected, all_ids = members.setdefault(ident, set()), members.get(parent, set())
        if not all_ids or not selected <= all_ids:
            raise ValueError("Review subgroup is not a parent partition")
        group, complement = data.loc[sorted(selected)], data.loc[sorted(all_ids - selected)]
        parent_rows = data.loc[sorted(all_ids)]
        if any(
            parent_rows[key].ne(candidate[key]).any() for key in ("execution_status", "product")
        ):
            raise ValueError("Review economic population mismatch")
        observed = {
            role: economics(part)
            for role, part in (
                ("parent", parent_rows),
                ("subgroup", group),
                ("complement", complement),
            )
        }
        is_global = candidate["source_kind"] == "global_descriptive_catalogue"
        if is_global:
            stat, _ = paired_comparison(
                group, complement, PrioritizationConfig(), seed=0, evaluate=False
            )
            setup_candidate = {**candidate, "parent_id": None}
            setup = resolve_decision(
                setup_candidate, evidence, "subgroup_tracking", ds["as_of"], ds["effective_at"]
            )
            exclusion = None
        else:
            stat = statistics.loc[ident].to_dict()
            setup, exclusion = (
                sheets[(ident, "subgroup_tracking")],
                sheets[(ident, "exclusion_comparison")],
            )
            if (
                any(
                    setup.get(key) != candidate.get(key)
                    for key in ("parent_id", "filter_id", "product")
                )
                or setup["historical_funding"] != candidate["execution_status"]
                or setup["logical_change"]["terms"] != json.loads(candidate["predicate_json"])
            ):
                raise ValueError("Review setup scope mismatch")
            for role in ("subgroup", "complement"):
                value = observed[role]
                if value["eligible_count"] != stat[role + "_eligible_count"] or not np.allclose(
                    [value["stake_usd"] if value["eligible_count"] else 0],
                    [stat[role + "_stake_usd"]],
                    rtol=1e-10,
                    atol=1e-8,
                ):
                    raise ValueError("Review economics differ from corrected source")
                for name in ("weighted_roi", "equal_stake_return"):
                    expected = stat[role + "_" + name]
                    actual = value[name]
                    if not np.isclose(
                        np.nan if actual is None else actual, expected, equal_nan=True
                    ):
                        raise ValueError("Review return differs from corrected source")
        # Global preflight produces no p-values or intervals. Existing statistics stay intact.
        tests = hypotheses.loc[hypotheses.signal_candidate_id.eq(ident)].sort_values(
            "hypothesis_id"
        )
        rejected = {
            direction: not is_global
            and bool(tests.loc[tests.direction.eq(direction), "exploratory_holm_reject"].all())
            for direction in ("positive", "negative")
        }
        direction_hint = (
            "positive"
            if stat["subgroup_weighted_roi"] > stat["complement_weighted_roi"]
            else "negative"
        )
        direction, tier = classify(stat, rejected[direction_hint])
        pinned_prod = bool(
            exclusion
            and exclusion["lifecycle"]["status"] == "PROD"
            and exclusion["lifecycle"]["confirmation_status"] == "confirmed"
            and exclusion["baseline"]
            and exclusion["baseline"].get("baseline_sha256")
        )
        queue = (
            "opportunities"
            if direction == "opportunity"
            else "production_changes"
            if direction == "loss_association" and pinned_prod
            else "investigations"
        )
        proposal = None
        if queue == "production_changes":
            proposal = {
                "design": "baseline_vs_excluded_subgroup_paper_comparison",
                "configuration_record_id": exclusion.get("decision_id"),
                "baseline_id": exclusion["baseline"]["baseline_id"],
                "baseline_sha256": exclusion["baseline"]["baseline_sha256"],
                "logical_change": exclusion["logical_change"],
                "parameter_change_sheet": exclusion["settings"],
                "removed_candidate_count": len(selected),
                "retained_candidate_count": len(all_ids - selected),
                "removed_eligible_stake_share": observed["subgroup"]["stake_usd"]
                / observed["parent"]["stake_usd"]
                if observed["parent"]["stake_usd"] and observed["subgroup"]["stake_usd"]
                else None,
                "implementation_status": exclusion["setup_status"],
                "implementation_blockers": exclusion["setup_blockers"],
                "production_change_authorized": False,
                "profit_saved_or_forecast": None,
                "interpretation": "historical_partition_not_causal_performance_of_modified_scanner",
            }
        predicate = json.loads(candidate["predicate_json"])
        canonical_terms = sorted(predicate, key=lambda term: json.dumps(term, sort_keys=True))
        historical_loss = observed["subgroup"]["reported_profit_usd"]
        action_sheet = exclusion if queue == "production_changes" else setup
        record = {
            **candidate,
            "filter_identity_state": "proposed_new_not_registered"
            if is_global
            else "historical_registered_parent",
            "hypothesis_group_id": "idea:" + fingerprint(canonical_terms),
            "direction": direction,
            "queue": queue,
            "evidence_tier": tier,
            "rank_components": components(stat, direction),
            "setup_status": action_sheet["setup_status"],
            "setup_blockers": action_sheet["setup_blockers"],
            "configuration_sheet": action_sheet,
            "production_lifecycle_confirmed_with_pinned_baseline": pinned_prod,
            "production_proposal": proposal,
            "historical_economics": observed,
            "comparison_evidence": stat,
            "original_hypotheses": tests.to_dict("records"),
            "corrected_direction_rejected_both_measures": rejected,
            "formal_inference_status": "not_tested_separate_descriptive_catalogue"
            if is_global
            else "unchanged_published_correction_family",
            "source_correction_family_id": None if is_global else hypotheses.family_id.iloc[0],
            "historical_loss_burden_usd": max(0.0, -historical_loss)
            if historical_loss is not None
            else None,
            "future_profit_forecast": None,
            "validated": False,
            "production_authorized": False,
            "next_action": "resolve_production_lifecycle_and_baseline"
            if direction == "loss_association" and not pinned_prod
            else "investigate_sizing_and_estimand_conflict"
            if direction == "sizing_or_estimand_conflict"
            else "review_evidence_and_resolve_setup_before_freezing_paper_test",
            "interpretation": "research_priority_not_proven_edge_or_automatic_filter_change",
        }
        if "blockers_json" in record:
            record["source_discovery_blockers_json"] = record.pop("blockers_json")
            record["source_discovery_blockers_interpretation"] = (
                "original_pre_inference_ledger_not_current_queue_status"
            )
        if proposal is not None:
            proposal["applied_configuration_status"] = "not_verified_by_prospective_baseline"
            proposal["next_action"] = (
                "confirm_current_applied_configuration_before_any_external_edit"
            )
        records.append(record)
    ordered, overlap, groups = rank_records(records, members, data, cfg)
    payload = clean(
        {
            "records": ordered,
            "hypothesis_groups": groups,
            "overlap": overlap,
            "catalogue_coverage": coverage,
            "search_contract": {
                "policy": asdict(cfg),
                "supported_features": [
                    "sportsbook",
                    "sport",
                    "league",
                    "market_family",
                    "advertised_ev_band",
                    "odds_band",
                ],
                "not_searched": [
                    {
                        "feature": "liquidity",
                        "reason": "No trusted decision-time USD capacity in the pinned source; future catalogue needs its own explicit contract.",
                    },
                    {
                        "feature": "time_to_event_or_calendar_timing",
                        "reason": "Placement/event timezone semantics unverified; source-date activity is not timing optimization evidence.",
                    },
                    {
                        "feature": "devig_formula_reference_books_weights",
                        "reason": "Stored EV alone cannot reconstruct alternative decision-time pricing models.",
                    },
                    {
                        "feature": "minimum_recommended_bet_or_sizing_controls",
                        "reason": "Unknown historical sizing and selected logged wagers cannot reconstruct alternative execution or scanner supply.",
                    },
                ],
                "universe": "logged_source_candidates_only_no_unlogged_opportunities",
            },
            "universe": {
                "candidate_count": len(raw),
                "eligible_count": int(raw.diagnostic_eligible.sum()),
                "populations": [
                    {"execution_status": key[0], "product": key[1], **economics(part)}
                    for key, part in raw.groupby(
                        ["execution_status", "product"], sort=True, dropna=False
                    )
                ],
                "coverage_scope": "explicit_reconciled_source_snapshot_not_all_possible_history_or_scanner_supply",
            },
        }
    )
    audit = {
        "record_count": len(records),
        "source_candidate_count": len(ids),
        "global_catalogue_count": len(global_candidates),
        "source_hypothesis_count": len(hypotheses),
        "source_correction_family_id": hypotheses.family_id.iloc[0],
        "original_shortlist_count": len(priority["signal_shortlist"]),
        "source_correction_unchanged": True,
        "new_hypothesis_tests_performed": False,
        "prospectively_validated_count": 0,
        "tier_counts": dict(Counter(r["evidence_tier"] for r in ordered)),
        "queue_counts": dict(Counter(r["queue"] for r in ordered)),
        "headline_counts": dict(Counter(r["queue"] for r in ordered if r["headline"])),
        "production_proposal_count": sum(r["production_proposal"] is not None for r in ordered),
        "overlap_pairs": len(overlap),
        "global_catalogue_missing_groups": len(coverage),
        "original_support_qualified_count": int(
            priority["signal_statistics"]
            .test_status.eq("approximate_conditional_exploratory")
            .sum()
        ),
        "rank_sensitivity_changed_count": sum(
            r["sensitivity_order_change_without_interval"] != 0 for r in ordered
        ),
        "independent_prospective_tier_status": "unavailable_no_verified_prospective_review_input",
        "ranking_policy": "tier_then_pareto_contrast_min_events_negative_max_event_share_then_events_share_stable_id",
        "sensitivity_policy": "repeat_fronts_without_pointwise_interval_objective_no_new_tests",
        "limitations": [
            "selected_logged_history",
            "chronology_unverified",
            "event_proxies_not_proven_independent",
            "global_catalogue_not_corrected",
            "fees_and_fillability_unverified",
            "no_causal_modification_claim",
        ],
    }
    return payload, clean(audit)
