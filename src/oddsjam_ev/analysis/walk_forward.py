"""
walk_forward.py
===============================================================================
Purpose
-------
Gate chronological research on explicit time and information-availability evidence.

What this module provides
-------------------------
Auditable development/selection/test partitions with event purging and embargo.

Design Notes
------------
Naive source timestamps never become UTC by assumption. Frozen candidate families
are enumerated from development only; later outcomes cannot alter definitions.
===============================================================================
"""

import json

import pandas as pd

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.segment_search import enumerate_segments

EVIDENCE_FLAGS = (
    "placement_time_verified",
    "decision_features_verified",
    "outcome_availability_verified",
    "outcome_values_asof_verified",
)
TIME_COLUMNS = (
    "created_at_utc",
    "features_available_at_utc",
    "outcome_available_at_utc",
    "information_end_utc",
)


def _aware(value):
    if pd.isna(value):
        return pd.NaT
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Temporal timestamps must have explicit timezone")
    return value.tz_convert("UTC")


def build_walk_forward(
    wagers: pd.DataFrame, cfg: DiscoveryConfig, evidence: dict | None = None
) -> dict[str, pd.DataFrame]:
    """Build evidence-gated partitions and development-only candidate ledgers.

    Parameters
    ----------
    wagers : pandas.DataFrame
        Atlas wagers; eligible rows require verified information timestamps.
    cfg : DiscoveryConfig
        Explicit rolling-origin windows and justified embargo.
    evidence : dict, optional
        Verifications with source_reference and timezone-aware observed_as_of_utc.
        These attestations require external supporting evidence, not inference.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Gate reasons, per-fold row dispositions and frozen development candidates.
        Blocked gates publish no partitions or temporal performance claims.
    """
    evidence = evidence or {}
    reasons = []
    for flag in EVIDENCE_FLAGS:
        if evidence.get(flag) is not True:
            reasons.append(flag.removesuffix("_verified") + "_unverified")
    if not evidence.get("source_reference"):
        reasons.append("evidence_source_reference_missing")
    if not cfg.temporal_windows:
        reasons.append("temporal_windows_unset")
    if cfg.embargo_days is None:
        reasons.append("embargo_unset")
    missing = set(TIME_COLUMNS) - set(wagers)
    reasons.extend("missing_" + c for c in sorted(missing))
    data = wagers.loc[wagers.reported_roi_eligible].copy()
    asof = pd.NaT
    try:
        asof = _aware(evidence.get("observed_as_of_utc"))
        if pd.isna(asof):
            reasons.append("observed_as_of_unverified")
        if not missing:
            for col in TIME_COLUMNS:
                data[col] = data[col].map(_aware)
    except (ValueError, TypeError):
        reasons.append("invalid_or_naive_temporal_evidence")
    if "created_at_utc" in data and data.created_at_utc.isna().all():
        reasons.append("no_verified_placement_times")
    gates = pd.DataFrame(
        [
            {
                "status": "blocked" if reasons else "available_conditional_on_evidence",
                "reasons_json": json.dumps(sorted(set(reasons))),
                "chronology_inferred": False,
            }
        ]
    )
    partition_columns = ["fold_id", "bet_id", "partition", "reason"]
    frozen_columns = [
        "fold_id",
        "candidate_id",
        "execution_status",
        "product",
        "view",
        "predicate_json",
        "candidate_count",
        "coverage",
        "role",
    ]
    metrics_columns = [
        "fold_id",
        "candidate_id",
        "partition",
        "eligible_count",
        "eligible_stake_usd",
        "eligible_reported_profit_usd",
        "reported_roi",
    ]
    stability_columns = [
        "fold_id",
        "partition",
        "calendar_month_utc",
        "execution_status",
        "product",
        "eligible_count",
        "eligible_stake_usd",
        "eligible_reported_profit_usd",
        "reported_roi",
    ]
    audit_columns = [
        "fold_id",
        "fold_status",
        "eligible_input_count",
        "development_count",
        "selection_count",
        "test_count",
        "excluded_count",
        "reconciles",
    ]
    if reasons:
        return {
            "temporal_gates": gates,
            "temporal_partitions": pd.DataFrame(columns=partition_columns),
            "frozen_candidates": pd.DataFrame(columns=frozen_columns),
            "temporal_metrics": pd.DataFrame(columns=metrics_columns),
            "temporal_stability": pd.DataFrame(columns=stability_columns),
            "temporal_fold_audit": pd.DataFrame(columns=audit_columns),
        }
    rows = []
    audits = []
    stability = []
    frozen = []
    metrics = []
    gap = pd.Timedelta(days=cfg.embargo_days)
    for window in cfg.temporal_windows:
        fold = window["fold_id"]
        start, selection, test, end = [
            _aware(window[k])
            for k in ["development_start", "selection_start", "test_start", "test_end"]
        ]
        part = pd.Series("outside_window", index=data.index)
        for name, lo, hi in [
            ("development", start, selection),
            ("selection", selection, test),
            ("test", test, end),
        ]:
            part.loc[data.created_at_utc.ge(lo) & data.created_at_utc.lt(hi)] = name
        reason = pd.Series("outside_window", index=data.index)
        reason.loc[part.ne("outside_window")] = "included"
        invalid = (
            data[list(TIME_COLUMNS)].isna().any(axis=1)
            | data.game_id.fillna("").eq("")
            | data.namespace.fillna("").eq("")
        )
        invalid |= data.features_available_at_utc.gt(data.created_at_utc)
        invalid |= data.outcome_available_at_utc.lt(
            data.created_at_utc
        ) | data.information_end_utc.lt(data.outcome_available_at_utc)
        invalid |= data.information_end_utc.gt(asof)
        # Missing or invalid members invalidate the entire known event cluster.
        keyed = data[["namespace", "game_id"]].apply(tuple, axis=1)
        invalid_keys = set(keyed[invalid])
        invalid |= keyed.isin(invalid_keys)
        reason.loc[invalid] = "invalid_or_unavailable_information"
        active = data.loc[~invalid].assign(partition=part[~invalid])
        for _, group in active.groupby(["namespace", "game_id"], sort=True):
            participating = set(group.partition) - {"outside_window"}
            if len(set(group.partition)) > 1 and participating:
                reason.loc[group.index] = "event_spans_partition_boundary"
            elif participating:
                name = next(iter(participating))
                cutoff = {"development": selection, "selection": test, "test": asof}[name]
                margin = gap if name != "test" else pd.Timedelta(0)
                if group.information_end_utc.max() >= cutoff - margin:
                    reason.loc[group.index] = "information_overlap_or_embargo"
        if end > asof:
            reason[:] = "incomplete_test_window"
        included = reason.eq("included")
        counts = {
            name: int((included & part.eq(name)).sum())
            for name in ["development", "selection", "test"]
        }
        excluded = int((~included).sum())
        audits.append(
            {
                "fold_id": fold,
                "fold_status": "complete_partitions_descriptive_only"
                if all(counts.values())
                else "blocked_incomplete_or_empty_partition",
                "eligible_input_count": len(data),
                "development_count": counts["development"],
                "selection_count": counts["selection"],
                "test_count": counts["test"],
                "excluded_count": excluded,
                "reconciles": sum(counts.values()) + excluded == len(data),
            }
        )
        stable = data.loc[included].copy()
        stable["partition"] = part.loc[included]
        stable["calendar_month_utc"] = stable.created_at_utc.dt.strftime("%Y-%m")
        for key, group in stable.groupby(
            ["partition", "calendar_month_utc", "execution_status", "product"], sort=True
        ):
            stake = group.reported_stake_usd.sum(min_count=1)
            profit = group.reported_profit_usd.sum(min_count=1)
            stability.append(
                dict(zip(stability_columns[:5], [fold, *key], strict=True))
                | {
                    "eligible_count": len(group),
                    "eligible_stake_usd": stake,
                    "eligible_reported_profit_usd": profit,
                    "reported_roi": profit / stake if pd.notna(stake) and stake > 0 else None,
                }
            )
        dev = data.loc[included & part.eq("development")]
        ledger, _ = enumerate_segments(dev, cfg)
        ledger.insert(0, "fold_id", fold)
        frozen.extend(ledger.to_dict("records"))
        for candidate in ledger.to_dict("records"):
            predicates = json.loads(candidate["predicate_json"])
            for name in ["development", "selection", "test"]:
                selected = data.loc[
                    included
                    & part.eq(name)
                    & data.execution_status.eq(candidate["execution_status"])
                    & data["product"].eq(candidate["product"])
                ]
                for dimension, value in predicates.items():
                    selected = selected.loc[selected[dimension].eq(value)]
                stake = selected.reported_stake_usd.sum(min_count=1)
                profit = selected.reported_profit_usd.sum(min_count=1)
                metrics.append(
                    {
                        "fold_id": fold,
                        "candidate_id": candidate["candidate_id"],
                        "partition": name,
                        "eligible_count": len(selected),
                        "eligible_stake_usd": stake,
                        "eligible_reported_profit_usd": profit,
                        "reported_roi": profit / stake if pd.notna(stake) and stake > 0 else None,
                    }
                )
        rows.extend(
            {
                "fold_id": fold,
                "bet_id": data.loc[i, "bet_id"],
                "partition": part.loc[i] if included.loc[i] else "excluded",
                "reason": reason.loc[i],
            }
            for i in data.index
        )
    return {
        "temporal_gates": gates,
        "temporal_partitions": pd.DataFrame(rows, columns=partition_columns),
        "frozen_candidates": pd.DataFrame(frozen, columns=frozen_columns),
        "temporal_metrics": pd.DataFrame(metrics, columns=metrics_columns),
        "temporal_stability": pd.DataFrame(stability, columns=stability_columns),
        "temporal_fold_audit": pd.DataFrame(audits, columns=audit_columns),
    }
