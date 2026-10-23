"""
reconcile.py
===============================================================================
Purpose
-------
Reconcile repeated source observations without guessing execution identity.

What this module provides
-------------------------
Conservative identity candidates, append-only revision views, and provenance-aware
user-assumed execution classification.

Design Notes
------------
Opportunity IDs and filter labels are never execution keys. Ambiguous groups are
excluded from current research state, while every source observation is retained.
===============================================================================
"""

from __future__ import annotations

import json
import math
import re
from datetime import date

import pandas as pd

from oddsjam_ev.io.ingestion import utc_time
from oddsjam_ev.io.source_csv import fingerprint

REAL_FILTER = "PROD - Exchanges: Core Exchange Conservative (Low Risk, Slow Build)"
IDENTITY_FIELDS = (
    "sportsbook",
    "game_id",
    "market_name",
    "bet_name",
    "odds",
    "stake",
    "bet_type",
    "parlay_id",
)
POLICY = {
    "version": 2,
    "provenance": "user_clarified_historical_funding_2026-10-22",
    "historical_real_filter": REAL_FILTER,
    "future_designations_effective": "2026-10-22",
    "unknown_timestamp_policy": "historical_only_unless_date_clearly_after_boundary",
    "core_selection": "selectively_chosen_funded_trades_not_scanner_universe",
    "historical_other_named_filters": "paper_regardless_of_designation",
    "exchange_order_behavior": "take_existing_orders_only",
    "exchange_order_behavior_provenance": "user_confirmed_2026-10-22",
}


def classify_execution(label: str, created_at: str) -> str:
    """Classify execution under the dated user-supplied policy.

    Parameters
    ----------
    label : str
        Original attribution; never split on commas inside filter names.
    created_at : str
        Original source timestamp. Historical non-Core named filters are paper,
        regardless of PROD/TEST spelling; mixed Core attribution stays unknown.

    Returns
    -------
    str
        real_money, paper, or unknown; never proof of actual funding or fills.
    """
    if not label.strip():
        return "unknown"
    try:
        day = date.fromisoformat(created_at[:10])
    except ValueError:
        return "unknown"
    if day <= date(2026, 10, 22):
        if label == REAL_FILTER:
            return "real_money"
        if REAL_FILTER in label:
            return "unknown"
        return "paper"
    designations = set(re.findall(r"(?:^|[,;|]\s*)(PROD|TEST)\s*-", label))
    if designations == {"PROD"}:
        return "real_money"
    if designations == {"TEST"}:
        return "paper"
    return "unknown"


def _time_key(row: pd.Series) -> str | None:
    try:
        value = pd.Timestamp(row.get("created_at", ""))
        if pd.isna(value):
            return None
        if value.tzinfo is not None:
            return value.tz_convert("UTC").isoformat()
        timezone = row["_ingest_timestamp_timezone"]
        if timezone == "unknown":
            return "unverified:" + str(row["created_at"])
        localized = value.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        if pd.isna(localized):
            return None
        return localized.tz_convert("UTC").isoformat()
    except (ValueError, TypeError, KeyError):
        return None


def reconcile(
    observations: pd.DataFrame, decisions: list[dict] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build deterministic current state plus a complete observation audit.

    Parameters
    ----------
    observations : pandas.DataFrame
        Lossless observations from load_observations.
    decisions : list[dict] or None
        Explicit observation_id, record_id, reason, and decided_at mappings.
        Each resolution is pinned in the snapshot manifest. To separate repeated
        bets, assign distinct record IDs; to link them, assign the same ID.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, dict]
        Selected current records, all observations with dispositions/revisions,
        and reconciliation counts.

    Notes
    -----
    Automatic keys require complete identity evidence and no within-file collision.
    Unknown timezone keys can match identical source timestamps, but conflicting
    timestamps across overlapping exports quarantine the semantic group. Updates
    require distinct, known export times. Import order never determines precedence.
    """
    if observations.empty:
        return (
            pd.DataFrame(),
            observations.copy(),
            {"observations": 0, "current_records": 0, "unresolved_observations": 0},
        )
    audit = observations.copy().reset_index(drop=True)
    audit["_ingest_record_id"] = ""
    audit["_ingest_disposition"] = "unresolved"
    audit["_ingest_reason"] = ""
    audit["_ingest_changed_fields"] = "[]"
    audit["_ingest_revision"] = 0
    audit["_ingest_is_current"] = False
    semantics = []
    coarse_keys = []
    times = []
    for _, row in audit.iterrows():
        fields = {
            key: str(row.get(key, "")) if pd.notna(row.get(key, "")) else ""
            for key in IDENTITY_FIELDS
        }
        event_value = row.get("event_name", "")
        event = fields["game_id"] or (str(event_value) if pd.notna(event_value) else "")
        required = [
            fields[k]
            for k in ("sportsbook", "market_name", "bet_name", "odds", "stake", "bet_type")
        ]
        time_key = _time_key(row)
        try:
            economics_valid = (
                math.isfinite(float(fields["stake"]))
                and float(fields["stake"]) > 0
                and math.isfinite(float(fields["odds"]))
                and float(fields["odds"]) != 0
            )
        except ValueError:
            economics_valid = False
        valid = (
            all(value.strip() for value in required)
            and bool(event.strip())
            and time_key is not None
            and economics_valid
        )
        semantics.append(
            fingerprint([row["_ingest_namespace"], fields, event])
            if valid
            else "missing:" + row["_ingest_observation_id"]
        )
        coarse_keys.append(
            fingerprint(
                [
                    row["_ingest_namespace"],
                    {k: v for k, v in fields.items() if k not in {"odds", "stake"}},
                    event,
                ]
            )
            if valid
            else "missing:" + row["_ingest_observation_id"]
        )
        times.append(time_key or "")
    audit["_ingest_semantic_key"] = semantics
    audit["_ingest_time_key"] = times
    audit["_ingest_candidate_key"] = coarse_keys
    blocked = set()
    for _, candidates in audit.groupby("_ingest_candidate_key", sort=True):
        if candidates["_ingest_import_id"].nunique() < 2:
            continue
        if (
            candidates["_ingest_time_key"].str.startswith("unverified:").any()
            and candidates["_ingest_time_key"].nunique() > 1
        ):
            blocked.update(candidates.index)
            audit.loc[candidates.index, "_ingest_reason"] = "uncertain_timestamp_overlap"
            continue
        for _, same_time in candidates.groupby("_ingest_time_key", sort=True):
            if (
                same_time["_ingest_import_id"].nunique() > 1
                and same_time["_ingest_semantic_key"].nunique() > 1
            ):
                blocked.update(same_time.index)
                audit.loc[same_time.index, "_ingest_reason"] = "possible_identity_field_correction"
    available = audit.loc[~audit.index.isin(blocked)]
    for semantic, group in available.groupby("_ingest_semantic_key", sort=True):
        idx = group.index
        if semantic.startswith("missing:"):
            audit.loc[idx, "_ingest_reason"] = "missing_identity_or_invalid_timestamp"
            continue
        unknown = group["_ingest_time_key"].str.startswith("unverified:").any()
        if (
            unknown
            and group["_ingest_import_id"].nunique() > 1
            and group["_ingest_time_key"].nunique() > 1
        ):
            audit.loc[idx, "_ingest_reason"] = "uncertain_timestamp_overlap"
            continue
        for time_key, candidates in group.groupby("_ingest_time_key", sort=True):
            ids = candidates.index
            if candidates["_ingest_import_id"].duplicated().any():
                audit.loc[ids, "_ingest_reason"] = "within_source_identity_collision"
                continue
            audit.loc[ids, "_ingest_record_id"] = "auto:" + fingerprint([semantic, time_key])
    seen = set()
    for decision in decisions or []:
        observation_id = decision["observation_id"]
        if observation_id in seen:
            raise ValueError("Multiple resolutions for one observation")
        seen.add(observation_id)
        if not decision.get("reason", "").strip() or not decision.get("record_id", "").strip():
            raise ValueError("Resolution requires record_id and reason")
        utc_time(decision["decided_at"])
        match = audit["_ingest_observation_id"].eq(observation_id)
        if not match.any():
            raise ValueError("Resolution refers to an unavailable observation")
        audit.loc[match, "_ingest_record_id"] = "reviewed:" + decision["record_id"]
        audit.loc[match, "_ingest_reason"] = "explicit_resolution: " + decision["reason"]
    linked = audit[audit["_ingest_record_id"].ne("")]
    for _, group in linked.groupby("_ingest_record_id", sort=True):
        if group["_ingest_namespace"].nunique() != 1:
            raise ValueError("Resolution cannot link different source namespaces")
        ids = group.index
        changed = group["_ingest_payload_hash"].nunique() > 1
        export_times = group["_ingest_exported_at"]
        if changed and (export_times.eq("").any() or export_times.duplicated().any()):
            audit.loc[ids, "_ingest_reason"] = "conflicting_payload_without_export_chronology"
            continue
        ordered = group.sort_values(
            ["_ingest_exported_at", "_ingest_import_id", "_ingest_source_row"]
        )
        revision = 0
        previous = None
        previous_row = None
        source_columns = [c for c in audit if not c.startswith("_ingest_")]
        for i, row in ordered.iterrows():
            payload = row["_ingest_payload_hash"]
            if payload != previous:
                revision += 1
                disposition = "new" if revision == 1 else "revised"
            else:
                disposition = "repeated_observation"
            if previous_row is not None:
                changes = [
                    c
                    for c in source_columns
                    if (None if pd.isna(row[c]) else row[c])
                    != (None if pd.isna(previous_row[c]) else previous_row[c])
                ]
                audit.at[i, "_ingest_changed_fields"] = json.dumps(changes, sort_keys=True)
            previous_row = row
            audit.at[i, "_ingest_revision"] = revision
            audit.at[i, "_ingest_disposition"] = disposition
            previous = payload
        audit.at[ordered.index[-1], "_ingest_is_current"] = True
    audit["_ingest_execution_status"] = [
        classify_execution(str(r.get("saved_filter_names", "") or ""), str(r.get("created_at", "")))
        for _, r in audit.fillna("").iterrows()
    ]
    audit["_ingest_execution_evidence"] = POLICY["provenance"]
    audit = audit.sort_values("_ingest_observation_id").reset_index(drop=True)
    current = audit[audit["_ingest_is_current"]].copy().reset_index(drop=True)
    summary = {
        "observations": len(audit),
        "current_records": len(current),
        "unresolved_observations": int(audit["_ingest_disposition"].eq("unresolved").sum()),
        "dispositions": {
            str(k): int(v) for k, v in audit["_ingest_disposition"].value_counts().items()
        },
        "reasons": {str(k): int(v) for k, v in audit["_ingest_reason"].value_counts().items() if k},
        "execution_status": {
            str(k): int(v) for k, v in current["_ingest_execution_status"].value_counts().items()
        },
        "settlement_status": {str(k): int(v) for k, v in current["status"].value_counts().items()},
    }
    assert sum(summary["dispositions"].values()) == len(audit)
    return current, audit, summary
