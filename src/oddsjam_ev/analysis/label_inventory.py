"""
label_inventory.py
===============================================================================
Purpose
-------
Inventory exact historical source labels without inferring aliases or settings.

What this module provides
-------------------------
Separate source/current/wager counts, lineage, lexical time bounds and overlap audits.

Design Notes
------------
Source strings are never split or normalized. All counts retain their grain;
shared wager candidates and overlapping filter memberships are not additive.
===============================================================================
"""

from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd

from oddsjam_ev.analysis.filter_attribution import resolve_labels, validate_registry
from oddsjam_ev.io.source_csv import fingerprint


def _label(value):
    if value is None or pd.isna(value):
        return None
    if not isinstance(value, str):
        raise ValueError("Source labels must be strings or null")
    return value


def _bounds(values):
    values = [v for v in values if isinstance(v, str) and v != ""]
    return {
        "raw_created_at_lexical_min": min(values) if values else None,
        "raw_created_at_lexical_max": max(values) if values else None,
    }


def build_label_inventory(observations, current, tables, registry):
    """Build non-economic inventories using verified snapshot and canonical tables.

    Parameters
    ----------
    observations, current : pandas.DataFrame
        Full snapshot audit and its current subset, including source lineage.
    tables : dict[str, pandas.DataFrame]
        Verified canonical source audit, wager facts and attribution bridge.
    registry : dict
        Exact registry pinned by the canonical bundle; no new aliases are applied.

    Returns
    -------
    dict
        JSON-compatible summary and tables; raw labels and time strings unchanged.
    """
    required = {
        "_ingest_observation_id",
        "_ingest_record_id",
        "_ingest_import_id",
        "_ingest_namespace",
        "_ingest_timestamp_timezone",
        "_ingest_is_current",
        "saved_filter_names",
        "created_at",
    }
    if required - set(observations) or required - set(current):
        raise ValueError("Missing inventory source fields")
    for column in ("_ingest_record_id", "_ingest_import_id", "_ingest_namespace"):
        if observations[column].isna().any():
            raise ValueError("Missing source lineage identity")
    audit = tables["source_audit"]
    bets = tables["bet_facts"]
    bridge = tables["filter_attribution"]
    for frame in (observations, current, audit):
        if (
            frame._ingest_observation_id.isna().any()
            or frame._ingest_observation_id.duplicated().any()
        ):
            raise ValueError("Duplicate or missing source observation identity")
    if set(audit._ingest_observation_id) != set(observations._ingest_observation_id):
        raise ValueError("Canonical source audit does not cover the selected observations")
    selected = observations.loc[observations._ingest_is_current.eq(True)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(selected, current.reset_index(drop=True))
    except AssertionError:
        raise ValueError("Snapshot current rows do not reconcile") from None
    if bets.bet_id.isna().any() or bets.bet_id.duplicated().any():
        raise ValueError("Invalid canonical wager identities")
    known = set(bets.bet_id)
    linked = set(audit.bet_id.dropna())
    if not linked <= known or not set(bridge.bet_id) <= known:
        raise ValueError("Orphan canonical wager link")
    mapping = validate_registry(registry)
    obs = observations.copy()
    obs["raw_label"] = obs.saved_filter_names.map(_label)
    obs["label_id"] = obs.raw_label.map(lambda v: fingerprint({"raw_label": v}))
    obs = obs.merge(
        audit[["_ingest_observation_id", "bet_id", "canonical_disposition"]],
        on="_ingest_observation_id",
        validate="one_to_one",
    )
    current_ids = set(current._ingest_observation_id)
    obs["inventory_current"] = obs._ingest_observation_id.isin(current_ids)
    rows, imports, times, dispositions = [], [], [], []
    wager_labels = defaultdict(set)
    for label_id, group in obs.groupby("label_id", sort=True):
        raw = group.iloc[0].raw_label
        ids, status = resolve_labels(raw or "", mapping)
        wager_ids = set(group.bet_id.dropna())
        for bet in wager_ids:
            wager_labels[bet].add(label_id)
        rows.append(
            {
                "label_id": label_id,
                "raw_label": raw,
                "label_presence": "null" if raw is None else "empty" if raw == "" else "present",
                "registry_resolution_status": status,
                "resolved_filter_ids": ids,
                "source_observation_count": len(group),
                "distinct_source_record_count": group._ingest_record_id.nunique(),
                "current_observation_count": int(group.inventory_current.sum()),
                "linked_wager_candidate_count": len(wager_ids),
                "current_linked_wager_candidate_count": group.loc[
                    group.inventory_current, "bet_id"
                ].nunique(),
                "unlinked_source_observation_count": int(group.bet_id.isna().sum()),
                "historical_settings_status": "configuration_effectivity_unknown",
                **_bounds(group.created_at),
            }
        )
        for (namespace, import_id), part in group.groupby(
            ["_ingest_namespace", "_ingest_import_id"], dropna=False, sort=True
        ):
            imports.append(
                {
                    "label_id": label_id,
                    "namespace": namespace,
                    "import_id": import_id,
                    "source_observation_count": len(part),
                    "current_observation_count": int(part.inventory_current.sum()),
                    "linked_wager_candidate_count": part.bet_id.nunique(),
                }
            )
        for timezone, part in group.groupby("_ingest_timestamp_timezone", dropna=False, sort=True):
            times.append(
                {
                    "label_id": label_id,
                    "source_timezone": None if pd.isna(timezone) else timezone,
                    "source_observation_count": len(part),
                    **_bounds(part.created_at),
                    "bounds_semantics": "lexical_raw_source_strings_not_verified_chronology",
                }
            )
        for disposition, part in group.groupby("canonical_disposition", dropna=False, sort=True):
            dispositions.append(
                {
                    "label_id": label_id,
                    "canonical_disposition": disposition,
                    "source_observation_count": len(part),
                    "linked_wager_candidate_count": part.bet_id.nunique(),
                }
            )
    raw_pairs = Counter(
        pair for values in wager_labels.values() for pair in combinations(sorted(values), 2)
    )
    memberships = defaultdict(set)
    for row in bridge.loc[bridge.attribution_status.eq("resolved")].itertuples():
        if pd.isna(row.filter_id):
            raise ValueError("Resolved membership has no filter identity")
        memberships[row.bet_id].add(row.filter_id)
    filter_pairs = Counter(
        pair for values in memberships.values() for pair in combinations(sorted(values), 2)
    )
    status_rows = [
        {
            "attribution_status": str(status),
            "attribution_row_count": len(part),
            "distinct_wager_candidate_count": part.bet_id.nunique(),
        }
        for status, part in bridge.groupby("attribution_status", sort=True)
    ]
    unique_labels = {row["label_id"]: row for row in rows}
    nulls = int(obs.raw_label.isna().sum())
    blank = int(obs.raw_label.eq("").sum())
    summary = {
        "source_observations": len(obs),
        "current_source_observations": len(current),
        "distinct_source_records": obs._ingest_record_id.nunique(),
        "canonical_wager_candidates": len(bets),
        "source_linked_wager_candidates": len(wager_labels),
        "unlinked_source_observations": int(obs.bet_id.isna().sum()),
        "distinct_exact_label_values_including_missing": len(unique_labels),
        "distinct_nonempty_label_strings": sum(r["label_presence"] == "present" for r in rows),
        "null_label_observations": nulls,
        "empty_label_observations": blank,
        "wagers_with_multiple_raw_label_values": sum(len(v) > 1 for v in wager_labels.values()),
        "wagers_with_multiple_resolved_filter_memberships": sum(
            len(v) > 1 for v in memberships.values()
        ),
    }
    return {
        "summary": summary,
        "labels": rows,
        "label_import_counts": imports,
        "label_timestamp_bounds": times,
        "label_source_dispositions": dispositions,
        "attribution_status_counts": status_rows,
        "raw_label_overlaps": [
            {"left_label_id": a, "right_label_id": b, "shared_wager_candidates": count}
            for (a, b), count in sorted(raw_pairs.items())
        ],
        "resolved_filter_overlaps": [
            {"left_filter_id": a, "right_filter_id": b, "shared_wager_candidates": count}
            for (a, b), count in sorted(filter_pairs.items())
        ],
    }
