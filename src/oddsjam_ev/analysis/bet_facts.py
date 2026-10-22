"""
bet_facts.py
===============================================================================
Purpose
-------
Separate source records, candidate wagers, and explicit ticket legs.

What this module provides
-------------------------
Bet/leg tables, lineage links, and a disposition for every source observation.

Design Notes
------------
Ticket status is not inferred from leg status. Source monetary values are recorded
once per ticket only when current legs agree; ticket completeness stays unknown.
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.analysis.filter_attribution import resolve_labels, validate_registry
from oddsjam_ev.io.source_csv import fingerprint

SOURCE_FEATURES = (
    "sportsbook",
    "sport",
    "league",
    "event_name",
    "game_id",
    "event_start_date",
    "market_name",
    "bet_name",
    "odds",
    "clv",
    "clv_source",
    "percentage",
    "market_width",
    "liquidity",
    "tags",
    "is_live_bet",
    "is_free_bet",
    "is_odds_boost",
    "saved_filter_names",
    "created_at",
    "bet_type",
    "parlay_id",
)


def _consensus(series: pd.Series):
    values = series.fillna("").unique()
    return values[0] if len(values) == 1 else None


def _numeric_consensus(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce")
    return (
        float(values.iloc[0])
        if len(values) and np.isfinite(values).all() and values.nunique() == 1
        else None
    )


def _utc_timestamp(raw: str, timezone: str):
    try:
        value = pd.Timestamp(raw)
        if pd.isna(value):
            return pd.NaT
        if value.tzinfo is None:
            if timezone == "unknown":
                return pd.NaT
            value = value.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        return value.tz_convert("UTC") if pd.notna(value) else pd.NaT
    except (TypeError, ValueError):
        return pd.NaT


def build_bet_facts(
    observations: pd.DataFrame, cfg: CanonicalConfig, registry: dict
) -> dict[str, pd.DataFrame]:
    """Build wager facts, explicit legs, current lineage, and full source audit.

    Parameters
    ----------
    observations : pandas.DataFrame
        Pinned ingestion observation audit, including unresolved records/history.
    cfg : CanonicalConfig
        Declared economic policy.
    registry : dict
        Dated registry for negative Pick'em eligibility evidence only.

    Returns
    -------
    dict[str, pandas.DataFrame]
        bet_facts, bet_legs, lineage, and source_audit. Original source remains in
        ingestion, linked by observation IDs; leg tables contain no ticket stake.
    """
    mapping = validate_registry(registry)
    if set(cfg.pickem_filter_ids) - set(mapping.values()):
        raise ValueError("Unknown pickem_filter_ids in economic policy")
    required = {
        "_ingest_observation_id",
        "_ingest_record_id",
        "_ingest_namespace",
        "_ingest_is_current",
        "_ingest_disposition",
        "_ingest_reason",
        "_ingest_timestamp_timezone",
        "_ingest_execution_status",
        "_ingest_execution_evidence",
        "stake",
        "bet_profit",
        "potential_payout",
        "status",
    }
    missing = required - set(observations)
    if missing:
        raise ValueError(f"Canonical source audit missing columns: {sorted(missing)}")
    if observations._ingest_observation_id.duplicated().any():
        raise ValueError("Observation IDs must be unique")
    work = observations.copy().fillna("")
    for column in SOURCE_FEATURES:
        if column not in work:
            work[column] = ""
    current_mask = work._ingest_is_current.eq(True)
    current = work[current_mask].copy()
    if current._ingest_record_id.eq("").any() or current._ingest_record_id.duplicated().any():
        raise ValueError("Current records must have unique nonempty identities")
    current["bet_id"] = "bet:" + current._ingest_record_id
    # A parlay ID is ticket evidence even if bet_type is inconsistent.
    explicit = current.parlay_id.ne("")
    current.loc[explicit, "bet_id"] = [
        "ticket:" + fingerprint([r["_ingest_namespace"], r["parlay_id"]])
        for r in current.loc[explicit].to_dict("records")
    ]
    facts = []
    legs = []
    for row in current.loc[~explicit].to_dict("records"):
        ids, _ = resolve_labels(row["saved_filter_names"], mapping)
        guard = bool(set(ids) & set(cfg.pickem_filter_ids)) or any(
            ident in cfg.pickem_filter_ids and alias in row["saved_filter_names"]
            for alias, ident in mapping.items()
        )
        facts.append(
            {
                "bet_id": row["bet_id"],
                "fact_grain": "source_single_candidate",
                "source_record_id": row["_ingest_record_id"],
                "namespace": row["_ingest_namespace"],
                **{c: row[c] for c in SOURCE_FEATURES},
                "reported_stake_usd": pd.to_numeric(row["stake"], errors="coerce"),
                "reported_profit_usd": pd.to_numeric(row["bet_profit"], errors="coerce"),
                "reported_potential_payout_usd": pd.to_numeric(
                    row["potential_payout"], errors="coerce"
                ),
                "settlement_status": row["status"].strip().lower(),
                "product": cfg.product(row["sportsbook"]),
                "pickem_label_guard": guard,
                "identity_status": "ingestion_candidate",
                "ticket_completeness": "not_applicable",
                "unresolved_leg_observations": 0,
                "observed_leg_count": 0,
                "ticket_economics_consistent": False,
                "execution_status": row["_ingest_execution_status"],
                "execution_evidence": row["_ingest_execution_evidence"],
                "timestamp_timezone": row["_ingest_timestamp_timezone"],
                "created_at_utc": _utc_timestamp(
                    row["created_at"], row["_ingest_timestamp_timezone"]
                ),
                "source_observation_id": row["_ingest_observation_id"],
            }
        )
    unresolved = work._ingest_disposition.eq("unresolved")
    ticket_source = work[(current_mask | unresolved) & work.parlay_id.ne("")]
    for (namespace, parlay_id), group in ticket_source.groupby(
        ["_ingest_namespace", "parlay_id"], sort=True
    ):
        accepted = group[group._ingest_is_current.eq(True)]
        bet_id = "ticket:" + fingerprint([namespace, parlay_id])
        amounts = {
            key: _numeric_consensus(accepted[column])
            for key, column in [
                ("reported_stake_usd", "stake"),
                ("reported_profit_usd", "bet_profit"),
                ("reported_potential_payout_usd", "potential_payout"),
            ]
        }
        venue = _consensus(accepted.sportsbook) or ""
        created = _consensus(accepted.created_at) or ""
        zone = _consensus(accepted._ingest_timestamp_timezone) or "unknown"
        execution = _consensus(accepted._ingest_execution_status) or "unknown"
        bad_count = int(group._ingest_disposition.eq("unresolved").sum())
        facts.append(
            {
                "bet_id": bet_id,
                "fact_grain": "ticket_candidate",
                "source_record_id": "",
                "namespace": namespace,
                **{c: "" for c in SOURCE_FEATURES},
                "sportsbook": venue,
                "created_at": created,
                "parlay_id": parlay_id,
                "bet_type": "parlay",
                **amounts,
                "settlement_status": "ticket_settlement_unverified",
                "product": cfg.product(venue),
                "pickem_label_guard": False,
                "identity_status": "unresolved_members" if bad_count else "ticket_candidate",
                "ticket_completeness": "unverified",
                "unresolved_leg_observations": bad_count,
                "observed_leg_count": len(accepted),
                "ticket_economics_consistent": bool(
                    len(accepted) and all(v is not None for v in amounts.values()) and venue
                ),
                "execution_status": execution,
                "execution_evidence": _consensus(accepted._ingest_execution_evidence)
                or "mixed_or_unknown",
                "timestamp_timezone": zone,
                "created_at_utc": _utc_timestamp(created, zone),
                "source_observation_id": "",
            }
        )
        for row in group.to_dict("records"):
            legs.append(
                {
                    "bet_id": bet_id,
                    "leg_id": row["_ingest_observation_id"],
                    "source_record_id": row["_ingest_record_id"],
                    "source_disposition": row["_ingest_disposition"],
                    "leg_status": row["status"],
                    **{c: row[c] for c in SOURCE_FEATURES},
                }
            )
    fact_columns = [
        "bet_id",
        "fact_grain",
        "source_record_id",
        "namespace",
        *SOURCE_FEATURES,
        "reported_stake_usd",
        "reported_profit_usd",
        "reported_potential_payout_usd",
        "settlement_status",
        "product",
        "pickem_label_guard",
        "identity_status",
        "ticket_completeness",
        "unresolved_leg_observations",
        "observed_leg_count",
        "ticket_economics_consistent",
        "execution_status",
        "execution_evidence",
        "timestamp_timezone",
        "created_at_utc",
        "source_observation_id",
    ]
    bets = pd.DataFrame(facts, columns=fact_columns).sort_values("bet_id").reset_index(drop=True)
    bets["created_at_utc"] = pd.to_datetime(bets["created_at_utc"], utc=True)
    for column in ("reported_stake_usd", "reported_profit_usd", "reported_potential_payout_usd"):
        bets[column] = pd.to_numeric(bets[column], errors="coerce")
    leg_frame = (
        pd.DataFrame(
            legs,
            columns=[
                "bet_id",
                "leg_id",
                "source_record_id",
                "source_disposition",
                "leg_status",
                *SOURCE_FEATURES,
            ],
        )
        .sort_values("leg_id")
        .reset_index(drop=True)
    )
    audit = work[
        ["_ingest_observation_id", "_ingest_record_id", "_ingest_disposition", "_ingest_reason"]
    ].copy()
    by_record = dict(zip(current._ingest_record_id, current.bet_id, strict=True))
    audit["bet_id"] = audit._ingest_record_id.map(by_record).astype(object)
    audit["canonical_disposition"] = "historical_observation"
    audit.loc[current_mask, "canonical_disposition"] = "current_single"
    audit.loc[current_mask & work.parlay_id.ne(""), "canonical_disposition"] = "current_ticket_leg"
    audit.loc[unresolved, "canonical_disposition"] = "unresolved_source"
    for idx in work.index[unresolved & work.parlay_id.ne("")]:
        audit.loc[idx, "bet_id"] = "ticket:" + fingerprint(
            [work.loc[idx, "_ingest_namespace"], work.loc[idx, "parlay_id"]]
        )
    return {"bet_facts": bets, "bet_legs": leg_frame, "lineage": current, "source_audit": audit}
