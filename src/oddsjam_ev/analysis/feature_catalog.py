"""
feature_catalog.py
===============================================================================
Purpose
-------
Audit actual source feature availability and outcome-leakage constraints.

What this module provides
-------------------------
A compact long-format availability table for current logged records.

Design Notes
------------
Presence is not proof a value was known at bet/quote decision time. CLV and outcome
fields are explicitly post-decision or timing-unverified, never default predictors.
===============================================================================
"""

from __future__ import annotations

import pandas as pd

FEATURES = {
    "sportsbook": ("source_category", "source_current_unverified"),
    "sport": ("source_category", "source_current_unverified"),
    "league": ("source_category", "source_current_unverified"),
    "market_name": ("source_market_label", "source_current_unverified"),
    "bet_name": ("source_selection_label", "source_current_unverified"),
    "odds": ("american_odds", "source_current_unverified"),
    "percentage": ("advertised_ev_percent", "source_current_unverified"),
    "liquidity": ("source_liquidity_untrusted", "source_current_unverified"),
    "market_width": ("source_market_width", "source_current_unverified"),
    "tags": ("source_tags_not_verified_capacity", "source_current_unverified"),
    "created_at": ("naive_or_offset_source_timestamp", "execution_time_unverified"),
    "event_start_date": ("source_event_start", "timezone_and_revision_unverified"),
    "clv": ("closing_reference_american_odds", "post_decision_or_unverified"),
    "status": ("source_settlement_status", "outcome"),
    "bet_profit": ("reported_usd_profit", "outcome"),
    "settled_at": ("source_settlement_time", "outcome"),
    "updated_at": ("source_update_time", "revision"),
    "player": ("standalone_player_id", "unavailable_unless_observed"),
    "line": ("standalone_selection_line", "unavailable_unless_observed"),
}


def build_feature_catalog(lineage: pd.DataFrame) -> pd.DataFrame:
    """Report availability without synthesizing unavailable market features.

    Parameters
    ----------
    lineage : pandas.DataFrame
        Current source records; missing columns remain explicitly unavailable.

    Returns
    -------
    pandas.DataFrame
        Feature counts, units/meaning, and time-availability limitations.
    """
    rows = []
    for name, (meaning, timing) in FEATURES.items():
        exists = name in lineage
        count = int(lineage[name].fillna("").astype(str).str.strip().ne("").sum()) if exists else 0
        rows.append(
            [
                name,
                exists,
                count,
                len(lineage),
                count / len(lineage) if len(lineage) else None,
                meaning,
                timing,
                False,
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "source_column",
            "column_present",
            "nonempty_records",
            "denominator_records",
            "coverage_rate",
            "meaning",
            "temporal_role",
            "decision_time_verified",
        ],
    )
