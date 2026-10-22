"""
opportunity_facts.py
===============================================================================
Purpose
-------
Describe logged selection/quote proxies without claiming independent opportunities.

What this module provides
-------------------------
Proxy opportunity rows and wager/leg links, with conservative identity fallbacks.

Design Notes
------------
No stake, profit, or representative execution is assigned to an opportunity proxy.
The source created timestamp is unverified as a quote observation or execution time.
===============================================================================
"""

from __future__ import annotations

import pandas as pd

from oddsjam_ev.io.source_csv import fingerprint


def build_opportunity_facts(lineage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build explicitly labelled proxies from current logged source records.

    Parameters
    ----------
    lineage : pandas.DataFrame
        Current lineage rows with bet IDs, source features, and ingestion metadata.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Non-economic opportunity proxies and one linkage per source observation.
        Missing identity evidence gets a unique observation-specific proxy.
    """
    rows = []
    links = []
    for row in lineage.to_dict("records"):
        observation = row["_ingest_observation_id"]
        namespace = row["_ingest_namespace"]
        game = row.get("game_id", "")
        event = row.get("event_name", "")
        start = row.get("event_start_date", "")
        market, selection = row.get("market_name", ""), row.get("bet_name", "")
        venue, odds, created = (
            row.get("sportsbook", ""),
            row.get("odds", ""),
            row.get("created_at", ""),
        )
        if game and market and selection:
            key = [namespace, "game_id", game, market, selection]
            quality = "event_selection_proxy"
        elif event and start and market and selection:
            key = [namespace, "event_name_and_raw_start", event, start, market, selection]
            quality = "event_name_proxy"
        else:
            key = [namespace, "observation_fallback", observation]
            quality = "observation_fallback"
        selection_id = "selection:" + fingerprint(key)
        if not all([venue, odds, created]):
            quote_key = [selection_id, observation]
            quality = "observation_fallback"
        else:
            quote_key = [selection_id, venue, odds, created, row["_ingest_timestamp_timezone"]]
        ident = "opportunity_proxy:" + fingerprint(quote_key)
        rows.append(
            {
                "opportunity_proxy_id": ident,
                "selection_proxy_id": selection_id,
                "namespace": namespace,
                "sportsbook": venue,
                "game_id": game,
                "event_name": event,
                "event_start_raw": start,
                "market_name": market,
                "bet_name": selection,
                "odds_raw": odds,
                "created_at_raw": created,
                "timestamp_timezone": row["_ingest_timestamp_timezone"],
                "identity_quality": quality,
                "unique_opportunity_verified": False,
                "timestamp_semantics": "source_created_at_unverified",
                "coverage_denominator": "logged_source_records_only",
            }
        )
        links.append(
            {
                "bet_id": row["bet_id"],
                "source_observation_id": observation,
                "source_record_id": row["_ingest_record_id"],
                "opportunity_proxy_id": ident,
            }
        )
    columns = [
        "opportunity_proxy_id",
        "selection_proxy_id",
        "namespace",
        "sportsbook",
        "game_id",
        "event_name",
        "event_start_raw",
        "market_name",
        "bet_name",
        "odds_raw",
        "created_at_raw",
        "timestamp_timezone",
        "identity_quality",
        "unique_opportunity_verified",
        "timestamp_semantics",
        "coverage_denominator",
    ]
    candidates = pd.DataFrame(rows, columns=columns)
    # Only invariant identity attributes are retained. Conflicting descriptive
    # metadata remains in source lineage, not an arbitrarily selected execution.
    result = []
    for ident, group in candidates.groupby("opportunity_proxy_id", sort=True):
        summary = {"opportunity_proxy_id": ident}
        for column in columns[1:]:
            unique = group[column].unique()
            summary[column] = unique[0] if len(unique) == 1 else None
        summary["linked_source_record_count"] = len(group)
        result.append(summary)
    opportunities = pd.DataFrame(result, columns=[*columns, "linked_source_record_count"])
    bridge = (
        pd.DataFrame(
            links,
            columns=["bet_id", "source_observation_id", "source_record_id", "opportunity_proxy_id"],
        )
        .sort_values("source_observation_id")
        .reset_index(drop=True)
    )
    return opportunities, bridge
