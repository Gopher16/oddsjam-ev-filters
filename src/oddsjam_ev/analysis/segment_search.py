"""
segment_search.py
===============================================================================
Purpose
-------
Enumerate outcome-independent, fully accounted research families.

What this module provides
-------------------------
Stable predicates, unique wager memberships and complete candidate ledgers.

Design Notes
------------
Observed categories describe logged coverage, not all available market opportunities.
===============================================================================
"""

import json

import pandas as pd

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.io.source_csv import fingerprint


def enumerate_segments(
    wagers: pd.DataFrame, cfg: DiscoveryConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate predicates without inspecting outcomes or economic eligibility.

    Parameters
    ----------
    wagers : pandas.DataFrame
        Atlas wager dimensions, one row per candidate bet.
    cfg : DiscoveryConfig
        Declared views and separated execution cohorts.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Complete ledger and non-additive candidate membership links.
    """
    required = {"bet_id", "execution_status", "product"} | {c for v in cfg.views for c in v}
    if required - set(wagers):
        raise ValueError(f"Missing search columns: {sorted(required - set(wagers))}")
    if wagers.bet_id.isna().any() or wagers.bet_id.duplicated().any():
        raise ValueError("Search requires unique wager identities")
    records = []
    links = []
    data = wagers[wagers.execution_status.isin(cfg.execution_modes)].sort_values("bet_id")
    for view in [(), *cfg.views]:
        columns = ["execution_status", "product", *view]
        for key, group in data.groupby(columns, dropna=False, sort=True):
            predicate = dict(zip(view, key[2:], strict=True))
            definition = {"execution_status": key[0], "product": key[1], "predicate": predicate}
            ident = "candidate:" + fingerprint(definition)
            records.append(
                {
                    "candidate_id": ident,
                    "execution_status": key[0],
                    "product": key[1],
                    "view": "__".join(view) if view else "overall",
                    "predicate_json": json.dumps(predicate, sort_keys=True),
                    "candidate_count": len(group),
                    "coverage": "observed_logged_categories",
                    "role": "primary" if key[0] == "real_money" else "secondary",
                }
            )
            links.extend({"candidate_id": ident, "bet_id": b} for b in group.bet_id)
    return (
        pd.DataFrame(
            records,
            columns=[
                "candidate_id",
                "execution_status",
                "product",
                "view",
                "predicate_json",
                "candidate_count",
                "coverage",
                "role",
            ],
        )
        .sort_values("candidate_id")
        .reset_index(drop=True),
        pd.DataFrame(links, columns=["candidate_id", "bet_id"])
        .sort_values(["candidate_id", "bet_id"])
        .reset_index(drop=True),
    )
