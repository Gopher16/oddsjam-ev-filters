"""
exchange_eval.py
===============================================================================

Purpose
-------
Build exchange-only liquidity and capacity diagnostics.

This module is separate from generic filter evaluation because exchange-style
venues behave differently from sportsbooks. On exchanges, deployability depends
on liquidity. The raw numeric `liquidity` field from OddsJam is currently
treated as unreliable, so this module uses `liquidity_bucket` as the canonical
capacity signal.

What this module provides
-------------------------
1. build_exchange_capacity_table(...)
   Produces one row per exchange filter with:
     - bucket-based liquidity coverage
     - bucket-proxy average / median liquidity
     - total stake
     - total EV
     - deployable dollar estimate
     - capacity ratio
     - liquidity-weighted EV proxy

Bucket-capacity design
----------------------
`liquidity_bucket` is mapped to representative dollar proxies. By default:
    <=500   ->   250
    500-1k  ->   750
    1k-2k   ->  1500
    2k-5k   ->  3500
    5k-10k  ->  7500
    >10k    -> 15000

These proxies are intentionally pragmatic rather than exact.

Typical usage
-------------
>>> from oddsjam_ev.analysis.exchange_eval import build_exchange_capacity_table
>>> exchange_capacity = build_exchange_capacity_table(df)
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BUCKET_PROXY_MAP: dict[str, float] = {
    "<=500": 250.0,
    "500-1k": 750.0,
    "1k-2k": 1500.0,
    "2k-5k": 3500.0,
    "5k-10k": 7500.0,
    ">10k": 15000.0,
}


def build_exchange_capacity_table(
    df: pd.DataFrame,
    filter_col: str = "saved_filter_names",
    regime_col: str = "regime",
    liquidity_bucket_col: str = "liquidity_bucket",
    stake_col: str = "stake",
    ev_col: str = "ev",
    bucket_proxy_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build exchange-only bucket-based liquidity / capacity diagnostics.

    Parameters
    ----------
    df
        Bet-level dataframe.
    filter_col
        Filter name column.
    regime_col
        Regime column.
    liquidity_bucket_col
        Liquidity bucket column.
    stake_col
        Stake column.
    ev_col
        EV dollars column.
    bucket_proxy_map
        Optional mapping from bucket labels to representative dollar capacity.

    Returns
    -------
    pd.DataFrame
        Exchange-only filter summary with bucket-based capacity metrics.
    """
    required = [filter_col, regime_col, liquidity_bucket_col, stake_col, ev_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    proxy_map = bucket_proxy_map or DEFAULT_BUCKET_PROXY_MAP

    work = df.copy()
    work = work.loc[work[regime_col].eq("exchange")].copy()

    work[stake_col] = pd.to_numeric(work[stake_col], errors="coerce")
    work[ev_col] = pd.to_numeric(work[ev_col], errors="coerce")

    work["bucket_capacity_proxy"] = (
        work[liquidity_bucket_col].astype("string").map(proxy_map).astype(float)
    )

    work["deployable_dollars"] = np.where(
        work["bucket_capacity_proxy"].notna() & work[stake_col].notna(),
        np.minimum(work[stake_col], work["bucket_capacity_proxy"]),
        np.nan,
    )

    work["liquidity_weighted_ev_component"] = work[ev_col] * work["bucket_capacity_proxy"]

    grouped = (
        work.groupby(filter_col, dropna=False)
        .agg(
            n_bets=(stake_col, "size"),
            liquidity_coverage=("bucket_capacity_proxy", lambda s: s.notna().mean()),
            avg_liquidity=("bucket_capacity_proxy", "mean"),
            median_liquidity=("bucket_capacity_proxy", "median"),
            total_stake=(stake_col, "sum"),
            total_ev=(ev_col, "sum"),
            capacity_estimate=("deployable_dollars", "sum"),
            total_liquidity=("bucket_capacity_proxy", "sum"),
            liquidity_weighted_ev_component=("liquidity_weighted_ev_component", "sum"),
        )
        .reset_index()
    )

    grouped["liquidity_weighted_ev"] = np.where(
        grouped["total_liquidity"] > 0,
        grouped["liquidity_weighted_ev_component"] / grouped["total_liquidity"],
        np.nan,
    )

    grouped["capacity_ratio"] = np.where(
        grouped["total_stake"] > 0,
        grouped["capacity_estimate"] / grouped["total_stake"],
        np.nan,
    )

    return grouped.sort_values("capacity_estimate", ascending=False).reset_index(drop=True)
