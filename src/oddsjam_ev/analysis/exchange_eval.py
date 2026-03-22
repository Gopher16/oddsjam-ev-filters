"""
exchange_eval.py
===============================================================================

Purpose
-------
Build exchange-only liquidity and capacity diagnostics.

This module is intentionally separate from the generic filter evaluation module
because exchange-style venues behave differently from sportsbooks. On exchanges,
liquidity is a first-class part of edge quality and deployable scale.

What this module provides
-------------------------
1. build_exchange_capacity_table(...)
   Produces one row per exchange filter with:
     - liquidity coverage
     - average / median liquidity
     - total stake
     - total EV
     - deployable dollar estimate
     - capacity ratio
     - liquidity-weighted EV proxy

Design Notes
------------
- This module expects that preprocessing has already added a regime column.
- It only evaluates rows where regime == 'exchange'.
- Liquidity is interpreted as market depth / available size proxy.
- capacity_estimate is intentionally simple:
      sum(min(stake, liquidity))
  This is a pragmatic first-pass estimate of deployable dollars.

Typical usage
-------------
>>> from oddsjam_ev.analysis.exchange_eval import build_exchange_capacity_table
>>> exchange_capacity = build_exchange_capacity_table(df)
===============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_exchange_capacity_table(
    df: pd.DataFrame,
    filter_col: str = "saved_filter_names",
    regime_col: str = "regime",
    liquidity_col: str = "liquidity",
    stake_col: str = "stake",
    ev_col: str = "ev",
) -> pd.DataFrame:
    """
    Build exchange-only liquidity / capacity diagnostics.

    Parameters
    ----------
    df
        Bet-level dataframe.
    filter_col
        Filter name column.
    regime_col
        Regime column.
    liquidity_col
        Liquidity column.
    stake_col
        Stake column.
    ev_col
        EV dollars column.

    Returns
    -------
    pd.DataFrame
        Exchange-only filter summary with capacity metrics.
    """
    required = [filter_col, regime_col, liquidity_col, stake_col, ev_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work = work.loc[work[regime_col].eq("exchange")].copy()

    work[liquidity_col] = pd.to_numeric(work[liquidity_col], errors="coerce")
    work[stake_col] = pd.to_numeric(work[stake_col], errors="coerce")
    work[ev_col] = pd.to_numeric(work[ev_col], errors="coerce")

    work["deployable_dollars"] = np.where(
        work[liquidity_col].notna() & work[stake_col].notna(),
        np.minimum(work[stake_col], work[liquidity_col]),
        np.nan,
    )

    work["liquidity_weighted_ev_component"] = work[ev_col] * work[liquidity_col]

    grouped = (
        work.groupby(filter_col, dropna=False)
        .agg(
            n_bets=(stake_col, "size"),
            liquidity_coverage=(liquidity_col, lambda s: s.notna().mean()),
            avg_liquidity=(liquidity_col, "mean"),
            median_liquidity=(liquidity_col, "median"),
            total_stake=(stake_col, "sum"),
            total_ev=(ev_col, "sum"),
            capacity_estimate=("deployable_dollars", "sum"),
            total_liquidity=(liquidity_col, "sum"),
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
