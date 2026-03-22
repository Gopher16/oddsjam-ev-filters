"""
scorecards.py
===============================================================================

Purpose
-------
Reusable scorecard builders for downstream betting-filter research.

This module exists to combine canonical filter-evaluation outputs with other
diagnostic layers (for example exchange-capacity metrics) into a single table
that is easier to rank, review, and use in decision workflows.

What this module provides
-------------------------
1. ExchangeScorecardConfig
   Configuration for exchange scorecard construction and ranking.

2. build_exchange_scorecard(...)
   Merges:
     - exchange filter evaluation table
     - exchange capacity table

   and adds:
     - production-aware ranking
     - EV capture diagnostics
     - simple scorecard sort order

Design Notes
------------
- This module does not replace the canonical evaluation table.
  It is a presentation / decision layer on top of it.
- It assumes the caller has already built a regime-filtered exchange evaluation
  table and a compatible exchange capacity table.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExchangeScorecardConfig:
    """
    Configuration for exchange scorecard construction.

    Attributes
    ----------
    filter_col
        Join key for filter name.
    prod_col
        Production-filter boolean column.
    sort_prod_first
        If True, sort production filters ahead of non-production filters.
    capacity_col
        Capacity metric column name from the capacity table.
    liquidity_weighted_ev_col
        Liquidity-weighted EV metric column name from the capacity table.
    """

    filter_col: str = "saved_filter_names"
    prod_col: str = "is_prod_filter"
    sort_prod_first: bool = True
    capacity_col: str = "capacity_estimate"
    liquidity_weighted_ev_col: str = "liquidity_weighted_ev"


def build_exchange_scorecard(
    filter_eval_exchange: pd.DataFrame,
    exchange_capacity: pd.DataFrame,
    cfg: ExchangeScorecardConfig | None = None,
) -> pd.DataFrame:
    """
    Build an exchange scorecard by merging filter evaluation and capacity outputs.

    Parameters
    ----------
    filter_eval_exchange
        Canonical exchange-only evaluation table.
    exchange_capacity
        Exchange capacity table keyed by filter name.
    cfg
        Optional scorecard config.

    Returns
    -------
    pd.DataFrame
        Exchange scorecard with merged metrics and practical ranking helpers.

    Notes
    -----
    This function is intentionally conservative:
    it only requires a shared filter column and gracefully degrades if some
    optional capacity columns are missing.
    """
    cfg = cfg or ExchangeScorecardConfig()

    if cfg.filter_col not in filter_eval_exchange.columns:
        raise KeyError(f"Missing required column '{cfg.filter_col}' in filter_eval_exchange.")
    if cfg.filter_col not in exchange_capacity.columns:
        raise KeyError(f"Missing required column '{cfg.filter_col}' in exchange_capacity.")

    merged = filter_eval_exchange.merge(
        exchange_capacity,
        on=cfg.filter_col,
        how="left",
        suffixes=("", "_capacity"),
    )

    if "total_ev_roi" in merged.columns and "total_actual_roi" in merged.columns:
        merged["ev_realization_gap"] = merged["total_actual_roi"] - merged["total_ev_roi"]

        merged["edge_capture_ratio"] = np.where(
            merged["total_ev_roi"].abs() > 0,
            merged["total_actual_roi"] / merged["total_ev_roi"],
            np.nan,
        )

    if "total_profit" in merged.columns and "total_ev" in merged.columns:
        merged["profit_minus_ev"] = merged["total_profit"] - merged["total_ev"]

    if cfg.capacity_col in merged.columns and "total_stake" in merged.columns:
        merged["capacity_to_stake_ratio"] = np.where(
            merged["total_stake"] > 0,
            merged[cfg.capacity_col] / merged["total_stake"],
            np.nan,
        )

    sort_cols: list[str] = []
    sort_ascending: list[bool] = []

    if cfg.sort_prod_first and cfg.prod_col in merged.columns:
        sort_cols.append(cfg.prod_col)
        sort_ascending.append(False)

    if "total_actual_roi" in merged.columns:
        sort_cols.append("total_actual_roi")
        sort_ascending.append(False)

    if cfg.capacity_col in merged.columns:
        sort_cols.append(cfg.capacity_col)
        sort_ascending.append(False)

    if "total_profit" in merged.columns:
        sort_cols.append("total_profit")
        sort_ascending.append(False)

    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=sort_ascending)

    return merged.reset_index(drop=True)
