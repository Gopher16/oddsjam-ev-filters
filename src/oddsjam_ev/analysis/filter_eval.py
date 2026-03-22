"""
filter_eval.py
===============================================================================

Purpose
-------
Build canonical one-row-per-filter evaluation tables from bet-level data.

This module is the core "decision layer" for downstream betting-filter research.
It converts execution-level bet history into standardized, regime-aware filter
summaries that can be used in notebooks, reports, and promotion / kill logic.

What this module provides
-------------------------
1. FilterEvalConfig
   Central configuration for the expected input columns.

2. build_filter_evaluation_table(...)
   Returns one row per filter (and regime), including:
     - volume / exposure
     - realized performance
     - expected performance
     - drawdown
     - bootstrap ROI confidence intervals
     - duplicate / opportunity-level diagnostics
     - sample tier labels
     - production-filter tagging
     - edge capture / survivability metrics

Design Notes
------------
- Evaluation is regime-aware:
    exchange and sportsbook rows should not be mixed in ranking logic.
- Evaluation is execution-level by default:
    duplicate_ratio and n_unique_opportunities are included to help downstream
    compare execution-level vs opportunity-level behavior.
- This module does not enforce promotion thresholds.
  It standardizes the raw evidence those decisions should rely on.
- Production awareness is config-driven:
    if a prod_filter column is present, this module can propagate that state into
    the summary row so downstream notebooks can cleanly separate live filters
    from test filters.
- Capture metrics are intended to answer a more practical question than raw EV:
    not just "does this filter look theoretically good?" but
    "how much of the theoretical edge is actually realized in execution?"

Typical usage
-------------
>>> from oddsjam_ev.analysis.filter_eval import (
...     FilterEvalConfig,
...     build_filter_evaluation_table,
... )
>>> cfg = FilterEvalConfig()
>>> summary = build_filter_evaluation_table(df, cfg=cfg, regime="exchange")
===============================================================================
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterEvalConfig:
    """
    Configuration for filter-level evaluation.

    Attributes
    ----------
    filter_col
        Column containing filter labels / names.
    regime_col
        Column containing regime labels such as 'exchange' or 'sportsbook'.
    stake_col
        Stake amount column.
    profit_col
        Realized bet profit column.
    ev_col
        Per-bet EV in dollars.
    ev_roi_col
        Per-bet EV ROI.
    clv_col
        CLV column, retained here for future expansion even if not yet aggregated.
    created_col
        Bet placement timestamp column.
    opportunity_col
        Opportunity-level identifier column.
    prod_filter_col
        Optional boolean / binary column identifying configured production filters.
    settled_statuses
        Status values considered settled for evaluation.
    status_col
        Bet status column.
    bootstrap_iterations
        Number of bootstrap iterations used for ROI confidence intervals.
    random_state
        Random seed for reproducibility.
    """

    filter_col: str = "saved_filter_names"
    regime_col: str = "regime"
    stake_col: str = "stake"
    profit_col: str = "bet_profit"
    ev_col: str = "ev"
    ev_roi_col: str = "ev_roi"
    clv_col: str = "clv"
    created_col: str = "created_at_et"
    opportunity_col: str = "opportunity_id"
    prod_filter_col: str = "prod_filter"
    settled_statuses: tuple[str, ...] = ("won", "lost", "refunded")
    status_col: str = "status"
    bootstrap_iterations: int = 2000
    random_state: int = 42


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """
    Validate that all required columns are present.

    Parameters
    ----------
    df
        Input dataframe.
    required
        Required column names.

    Raises
    ------
    KeyError
        If one or more required columns are missing.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Coerce selected columns to numeric.

    Parameters
    ----------
    df
        Input dataframe.
    cols
        Columns to coerce.

    Returns
    -------
    pd.DataFrame
        Copy with selected columns coerced to numeric where present.
    """
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_actual_roi(series_profit: pd.Series, series_stake: pd.Series) -> pd.Series:
    """
    Compute per-bet realized ROI.

    Parameters
    ----------
    series_profit
        Realized profit series.
    series_stake
        Stake series.

    Returns
    -------
    pd.Series
        Per-bet realized ROI.
    """
    return pd.Series(
        np.where(series_stake > 0, series_profit / series_stake, np.nan),
        index=series_profit.index,
    )


def _max_drawdown(profit_series: pd.Series) -> float:
    """
    Compute max drawdown from an ordered profit series.

    Parameters
    ----------
    profit_series
        Realized profit values ordered by time.

    Returns
    -------
    float
        Minimum drawdown value.
    """
    if profit_series.empty:
        return np.nan

    cum_profit = profit_series.cumsum()
    running_peak = cum_profit.cummax()
    drawdown = cum_profit - running_peak
    return float(drawdown.min())


def _bootstrap_roi_ci(
    profits: np.ndarray,
    stakes: np.ndarray,
    n_boot: int,
    random_state: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Bootstrap ROI confidence interval.

    Parameters
    ----------
    profits
        Per-bet realized profits.
    stakes
        Per-bet stakes.
    n_boot
        Number of bootstrap iterations.
    random_state
        Random seed.
    alpha
        Confidence level tail probability.

    Returns
    -------
    tuple[float, float]
        Lower and upper ROI confidence interval bounds.
    """
    if len(profits) == 0 or np.nansum(stakes) <= 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    n = len(profits)
    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_profit = np.nansum(profits[idx])
        boot_stake = np.nansum(stakes[idx])
        boot[i] = boot_profit / boot_stake if boot_stake > 0 else np.nan

    return (
        float(np.nanquantile(boot, alpha / 2)),
        float(np.nanquantile(boot, 1 - alpha / 2)),
    )


def _sample_tier(n_bets: int) -> str:
    """
    Map bet count to a qualitative sample tier.

    Parameters
    ----------
    n_bets
        Number of bets.

    Returns
    -------
    str
        Sample tier label.
    """
    if n_bets < 100:
        return "thin"
    if n_bets < 500:
        return "moderate"
    return "robust"


def _capture_band(total_ev_roi: float, total_actual_roi: float) -> str:
    """
    Map EV-vs-realized relationship into a practical capture label.

    Parameters
    ----------
    total_ev_roi
        Headline EV ROI at the filter level.
    total_actual_roi
        Headline realized ROI at the filter level.

    Returns
    -------
    str
        Human-readable capture / survivability label.

    Notes
    -----
    This is intentionally heuristic. It exists to create a lightweight
    operational label for notebooks and scorecards, not a statistically
    rigorous test.
    """
    if pd.isna(total_ev_roi) or pd.isna(total_actual_roi):
        return "unknown"

    if total_ev_roi <= 0:
        if total_actual_roi > 0:
            return "positive_realized_negative_ev"
        return "negative_ev"

    ratio = total_actual_roi / total_ev_roi if total_ev_roi != 0 else np.nan
    if pd.isna(ratio):
        return "unknown"
    if ratio < 0:
        return "broken"
    if ratio < 0.5:
        return "weak_capture"
    if ratio < 1.0:
        return "partial_capture"
    if ratio < 1.5:
        return "full_capture"
    return "outperforming_ev"


def _build_filter_opportunity_stats(
    df: pd.DataFrame,
    filter_col: str,
    regime_col: str,
    opportunity_col: str,
) -> pd.DataFrame:
    """
    Build opportunity-level duplicate diagnostics by filter and regime.

    Parameters
    ----------
    df
        Bet-level dataframe.
    filter_col
        Filter label column.
    regime_col
        Regime column.
    opportunity_col
        Opportunity ID column.

    Returns
    -------
    pd.DataFrame
        Filter/regime-level duplicate diagnostics.
    """
    if opportunity_col not in df.columns:
        unique_pairs = df[[filter_col, regime_col]].drop_duplicates()
        unique_pairs["n_unique_opportunities"] = np.nan
        unique_pairs["duplicate_ratio"] = np.nan
        return unique_pairs.reset_index(drop=True)

    grouped = (
        df.groupby([filter_col, regime_col], dropna=False)
        .agg(
            n_rows=(opportunity_col, "size"),
            n_unique_opportunities=(opportunity_col, "nunique"),
        )
        .reset_index()
    )

    grouped["duplicate_ratio"] = np.where(
        grouped["n_unique_opportunities"] > 0,
        grouped["n_rows"] / grouped["n_unique_opportunities"],
        np.nan,
    )

    return grouped[[filter_col, regime_col, "n_unique_opportunities", "duplicate_ratio"]]


def _build_prod_filter_stats(
    df: pd.DataFrame,
    *,
    filter_col: str,
    regime_col: str,
    prod_filter_col: str,
) -> pd.DataFrame:
    """
    Aggregate production-filter tagging by filter and regime.

    Parameters
    ----------
    df
        Bet-level dataframe.
    filter_col
        Filter label column.
    regime_col
        Regime column.
    prod_filter_col
        Source production flag column.

    Returns
    -------
    pd.DataFrame
        One row per filter/regime with boolean production status.

    Notes
    -----
    The summary uses `.any()` intentionally:
    if any execution row for the filter/regime is marked as production,
    the filter summary is treated as production-aware.
    """
    if prod_filter_col not in df.columns:
        unique_pairs = df[[filter_col, regime_col]].drop_duplicates()
        unique_pairs["is_prod_filter"] = np.nan
        return unique_pairs.reset_index(drop=True)

    grouped = (
        df.groupby([filter_col, regime_col], dropna=False)[prod_filter_col]
        .agg(lambda s: bool(pd.Series(s).fillna(False).astype(bool).any()))
        .reset_index()
        .rename(columns={prod_filter_col: "is_prod_filter"})
    )
    return grouped


def build_filter_evaluation_table(
    df: pd.DataFrame,
    cfg: FilterEvalConfig | None = None,
    regime: str | None = None,
    settled_only: bool = True,
) -> pd.DataFrame:
    """
    Build a canonical one-row-per-filter evaluation table.

    Parameters
    ----------
    df
        Bet-level dataframe.
    cfg
        Optional evaluation config. Defaults to FilterEvalConfig().
    regime
        Optional regime filter. Example: 'exchange' or 'sportsbook'.
    settled_only
        If True, limit evaluation to settled statuses.

    Returns
    -------
    pd.DataFrame
        Filter-level evaluation table.

    Output fields
    -------------
    Core exposure / performance
      - bet_count
      - total_stake
      - total_profit
      - total_ev
      - avg_ev_roi
      - avg_actual_roi
      - total_ev_roi
      - total_actual_roi
      - avg_clv
      - first_bet_ts
      - last_bet_ts
      - active_days

    Diagnostics
      - max_drawdown
      - roi_ci_low
      - roi_ci_high
      - n_unique_opportunities
      - duplicate_ratio
      - sample_tier

    Production / capture extensions
      - is_prod_filter
      - profit_minus_ev
      - ev_realization_gap
      - edge_capture_ratio
      - capture_band
    """
    cfg = cfg or FilterEvalConfig()

    required = [
        cfg.filter_col,
        cfg.regime_col,
        cfg.stake_col,
        cfg.profit_col,
        cfg.ev_col,
        cfg.ev_roi_col,
        cfg.created_col,
    ]
    _validate_columns(df, required)

    work = _coerce_numeric(
        df,
        [cfg.stake_col, cfg.profit_col, cfg.ev_col, cfg.ev_roi_col, cfg.clv_col],
    )

    if regime is not None:
        work = work.loc[work[cfg.regime_col].eq(regime)].copy()

    if settled_only and cfg.status_col in work.columns:
        settled_norm = {str(x).strip().lower() for x in cfg.settled_statuses}
        work = work.loc[
            work[cfg.status_col].astype("string").str.strip().str.lower().isin(settled_norm)
        ].copy()

    work[cfg.created_col] = pd.to_datetime(work[cfg.created_col], errors="coerce")
    work["actual_roi"] = _build_actual_roi(work[cfg.profit_col], work[cfg.stake_col])
    work["bet_date"] = work[cfg.created_col].dt.date

    grouped = (
        work.groupby([cfg.filter_col, cfg.regime_col], dropna=False)
        .agg(
            bet_count=(cfg.stake_col, "size"),
            total_stake=(cfg.stake_col, "sum"),
            total_profit=(cfg.profit_col, "sum"),
            total_ev=(cfg.ev_col, "sum"),
            avg_ev_roi=(cfg.ev_roi_col, "mean"),
            avg_actual_roi=("actual_roi", "mean"),
            avg_clv=(cfg.clv_col, "mean"),
            first_bet_ts=(cfg.created_col, "min"),
            last_bet_ts=(cfg.created_col, "max"),
            active_days=("bet_date", "nunique"),
        )
        .reset_index()
    )

    grouped["total_ev_roi"] = np.where(
        grouped["total_stake"] > 0,
        grouped["total_ev"] / grouped["total_stake"],
        np.nan,
    )
    grouped["total_actual_roi"] = np.where(
        grouped["total_stake"] > 0,
        grouped["total_profit"] / grouped["total_stake"],
        np.nan,
    )
    grouped["sample_tier"] = grouped["bet_count"].map(_sample_tier)

    # Capture / survivability diagnostics
    grouped["profit_minus_ev"] = grouped["total_profit"] - grouped["total_ev"]
    grouped["ev_realization_gap"] = grouped["total_actual_roi"] - grouped["total_ev_roi"]
    grouped["edge_capture_ratio"] = np.where(
        grouped["total_ev_roi"].abs() > 0,
        grouped["total_actual_roi"] / grouped["total_ev_roi"],
        np.nan,
    )
    grouped["capture_band"] = grouped.apply(
        lambda row: _capture_band(row["total_ev_roi"], row["total_actual_roi"]),
        axis=1,
    )

    drawdown_rows: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []

    for (filter_name, filter_regime), subdf in work.groupby(
        [cfg.filter_col, cfg.regime_col],
        dropna=False,
    ):
        ordered = subdf.sort_values(cfg.created_col)

        drawdown_rows.append(
            {
                cfg.filter_col: filter_name,
                cfg.regime_col: filter_regime,
                "max_drawdown": _max_drawdown(ordered[cfg.profit_col]),
            }
        )

        roi_ci_low, roi_ci_high = _bootstrap_roi_ci(
            profits=ordered[cfg.profit_col].to_numpy(dtype=float),
            stakes=ordered[cfg.stake_col].to_numpy(dtype=float),
            n_boot=cfg.bootstrap_iterations,
            random_state=cfg.random_state,
        )
        ci_rows.append(
            {
                cfg.filter_col: filter_name,
                cfg.regime_col: filter_regime,
                "roi_ci_low": roi_ci_low,
                "roi_ci_high": roi_ci_high,
            }
        )

    drawdown_df = pd.DataFrame(drawdown_rows)
    ci_df = pd.DataFrame(ci_rows)

    grouped = grouped.merge(drawdown_df, on=[cfg.filter_col, cfg.regime_col], how="left")
    grouped = grouped.merge(ci_df, on=[cfg.filter_col, cfg.regime_col], how="left")

    opp_stats = _build_filter_opportunity_stats(
        work,
        filter_col=cfg.filter_col,
        regime_col=cfg.regime_col,
        opportunity_col=cfg.opportunity_col,
    )
    grouped = grouped.merge(
        opp_stats,
        on=[cfg.filter_col, cfg.regime_col],
        how="left",
    )

    prod_stats = _build_prod_filter_stats(
        work,
        filter_col=cfg.filter_col,
        regime_col=cfg.regime_col,
        prod_filter_col=cfg.prod_filter_col,
    )
    grouped = grouped.merge(
        prod_stats,
        on=[cfg.filter_col, cfg.regime_col],
        how="left",
    )

    return grouped.sort_values(
        [cfg.regime_col, "total_profit"],
        ascending=[True, False],
    ).reset_index(drop=True)
