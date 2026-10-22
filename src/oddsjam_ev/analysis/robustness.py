"""
robustness.py
===============================================================================
Purpose
-------
Estimate conditional event-proxy uncertainty and concentration sensitivity.

What this module provides
-------------------------
Cluster bootstrap ROI, event-profit tests, coverage and leave-one-event diagnostics.

Design Notes
------------
Event proxies are not proven independent. These outputs never authorize promotion.
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import t

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig


def event_totals(wagers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate eligible known-event wagers without guessing missing identities.

    Parameters
    ----------
    wagers : pandas.DataFrame
        Canonically eligible wagers with namespace and game IDs.

    Returns
    -------
    pandas.DataFrame
        Namespace/event clusters with summed eligible profit, stake and counts.
    """
    known = wagers.game_id.fillna("").astype(str).str.strip().ne("") & wagers.namespace.fillna(
        ""
    ).astype(str).str.strip().ne("")
    return (
        wagers.loc[known]
        .groupby(["namespace", "game_id"], sort=True)
        .agg(
            stake=("reported_stake_usd", "sum"),
            profit=("reported_profit_usd", "sum"),
            wager_count=("bet_id", "size"),
        )
        .reset_index()
    )


def cluster_bootstrap(groups: pd.DataFrame, cfg: DiscoveryConfig, seed: int) -> tuple[float, float]:
    """Resample entire events, preserving stake/profit pairing in the ROI ratio.

    Parameters
    ----------
    groups : pandas.DataFrame
        Stable ordered event totals.
    cfg : DiscoveryConfig
        Interval level and iteration count.
    seed : int
        Stable candidate-specific random seed.

    Returns
    -------
    tuple[float, float]
        Percentile interval, or null bounds for degenerate samples.
    """
    if len(groups) < cfg.min_event_groups:
        return np.nan, np.nan
    stake = groups.stake.to_numpy(float)
    profit = groups.profit.to_numpy(float)
    if not np.isfinite(stake).all() or not np.isfinite(profit).all() or (stake <= 0).any():
        raise ValueError("Cluster economics must be finite with positive stake")
    rng = np.random.default_rng(seed)
    draws = []
    batch = max(1, min(128, 1_000_000 // len(groups)))
    for start in range(0, cfg.bootstrap_resamples, batch):
        indexes = rng.integers(
            0, len(groups), size=(min(batch, cfg.bootstrap_resamples - start), len(groups))
        )
        draws.extend(profit[indexes].sum(axis=1) / stake[indexes].sum(axis=1))
    if np.ptp(draws) == 0:
        return np.nan, np.nan
    tail = (1 - cfg.confidence_level) / 2
    return tuple(float(x) for x in np.quantile(draws, [tail, 1 - tail]))


def event_profit_pvalue(groups: pd.DataFrame) -> float:
    """Approximate one-sided event-profit mean test against zero.

    Parameters
    ----------
    groups : pandas.DataFrame
        Event profit totals, assumed independent with finite variance.

    Returns
    -------
    float
        Student t upper-tail p-value; null for insufficient/degenerate inputs.

    Notes
    -----
    This is not a test on the mean of individual bet ROIs. It is approximate
    under skew and unequal event exposure; multiple testing cannot repair it.
    """
    x = groups.profit.to_numpy(float)
    if len(x) < 2 or not np.isfinite(x).all() or np.std(x, ddof=1) == 0:
        return np.nan
    statistic = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return float(t.sf(statistic, len(x) - 1))


def evaluate_candidate(wagers: pd.DataFrame, cfg: DiscoveryConfig, seed: int) -> dict:
    """Summarize one candidate and its conditional event-covered subset.

    Parameters
    ----------
    wagers : pandas.DataFrame
        Unique atlas wager members; excluded products stay excluded.
    cfg : DiscoveryConfig
        Evidence thresholds and numerical policy.
    seed : int
        Candidate-specific bootstrap seed.

    Returns
    -------
    dict
        Coverage, reported economics, conditional inference and sensitivities.
    """
    eligible = wagers.loc[wagers.reported_roi_eligible].sort_values("bet_id")
    stake = eligible.reported_stake_usd.sum(min_count=1)
    profit = eligible.reported_profit_usd.sum(min_count=1)
    groups = event_totals(eligible)
    gs = groups.stake.sum(min_count=1)
    gp = groups.profit.sum(min_count=1)
    count = int(groups.wager_count.sum())
    enough = len(groups) >= cfg.min_event_groups
    lo, hi = cluster_bootstrap(groups, cfg, seed) if enough else (np.nan, np.nan)
    reasons = []
    if not enough:
        reasons.append("insufficient_event_proxy_groups")
    if cfg.min_eligible_stake_usd is None:
        reasons.append("stake_threshold_unset")
    elif pd.isna(gs) or gs < cfg.min_eligible_stake_usd:
        reasons.append("insufficient_event_covered_stake")
    p = event_profit_pvalue(groups) if not reasons else np.nan
    if not reasons and pd.isna(p):
        reasons.append("degenerate_event_profit_distribution")
    loo = (gp - groups.profit) / (gs - groups.stake).where(gs - groups.stake > 0)
    largest_stake = groups.loc[groups.stake.idxmax()] if len(groups) else None
    positive = groups[groups.profit.gt(0)]
    largest_win = positive.loc[positive.profit.idxmax()] if len(positive) else None

    def removed(row):
        return (
            (gp - row.profit) / (gs - row.stake) if row is not None and gs > row.stake else np.nan
        )

    return {
        "eligible_count": len(eligible),
        "eligible_stake_usd": stake,
        "eligible_reported_profit_usd": profit,
        "reported_roi": profit / stake if pd.notna(stake) and stake > 0 else np.nan,
        "event_proxy_groups": len(groups),
        "event_covered_wagers": count,
        "missing_event_wagers": len(eligible) - count,
        "missing_event_stake_usd": stake - gs if len(groups) else stake,
        "missing_event_profit_usd": profit - gp if len(groups) else profit,
        "event_covered_stake_usd": gs,
        "event_covered_profit_usd": gp,
        "event_covered_roi": gp / gs if pd.notna(gs) and gs > 0 else np.nan,
        "event_coverage_rate": count / len(eligible) if len(eligible) else np.nan,
        "cluster_roi_ci_low": lo,
        "cluster_roi_ci_high": hi,
        "interval_status": "conditional_exploratory" if pd.notna(lo) else "unavailable",
        "leave_one_event_out_roi_min": loo.min(),
        "leave_one_event_out_roi_max": loo.max(),
        "without_largest_stake_event_roi": removed(largest_stake),
        "without_largest_positive_profit_event_roi": removed(largest_win),
        "equal_wager_mean_roi": (eligible.reported_profit_usd / eligible.reported_stake_usd).mean(),
        "largest_event_stake_share": groups.stake.max() / gs if pd.notna(gs) and gs > 0 else np.nan,
        "raw_pvalue": p,
        "test_status": "blocked:" + ",".join(reasons)
        if reasons
        else "approximate_conditional_exploratory",
        "inference_assumption": "namespace_event_proxies_independent_unverified",
        "currency": "USD",
        "fee_inclusion": "unverified",
        "promotion_allowed": False,
    }
