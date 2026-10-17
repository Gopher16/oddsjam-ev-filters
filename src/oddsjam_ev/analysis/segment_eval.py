"""
segment_eval.py
===============================================================================

Purpose
-------
Provide reusable, descriptive segment summaries for historical OddsJam
Bet Tracker executions without changing the canonical filter evaluation.

What this module provides
-------------------------
1. SegmentEvalConfig
   Configures source columns, minimum sample size, and grouping behavior.

2. build_segment_evaluation(...)
   Summarizes counts, stake, realized profit, realized ROI, optional modeled
   EV, and opportunity-group coverage for a categorical or binned dimension.

Design Notes
------------
- All statistics are execution-level; repeated opportunity groups may induce
  dependence and are not automatically deduplicated.
- Realized ROI is total profit divided by total stake, not mean bet ROI.
- EV totals are included only when an explicit per-execution EV amount column
  is supplied. A percentage EV field must not be summed as dollars.
- Sample-size thresholds flag exploratory results; they do not establish
  statistical significance or out-of-sample reliability.
- Bin edges are supplied by the caller, not inferred from historical returns.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentEvalConfig:
    """
    Configuration for execution-level segment summaries.

    Attributes
    ----------
    stake_column
        Executed stake amount.
    profit_column
        Realized execution profit amount.
    opportunity_key_column
        Optional grouping key for descriptive distinct-group counts.
    ev_amount_column
        Optional modeled EV in currency units per execution.
    min_executions
        Minimum count required to mark a segment as sufficiently sampled.
    include_missing
        If True, retain missing dimension values as an explicit segment.
    """

    stake_column: str = "stake"
    profit_column: str = "bet_profit"
    opportunity_key_column: str | None = "research_opportunity_key"
    ev_amount_column: str | None = None
    min_executions: int = 100
    include_missing: bool = True


def build_segment_evaluation(
    executions: pd.DataFrame,
    dimension: str,
    cfg: SegmentEvalConfig | None = None,
    *,
    bin_edges: Sequence[float] | None = None,
) -> pd.DataFrame:
    """
    Calculate descriptive performance by one execution-level dimension.

    Parameters
    ----------
    executions
        Execution-level dataset; each input row represents one recorded bet.
    dimension
        Source column used for categorical grouping or numeric binning.
    cfg
        Optional column mapping and sample-size configuration.
    bin_edges
        Optional strictly increasing numeric bin boundaries. Uses left-closed,
        right-open intervals; the final upper edge is exclusive.

    Returns
    -------
    pd.DataFrame
        Segment-level count, stake, profit, stake-weighted realized ROI,
        optional modeled EV metrics, opportunity-group count, and a sample
        sufficiency flag. ROI is a decimal fraction, not a percentage.

    Raises
    ------
    ValueError
        If configuration or bin edges are invalid.
    KeyError
        If a required source column is absent.

    Notes
    -----
    Missing and out-of-range numeric values are grouped as ``<missing/outside>``
    when include_missing=True. Segments with zero total stake have undefined
    ROI. This function does not conduct hypothesis tests or rank filters.
    """
    cfg = cfg or SegmentEvalConfig()
    if cfg.min_executions < 1:
        raise ValueError("min_executions must be at least 1")
    required = [dimension, cfg.stake_column, cfg.profit_column]
    for optional in (cfg.opportunity_key_column, cfg.ev_amount_column):
        if optional is not None:
            required.append(optional)
    missing = sorted(set(required) - set(executions.columns))
    if missing:
        raise KeyError(f"Missing segment evaluation columns: {missing}")

    data = executions.copy()
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("bin_edges must be a strictly increasing sequence")
        labels = pd.cut(pd.to_numeric(data[dimension], errors="coerce"), bins=edges, right=False)
        data["_segment"] = labels.astype("string")
    else:
        data["_segment"] = data[dimension].astype("string")
    if cfg.include_missing:
        data["_segment"] = data["_segment"].fillna("<missing/outside>")
    else:
        data = data.loc[data["_segment"].notna()].copy()

    for col in [cfg.stake_column, cfg.profit_column, cfg.ev_amount_column]:
        if col is not None:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    group = data.groupby("_segment", dropna=False, sort=True)
    result = group.agg(
        execution_count=(cfg.stake_column, "size"),
        total_stake=(cfg.stake_column, lambda s: s.sum(min_count=1)),
        total_profit=(cfg.profit_column, lambda s: s.sum(min_count=1)),
        valid_stake_count=(cfg.stake_column, "count"),
        valid_profit_count=(cfg.profit_column, "count"),
    )
    if cfg.opportunity_key_column is not None:
        result["opportunity_group_count"] = group[cfg.opportunity_key_column].nunique(dropna=True)
    if cfg.ev_amount_column is not None:
        result["total_modeled_ev"] = group[cfg.ev_amount_column].sum(min_count=1)
        result["modeled_ev_roi"] = result["total_modeled_ev"].div(
            result["total_stake"].where(result["total_stake"] > 0)
        )
    result["realized_roi"] = result["total_profit"].div(
        result["total_stake"].where(result["total_stake"] > 0)
    )
    result["meets_min_executions"] = result["execution_count"].ge(cfg.min_executions)
    result["dimension"] = dimension
    return result.reset_index().rename(columns={"_segment": "segment"})
