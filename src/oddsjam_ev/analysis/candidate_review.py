"""
candidate_review.py
===============================================================================

Purpose
-------
Build conservative, descriptive evidence-review tables from canonical
betting-filter evaluation results without prescribing live filter changes.

What this module provides
-------------------------
1. CandidateReviewConfig
   Configurable evidence-coverage thresholds and source-column mappings.

2. build_candidate_review(...)
   Preserves canonical filter metrics and adds evidence flags, limitations,
   and next validation steps for each regime-filter combination.

Design Notes
------------
- Historical ROI and modeled EV are not proof of future profitability.
- Missing uncertainty estimates, liquidity coverage, and execution-quality
  evidence are identified explicitly rather than imputed.
- Sample thresholds are review heuristics, not significance tests.
- Filter labels can overlap and regime labels need not match filter names.
- No output constitutes an automatic production recommendation.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateReviewConfig:
    """
    Configuration for conservative filter evidence review.

    Attributes
    ----------
    filter_column
        Saved-filter label in the canonical evaluation table.
    regime_column
        Execution regime associated with the evaluated filter population.
    count_column
        Number of evaluated executions.
    roi_column
        Stake-weighted realized ROI.
    ci_low_column
        Lower endpoint of the available ROI uncertainty interval.
    ci_high_column
        Upper endpoint of the available ROI uncertainty interval.
    liquidity_coverage_column
        Fraction of records with available liquidity diagnostics.
    min_executions
        Descriptive minimum count for prioritizing further review.
    min_liquidity_coverage
        Minimum diagnostic coverage for capacity-sensitive conclusions.
    """

    filter_column: str = "saved_filter_names"
    regime_column: str = "regime"
    count_column: str = "bet_count"
    roi_column: str = "total_actual_roi"
    ci_low_column: str = "roi_ci_low"
    ci_high_column: str = "roi_ci_high"
    liquidity_coverage_column: str = "liquidity_coverage"
    min_executions: int = 100
    min_liquidity_coverage: float = 0.8


def build_candidate_review(
    evaluation: pd.DataFrame,
    capacity: pd.DataFrame | None = None,
    cfg: CandidateReviewConfig | None = None,
) -> pd.DataFrame:
    """
    Summarize historical filter evidence and outstanding validation needs.

    Parameters
    ----------
    evaluation
        Canonical evaluation table, preferably regime-specific, containing
        filter names, execution counts, and realized ROI.
    capacity
        Optional capacity table keyed by saved-filter name. Its liquidity
        coverage is used only for exchange-regime rows.
    cfg
        Optional column mappings and descriptive evidence thresholds.

    Returns
    -------
    pd.DataFrame
        One row per supplied evaluation row with evidence flags, review
        classification, and required follow-up. Input metrics are retained.

    Raises
    ------
    KeyError
        If required evaluation or supplied capacity join columns are absent.
    ValueError
        If evidence thresholds are outside supported ranges or a capacity
        join would duplicate evaluation rows.

    Notes
    -----
    - The report is descriptive and does not identify causal filter effects.
    - Confidence intervals are consumed as provided, not recalculated; their
      validity and dependence assumptions must be reviewed separately.
    - Missing liquidity coverage is not interpreted as zero liquidity.
    - The report never returns a deploy/enable instruction.
    """
    cfg = cfg or CandidateReviewConfig()
    if cfg.min_executions < 1 or not 0 <= cfg.min_liquidity_coverage <= 1:
        raise ValueError("Evidence thresholds must be positive and coverage between 0 and 1.")
    required = [cfg.filter_column, cfg.regime_column, cfg.count_column, cfg.roi_column]
    missing = [name for name in required if name not in evaluation.columns]
    if missing:
        raise KeyError(f"Missing candidate-review evaluation columns: {missing}")

    result = evaluation.copy().reset_index(drop=True)
    if capacity is not None:
        if cfg.filter_column not in capacity.columns:
            raise KeyError(f"Missing capacity join column: {cfg.filter_column}")
        if capacity[cfg.filter_column].duplicated().any():
            raise ValueError("Capacity contains duplicate filter keys; join would multiply rows.")
        if cfg.liquidity_coverage_column in capacity.columns:
            result = result.merge(
                capacity[[cfg.filter_column, cfg.liquidity_coverage_column]],
                on=cfg.filter_column,
                how="left",
                suffixes=("", "_capacity"),
                validate="many_to_one",
            )
            alt = f"{cfg.liquidity_coverage_column}_capacity"
            if alt in result.columns:
                result[cfg.liquidity_coverage_column] = result[alt].combine_first(
                    result[cfg.liquidity_coverage_column]
                )
                result = result.drop(columns=[alt])

    counts = pd.to_numeric(result[cfg.count_column], errors="coerce")
    roi = pd.to_numeric(result[cfg.roi_column], errors="coerce")
    low = (
        pd.to_numeric(result[cfg.ci_low_column], errors="coerce")
        if cfg.ci_low_column in result
        else pd.Series(np.nan, index=result.index)
    )
    high = (
        pd.to_numeric(result[cfg.ci_high_column], errors="coerce")
        if cfg.ci_high_column in result
        else pd.Series(np.nan, index=result.index)
    )
    coverage = (
        pd.to_numeric(result[cfg.liquidity_coverage_column], errors="coerce")
        if cfg.liquidity_coverage_column in result
        else pd.Series(np.nan, index=result.index)
    )
    result["meets_review_count"] = counts.ge(cfg.min_executions)
    result["roi_interval_available"] = low.notna() & high.notna() & low.le(high)
    result["roi_interval_excludes_zero"] = result["roi_interval_available"] & (
        low.gt(0) | high.lt(0)
    )
    result["roi_interval_positive"] = result["roi_interval_available"] & low.gt(0)
    exchange = result[cfg.regime_column].astype("string").str.lower().eq("exchange")
    result["exchange_liquidity_adequately_observed"] = (~exchange) | coverage.ge(
        cfg.min_liquidity_coverage
    )

    classifications = []
    followups = []
    for idx in result.index:
        reasons = []
        if not bool(result.at[idx, "meets_review_count"]):
            reasons.append("Increase execution sample and distinct opportunity coverage")
        if not bool(result.at[idx, "roi_interval_available"]):
            reasons.append("Establish valid group-aware uncertainty estimates")
        if bool(exchange.at[idx]) and not bool(
            result.at[idx, "exchange_liquidity_adequately_observed"]
        ):
            reasons.append("Audit liquidity missingness and realistic executable capacity")
        if not np.isfinite(roi.at[idx]):
            reasons.append("Reconcile realized ROI and settlement inputs")
        reasons.append("Validate on a later non-overlapping period with execution and CLV checks")
        if (
            bool(result.at[idx, "meets_review_count"])
            and bool(result.at[idx, "roi_interval_positive"])
            and np.isfinite(roi.at[idx])
            and roi.at[idx] > 0
        ):
            classification = "Further validation candidate"
        elif np.isfinite(roi.at[idx]) and roi.at[idx] <= 0:
            classification = "Historical result non-positive"
        else:
            classification = "Insufficient evidence for prioritization"
        classifications.append(classification)
        followups.append("; ".join(reasons))
    result["review_classification"] = classifications
    result["required_follow_up"] = followups
    result["production_change_authorized"] = False
    return result
