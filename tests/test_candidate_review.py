"""
test_candidate_review.py
===============================================================================

Purpose
-------
Test conservative evidence classifications and safe capacity reconciliation.

What this module provides
-------------------------
1. test_candidate_flags_and_no_deployment(...)
   Verifies uncertainty, count, and liquidity evidence flags.

2. test_missing_intervals_and_capacity_guardrails(...)
   Verifies conservative missing-data behavior and join validation.

3. test_invalid_inputs(...)
   Verifies clear errors for unsupported inputs and thresholds.

Design Notes
------------
- Synthetic fixtures avoid external data and Jupyter dependencies.
- Tests do not assume descriptive evidence proves an actionable edge.
===============================================================================
"""

import pandas as pd
import pytest

from oddsjam_ev.analysis.candidate_review import CandidateReviewConfig, build_candidate_review


def _evaluation():
    """Return deterministic exchange and sportsbook filter summaries."""
    return pd.DataFrame(
        {
            "saved_filter_names": ["A", "B", "C"],
            "regime": ["exchange", "sportsbook", "exchange"],
            "bet_count": [150, 50, 180],
            "total_actual_roi": [0.08, 0.10, -0.03],
            "roi_ci_low": [0.01, -0.04, -0.08],
            "roi_ci_high": [0.15, 0.20, 0.02],
        }
    )


def test_candidate_flags_and_no_deployment():
    """Flag uncertainty and capacity without authorizing production changes."""
    capacity = pd.DataFrame({"saved_filter_names": ["A", "C"], "liquidity_coverage": [0.4, 0.9]})
    result = build_candidate_review(_evaluation(), capacity).set_index("saved_filter_names")
    assert result.loc["A", "review_classification"] == "Further validation candidate"
    assert not bool(result.loc["A", "exchange_liquidity_adequately_observed"])
    assert not bool(result.loc["B", "meets_review_count"])
    assert result.loc["C", "review_classification"] == "Historical result non-positive"
    assert not result["production_change_authorized"].any()
    assert len(result) == 3


def test_missing_intervals_and_capacity_guardrails():
    """Treat absent uncertainty as missing and reject multiplying joins."""
    evaluation = _evaluation().drop(columns=["roi_ci_low", "roi_ci_high"])
    result = build_candidate_review(evaluation)
    assert not result["roi_interval_available"].any()
    assert result.loc[0, "review_classification"] == "Insufficient evidence for prioritization"
    repeated = pd.DataFrame({"saved_filter_names": ["A", "A"], "liquidity_coverage": [0.2, 0.3]})
    with pytest.raises(ValueError, match="duplicate filter keys"):
        build_candidate_review(_evaluation(), repeated)


def test_invalid_inputs():
    """Reject missing source fields and unsupported evidence thresholds."""
    with pytest.raises(KeyError, match="Missing candidate-review"):
        build_candidate_review(_evaluation().drop(columns=["regime"]))
    with pytest.raises(ValueError, match="Evidence thresholds"):
        build_candidate_review(_evaluation(), cfg=CandidateReviewConfig(min_executions=0))
