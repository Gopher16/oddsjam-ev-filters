"""
test_segment_eval.py
===============================================================================

Purpose
-------
Verify deterministic segment aggregation, numeric bins, and input safeguards.

What this module provides
-------------------------
1. test_categorical_segments_and_roi(...)
   Validates weighted ROI, group counts, and minimum-sample flags.
2. test_numeric_bins_and_missing(...)
   Validates explicit numeric bins and missing/outside observations.
3. test_invalid_inputs(...)
   Validates missing columns and invalid configuration.

Design Notes
------------
- Synthetic examples avoid local Parquet and network dependencies.
- Modeled EV amounts are never confused with percentage EV estimates.
===============================================================================
"""

import pandas as pd
import pytest

from oddsjam_ev.analysis.segment_eval import SegmentEvalConfig, build_segment_evaluation


def _executions():
    """Return a small deterministic execution-level dataset."""
    return pd.DataFrame({
        "sportsbook": ["A", "A", "B", None],
        "odds": [-110, 120, 150, None],
        "stake": [10.0, 30.0, 20.0, 40.0],
        "bet_profit": [5.0, -15.0, 10.0, 0.0],
        "ev_amount": [1.0, 3.0, 2.0, 4.0],
        "research_opportunity_key": ["x", "x", "y", "z"],
    })


def test_categorical_segments_and_roi():
    """Use stake-weighted ROI and preserve repeated-group diagnostics."""
    cfg = SegmentEvalConfig(min_executions=2, ev_amount_column="ev_amount")
    result = build_segment_evaluation(_executions(), "sportsbook", cfg).set_index("segment")
    assert result.loc["A", "execution_count"] == 2
    assert result.loc["A", "opportunity_group_count"] == 1
    assert result.loc["A", "realized_roi"] == pytest.approx(-0.25)
    assert result.loc["A", "modeled_ev_roi"] == pytest.approx(0.1)
    assert bool(result.loc["A", "meets_min_executions"])
    assert not bool(result.loc["B", "meets_min_executions"])
    assert result["execution_count"].sum() == 4


def test_numeric_bins_and_missing():
    """Retain out-of-range and null numeric values when requested."""
    result = build_segment_evaluation(
        _executions(), "odds", SegmentEvalConfig(min_executions=1), bin_edges=[-200, 0, 130]
    ).set_index("segment")
    assert result.loc["[-200.0, 0.0)", "execution_count"] == 1
    assert result.loc["[0.0, 130.0)", "execution_count"] == 1
    assert result.loc["<missing/outside>", "execution_count"] == 2


def test_invalid_inputs():
    """Fail clearly for absent columns and malformed boundaries."""
    with pytest.raises(KeyError, match="Missing segment evaluation columns"):
        build_segment_evaluation(_executions(), "unknown")
    with pytest.raises(ValueError, match="strictly increasing"):
        build_segment_evaluation(_executions(), "odds", bin_edges=[0, 0, 1])
    with pytest.raises(ValueError, match="min_executions"):
        build_segment_evaluation(_executions(), "sportsbook", SegmentEvalConfig(min_executions=0))
