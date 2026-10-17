"""
test_research_grains.py
===============================================================================

Purpose
-------
Validate execution preservation, opportunity grouping, and required-column
checks in the research-grain builder.

What this module provides
-------------------------
1. test_grains_preserve_executions_and_reconcile(...)
   Confirms repeated and missing opportunity IDs are handled without dropping
   executions or changing the source dataframe.

2. test_missing_required_column_fails_loudly(...)
   Confirms missing required research columns produce an actionable error.

Design Notes
------------
- Tests use small deterministic inputs and do not depend on local Parquet files.
- Grouped profit totals are checked only as reconciliation diagnostics.
===============================================================================
"""

import pandas as pd
import pytest

from oddsjam_ev.analysis.research_grains import build_research_grains


def test_grains_preserve_executions_and_reconcile():
    """Preserve every execution and reconcile source and surrogate groups."""
    data = pd.DataFrame(
        {
            "opportunity_id": ["abc", "abc", None, ""],
            "created_at_et": pd.to_datetime(["2026-01-01"] * 4, utc=True),
            "stake": [10.0, 20.0, 30.0, 40.0],
            "bet_profit": [5.0, -20.0, 2.0, 0.0],
        }
    )
    executions, opportunities = build_research_grains(data)
    assert len(executions) == 4
    assert len(opportunities) == 3
    assert sorted(opportunities.execution_count.tolist()) == [1, 1, 2]
    assert executions.opportunity_id_missing.sum() == 2
    assert opportunities.execution_count.sum() == len(data)
    assert opportunities.summed_execution_profit.sum() == pytest.approx(data.bet_profit.sum())
    assert data.columns.tolist() == ["opportunity_id", "created_at_et", "stake", "bet_profit"]


def test_missing_required_column_fails_loudly():
    """Reject inputs missing columns required for research-grain construction."""
    with pytest.raises(ValueError, match="Required research-grain columns missing"):
        build_research_grains(pd.DataFrame({"stake": [1]}))
