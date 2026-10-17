"""
test_temporal_validation.py
===============================================================================

Purpose
-------
Verify group-aware chronological splitting and reconciliation safeguards.

What this module provides
-------------------------
1. test_spanning_groups_are_quarantined(...)
   Checks disjoint groups, exclusion counts, and chronology.
2. test_missing_timestamps_are_audited(...)
   Checks explicit accounting for unparseable timestamps.
3. test_invalid_inputs(...)
   Checks failure modes for invalid configuration and missing fields.

Design Notes
------------
- Synthetic fixtures keep tests independent of private Parquet snapshots.
===============================================================================
"""

import pandas as pd
import pytest

from oddsjam_ev.analysis.temporal_validation import (
    TemporalValidationConfig,
    build_group_aware_temporal_split,
)


def test_spanning_groups_are_quarantined():
    """Keep opportunity identifiers disjoint across chronological populations."""
    data = pd.DataFrame(
        {
            "created_at_et": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
                "2026-01-06",
            ],
            "research_opportunity_key": ["a", "cross", "b", "cross", "c", "d"],
        }
    )
    earlier, later, audit = build_group_aware_temporal_split(data)
    assert len(earlier) == 2
    assert len(later) == 2
    assert audit.loc[0, "quarantined_spanning_executions"] == 2
    assert audit.loc[0, "quarantined_spanning_groups"] == 1
    assert audit.loc[0, "shared_opportunity_groups"] == 0
    assert bool(audit.loc[0, "reconciles"])
    assert set(earlier.research_opportunity_key).isdisjoint(later.research_opportunity_key)


def test_missing_timestamps_are_audited():
    """Exclude and count missing timestamps without losing reconciliation."""
    data = pd.DataFrame(
        {
            "created_at_et": ["2026-01-01", "2026-01-02", None, "2026-01-03"],
            "research_opportunity_key": ["a", "b", "missing", "c"],
        }
    )
    earlier, later, audit = build_group_aware_temporal_split(data)
    assert len(earlier) + len(later) == 3
    assert audit.loc[0, "missing_timestamp_executions"] == 1
    assert bool(audit.loc[0, "reconciles"])


def test_invalid_inputs():
    """Reject missing grouping keys, missing columns, and invalid cutoffs."""
    data = pd.DataFrame(
        {"created_at_et": ["2026-01-01", "2026-01-02"], "research_opportunity_key": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="earlier_fraction"):
        build_group_aware_temporal_split(data, TemporalValidationConfig(earlier_fraction=1))
    with pytest.raises(KeyError, match="Missing temporal"):
        build_group_aware_temporal_split(data.drop(columns="created_at_et"))
    with pytest.raises(ValueError, match="non-null"):
        build_group_aware_temporal_split(data.assign(research_opportunity_key=[None, "b"]))
