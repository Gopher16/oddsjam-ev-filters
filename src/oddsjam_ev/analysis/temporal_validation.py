"""
temporal_validation.py
===============================================================================

Purpose
-------
Construct auditable chronological research splits without sharing opportunity
identifiers across the earlier and later evaluation populations.

What this module provides
-------------------------
1. TemporalValidationConfig
   Configures execution timestamp, grouping key, and split fraction.

2. build_group_aware_temporal_split(...)
   Returns earlier/later execution populations and a reconciliation audit.

Design Notes
------------
- The split boundary is based on execution timestamps, not outcome timestamps.
- Groups spanning the boundary are quarantined from both populations.
- Missing timestamps are excluded and counted explicitly.
- A clean split is not proof of prospective validity: data selection, labeling,
  overlapping saved filters, and model development can still cause leakage.
- This helper does not compute p-values or authorize production changes.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalValidationConfig:
    """Configuration for chronological opportunity-group separation.

    Attributes
    ----------
    timestamp_column
        Recorded execution timestamp; must represent when a bet was placed.
    group_column
        Research opportunity grouping key, including surrogate keys.
    earlier_fraction
        Target fraction of valid executions used to select the cutoff.
    """

    timestamp_column: str = "created_at_et"
    group_column: str = "research_opportunity_key"
    earlier_fraction: float = 0.5


def build_group_aware_temporal_split(
    executions: pd.DataFrame,
    cfg: TemporalValidationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create earlier/later populations and an explicit exclusions audit.

    Parameters
    ----------
    executions
        Execution-level dataset containing the configured columns.
    cfg
        Optional timestamp, grouping, and cutoff configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Earlier executions, later executions, and a one-row audit containing
        cutoff, included/excluded counts, and group overlap checks.

    Raises
    ------
    KeyError
        If a required column is absent.
    ValueError
        If the fraction is invalid, fewer than two valid timestamps exist,
        or a meaningful two-sided split cannot be constructed.

    Notes
    -----
    A group that has any valid executions on both sides of the boundary is
    quarantined in full. Rows with missing timestamps are also excluded, even
    when other rows in their group have valid timestamps.
    """
    cfg = cfg or TemporalValidationConfig()
    if not 0 < cfg.earlier_fraction < 1:
        raise ValueError("earlier_fraction must be strictly between zero and one")
    missing = [c for c in (cfg.timestamp_column, cfg.group_column) if c not in executions]
    if missing:
        raise KeyError(f"Missing temporal validation columns: {missing}")
    if executions[cfg.group_column].isna().any():
        raise ValueError("Group keys must be non-null; construct research grains first")

    timestamps = pd.to_datetime(executions[cfg.timestamp_column], errors="coerce", utc=True)
    valid = timestamps.notna()
    ordered = timestamps.loc[valid].sort_values()
    if len(ordered) < 2 or ordered.nunique() < 2:
        raise ValueError("At least two distinct valid execution timestamps are required")
    cutoff_index = min(int(len(ordered) * cfg.earlier_fraction), len(ordered) - 1)
    cutoff = ordered.iloc[cutoff_index]
    earlier_mask = valid & timestamps.lt(cutoff)
    later_mask = valid & timestamps.ge(cutoff)
    if not earlier_mask.any() or not later_mask.any():
        raise ValueError("Timestamp distribution does not permit a two-sided split")

    early_keys = set(executions.loc[earlier_mask, cfg.group_column])
    late_keys = set(executions.loc[later_mask, cfg.group_column])
    spanning = early_keys & late_keys
    quarantined = executions[cfg.group_column].isin(spanning)
    earlier = executions.loc[earlier_mask & ~quarantined].copy()
    later = executions.loc[later_mask & ~quarantined].copy()
    if earlier.empty or later.empty:
        raise ValueError("Quarantining spanning groups eliminated one split population")

    overlap = set(earlier[cfg.group_column]) & set(later[cfg.group_column])
    assert not overlap
    audit = pd.DataFrame(
        [
            {
                "cutoff_utc": cutoff,
                "total_executions": len(executions),
                "earlier_executions": len(earlier),
                "later_executions": len(later),
                "missing_timestamp_executions": int((~valid).sum()),
                "quarantined_spanning_executions": int((valid & quarantined).sum()),
                "quarantined_spanning_groups": len(spanning),
                "shared_opportunity_groups": len(overlap),
                "reconciles": len(earlier)
                + len(later)
                + int((~valid).sum())
                + int((valid & quarantined).sum())
                == len(executions),
            }
        ]
    )
    assert bool(audit.loc[0, "reconciles"])
    return earlier, later, audit
