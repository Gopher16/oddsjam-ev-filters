"""
research_grains.py
===============================================================================

Purpose
-------
Build explicit execution-level and opportunity-group views of processed
OddsJam Bet Tracker data for downstream betting-filter research.

What this module provides
-------------------------
1. ResearchGrainsConfig
   Column mappings used to construct the two research views.

2. build_research_grains(...)
   Returns:
     - execution-level rows, preserving every processed bet
     - opportunity-group audit rows, summarizing executions sharing a source
       opportunity identifier

Design Notes
------------
- Execution rows are the authoritative grain for stake, realized profit, and
  execution-level ROI calculations.
- A source opportunity identifier is a grouping proxy, not proof that the
  associated executions represent one unique market quote.
- Missing or blank opportunity identifiers receive distinct, dataset-local
  surrogate keys so unrelated executions are not combined.
- Opportunity-level stake and profit sums are reconciliation diagnostics only;
  opportunity-group rows must not be treated as independent bets.
- Grouping keys are not guaranteed to remain stable across dataset exports.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchGrainsConfig:
    """
    Configuration for execution and opportunity-group construction.

    Attributes
    ----------
    opportunity_column
        Source identifier used to group executions into opportunity audits.
    timestamp_column
        Execution timestamp used for first and last activity in each group.
    profit_column
        Realized execution-profit column used for reconciliation totals.
    stake_column
        Execution stake column used for reconciliation totals.
    """

    opportunity_column: str = "opportunity_id"
    timestamp_column: str = "created_at_et"
    profit_column: str = "bet_profit"
    stake_column: str = "stake"


def build_research_grains(
    processed: pd.DataFrame,
    cfg: ResearchGrainsConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build execution-level records and a descriptive opportunity-group audit.

    Parameters
    ----------
    processed
        Processed Bet Tracker dataframe containing the configured identifier,
        timestamp, profit, and stake columns.
    cfg
        Optional column-mapping configuration. Defaults to
        ResearchGrainsConfig().

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Execution rows with dataset-local row IDs, missing-identifier flags,
        and grouping keys; and opportunity-group audit rows with execution
        counts, activity bounds, and reconciliation totals.

    Raises
    ------
    ValueError
        If any configured required column is absent.
    AssertionError
        If the grouped execution counts do not reconcile to the input rows.

    Notes
    -----
    - All input executions are retained; the input dataframe is not modified.
    - Blank or missing source opportunity IDs receive distinct surrogate keys.
    - Source IDs are prefixed to avoid collisions with surrogate keys.
    - Opportunity-group profit and stake sums are audit quantities, not a
      replacement for execution-level performance metrics.
    """
    cfg = cfg or ResearchGrainsConfig()
    required = [cfg.opportunity_column, cfg.timestamp_column, cfg.profit_column, cfg.stake_column]
    missing = [col for col in required if col not in processed.columns]
    if missing:
        raise ValueError(f"Required research-grain columns missing: {missing}")

    executions = processed.copy().reset_index(drop=True)
    executions.insert(0, "execution_row_id", range(len(executions)))
    raw_id = executions[cfg.opportunity_column].astype("string")
    valid_id = raw_id.notna() & raw_id.str.strip().fillna("").ne("")
    executions["opportunity_id_missing"] = ~valid_id
    executions["research_opportunity_key"] = raw_id.where(
        valid_id, "__missing_opportunity_row_" + executions["execution_row_id"].astype(str)
    ).astype(str)
    # Prefix source IDs to keep them separate from dataset-local surrogate keys.
    executions.loc[valid_id, "research_opportunity_key"] = (
        "source:" + raw_id.loc[valid_id].astype(str)
    )
    executions.loc[~valid_id, "research_opportunity_key"] = (
        "surrogate:" + executions.loc[~valid_id, "execution_row_id"].astype(str)
    )

    grouped = executions.groupby("research_opportunity_key", sort=False, dropna=False)
    opportunities = grouped.agg(
        execution_count=("execution_row_id", "size"),
        first_execution_at=(cfg.timestamp_column, "min"),
        last_execution_at=(cfg.timestamp_column, "max"),
        summed_execution_stake=(cfg.stake_column, "sum"),
        summed_execution_profit=(cfg.profit_column, "sum"),
        missing_source_id=("opportunity_id_missing", "max"),
    ).reset_index()
    opportunities["repeated_execution_group"] = opportunities["execution_count"].gt(1)
    opportunities["research_grain"] = "opportunity_group_audit_not_independent_observation"

    # Reconcile execution counts without interpreting groups as independent bets.
    if int(opportunities["execution_count"].sum()) != len(executions):
        raise AssertionError("Execution-to-opportunity reconciliation failed")
    return executions, opportunities
