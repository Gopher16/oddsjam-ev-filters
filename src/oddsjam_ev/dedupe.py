# src/oddsjam_ev/dedupe.py
from __future__ import annotations

import pandas as pd


def present_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def dedupe_artifacts(
    df: pd.DataFrame,
    *,
    identity_cols: list[str],
    time_col: str = "created_at_et",
    stake_col: str = "stake",
    filter_col: str = "saved_filter_names",
    strategy: str = "earliest",
    scope: str = "within_filter",
) -> pd.DataFrame:
    """
    Deduplicate likely export artifacts by grouping on identity_cols (+ optionally filter_col),
    then keeping one row per group.

    OFF by default in the pipeline. Use mainly for modeling / opportunity-level datasets,
    not for early ROI truth.
    """
    out = df.copy()

    id_cols = present_cols(out, identity_cols)
    if not id_cols:
        raise ValueError("No identity_cols were found in df; cannot dedupe.")

    group_cols = id_cols + (
        [filter_col] if scope == "within_filter" and filter_col in out.columns else []
    )

    if time_col not in out.columns:
        raise KeyError(f"Missing time_col: {time_col}")
    if stake_col not in out.columns:
        raise KeyError(f"Missing stake_col: {stake_col}")

    out[stake_col] = pd.to_numeric(out[stake_col], errors="coerce")
    out["_t"] = pd.to_datetime(out[time_col], errors="coerce")

    # Sort so "first" is the keeper depending on strategy
    if strategy == "earliest":
        out = out.sort_values(group_cols + ["_t"], ascending=[True] * len(group_cols) + [True])
    elif strategy == "latest":
        out = out.sort_values(group_cols + ["_t"], ascending=[True] * len(group_cols) + [False])
    elif strategy == "max_stake":
        out = out.sort_values(
            group_cols + [stake_col, "_t"],
            ascending=[True] * len(group_cols) + [False, True],
        )
    else:
        raise ValueError("strategy must be one of: earliest, latest, max_stake")

    deduped = out.drop_duplicates(subset=group_cols, keep="first").drop(columns=["_t"])
    return deduped
