# src/oddsjam_ev/qa.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FieldValidationResult:
    n_rows: int
    missing_rate: dict[str, float]
    impossible_counts: dict[str, int]


def validate_core_fields(
    df: pd.DataFrame,
    *,
    stake_col: str = "stake",
    odds_col: str = "odds",
    profit_col: str = "bet_profit",
) -> FieldValidationResult:
    """
    Validate core numeric fields for missingness and obvious impossible values.

    Rules (lightweight, non-opinionated):
      - stake <= 0 is impossible for a placed bet
      - odds == 0 is invalid; NaN allowed
      - profit can be NaN (unsettled) but should not be inf
    """
    out_missing: dict[str, float] = {}
    out_impossible: dict[str, int] = {}

    n = int(len(df))

    for col in [stake_col, odds_col, profit_col]:
        if col in df.columns:
            out_missing[col] = float(df[col].isna().mean())
        else:
            out_missing[col] = 1.0

    if stake_col in df.columns:
        stake = pd.to_numeric(df[stake_col], errors="coerce")
        out_impossible[f"{stake_col}<=0"] = int((stake <= 0).sum())

    if odds_col in df.columns:
        odds = pd.to_numeric(df[odds_col], errors="coerce")
        out_impossible[f"{odds_col}==0"] = int((odds == 0).sum())

    if profit_col in df.columns:
        prof = pd.to_numeric(df[profit_col], errors="coerce")
        out_impossible[f"{profit_col} is inf"] = int(np.isinf(prof.to_numpy()).sum())

    return FieldValidationResult(
        n_rows=n,
        missing_rate=out_missing,
        impossible_counts=out_impossible,
    )


def duplicate_audit(
    df: pd.DataFrame,
    *,
    keys: list[list[str]],
) -> pd.DataFrame:
    """
    Report duplicate rates under multiple identity key definitions.

    Parameters
    ----------
    keys:
      list of key column lists (each entry defines one identity scheme)

    Returns
    -------
    pd.DataFrame with columns:
      - key_name
      - n_rows
      - n_dupe_rows
      - dupe_row_rate
      - n_unique
    """
    rows = []
    n = int(len(df))

    for cols in keys:
        present = [c for c in cols if c in df.columns]
        key_name = " | ".join(cols)

        if len(present) != len(cols):
            rows.append(
                {
                    "key_name": key_name,
                    "n_rows": n,
                    "n_dupe_rows": np.nan,
                    "dupe_row_rate": np.nan,
                    "n_unique": np.nan,
                    "note": f"missing cols: {sorted(set(cols) - set(present))}",
                }
            )
            continue

        dupe_mask = df.duplicated(subset=cols, keep=False)
        n_dupe = int(dupe_mask.sum())
        n_unique = int(df[cols].drop_duplicates().shape[0])
        rows.append(
            {
                "key_name": key_name,
                "n_rows": n,
                "n_dupe_rows": n_dupe,
                "dupe_row_rate": float(n_dupe / n) if n > 0 else 0.0,
                "n_unique": n_unique,
                "note": "",
            }
        )

    return pd.DataFrame(rows).sort_values("dupe_row_rate", ascending=False, na_position="last")
