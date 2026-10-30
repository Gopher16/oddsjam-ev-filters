# src/oddsjam_ev/io/load.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

NUMERIC_COLS = [
    "odds",
    "stake",
    "payout",
    "profit",
    "bet_profit",
    "line",
    "clv",
    "hold_percentage",
    "implied_probability",
    "true_probability",
    "ev_percentage",
    "percentage",
    "liquidity",
    "market_width",
]


def standardize_timestamp_to_et(
    df: pd.DataFrame,
    *,
    src_col: str = "created_at",
    out_col: str = "created_at_et",
    assume_tz_for_naive: str = "America/New_York",
) -> pd.DataFrame:
    """
    Standardize timestamps into ET (America/New_York).

    Behavior:
      - If src_col is tz-aware: convert to America/New_York
      - If src_col is tz-naive: localize using assume_tz_for_naive
        (this is an assumption; make it explicit in Notebook 01 caveats)

    Returns a COPY with out_col added.
    """
    if src_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {src_col}")

    out = df.copy()
    ts = pd.to_datetime(out[src_col], errors="coerce")

    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        out[out_col] = ts.dt.tz_convert("America/New_York")
    else:
        out[out_col] = ts.dt.tz_localize(
            assume_tz_for_naive,
            nonexistent="shift_forward",
            ambiguous="infer",
        ).dt.tz_convert("America/New_York")

    return out


def load_bet_tracker_csv(path: str | Path) -> pd.DataFrame:
    """
    Load an OddsJam Bet Tracker export CSV with basic normalization.

    - Parses timestamps (created_at)
    - Coerces numeric columns
    - Leaves naming untouched (schema-first approach)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # Keep EV's lexical precision before the legacy numeric preview is computed.
    df = pd.read_csv(path, low_memory=False, converters={"percentage": str})

    return normalize_bet_tracker_frame(df)


def normalize_bet_tracker_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply legacy numeric/date coercion to a copy of source records.

    Parameters
    ----------
    df : pandas.DataFrame
        Source records; immutable ingestion observations remain separate.

    Returns
    -------
    pandas.DataFrame
        Typed records using the existing research preprocessing semantics.
    """
    df = df.copy()
    if "percentage" in df and "percentage_source_raw" not in df:
        df["percentage_source_raw"] = df["percentage"].copy()
    # Timestamp (raw parsing only; timezone standardization is a separate explicit step)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
