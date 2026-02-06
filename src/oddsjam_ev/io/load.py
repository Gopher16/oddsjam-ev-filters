# src/oddsjam_ev/io/load.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

NUMERIC_COLS = [
    "odds",
    "stake",
    "payout",
    "profit",
    "line",
    "clv",
    "hold_percentage",
    "implied_probability",
    "true_probability",
    "ev_percentage",
    "liquidity",
    "market_width",
]


def load_bet_tracker_csv(path: str | Path) -> pd.DataFrame:
    """
    Load an OddsJam Bet Tracker export CSV with basic normalization.

    - Parses timestamps
    - Coerces numeric columns
    - Leaves naming untouched (schema-first approach)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Timestamps
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
