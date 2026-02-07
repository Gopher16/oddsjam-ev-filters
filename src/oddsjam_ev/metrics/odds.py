from __future__ import annotations

import numpy as np
import pandas as pd


def american_to_prob(odds: float | int | None) -> float:
    """Convert American odds to implied probability."""
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)


def american_to_multiplier(odds: float | int | None) -> float:
    """
    Convert American odds to profit multiplier per $1 stake.
      +110 -> 1.10
      -110 -> 0.909...
    """
    if odds is None or pd.isna(odds):
        return np.nan
    o = float(odds)
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def ensure_prob_clv(
    df: pd.DataFrame, *, clv_col: str = "clv", out_col: str = "prob_clv"
) -> pd.DataFrame:
    """Ensure df[out_col] exists using American odds from df[clv_col] if needed."""
    if out_col in df.columns and df[out_col].notna().any():
        return df
    if clv_col not in df.columns:
        df[out_col] = np.nan
        return df
    df[out_col] = df[clv_col].apply(american_to_prob)
    return df


def ensure_odds_multiplier(
    df: pd.DataFrame, *, odds_col: str = "odds", out_col: str = "odds_multiplier"
) -> pd.DataFrame:
    """Ensure df[out_col] exists using American odds from df[odds_col] if needed."""
    if out_col in df.columns and df[out_col].notna().any():
        return df
    if odds_col not in df.columns:
        df[out_col] = np.nan
        return df
    df[out_col] = df[odds_col].apply(american_to_multiplier)
    return df


def clv_ev_pct(prob_clv: pd.Series, odds_multiplier: pd.Series) -> pd.Series:
    """
    EV% based on CLV implied prob and placed-odds multiplier:
      EV% = (p * mult - (1 - p)) * 100
    """
    p = pd.to_numeric(prob_clv, errors="coerce")
    m = pd.to_numeric(odds_multiplier, errors="coerce")
    return (p * m - (1.0 - p)) * 100.0


def compute_ev(
    stake: pd.Series,
    prob: pd.Series,
    odds_multiplier: pd.Series,
) -> pd.Series:
    """
    Compute expected value (EV) per bet.

    EV = p * (stake * multiplier) - (1 - p) * stake
    """
    return prob * (stake * odds_multiplier) - (1 - prob) * stake


def compute_ev_roi(ev: pd.Series, stake: pd.Series) -> pd.Series:
    """
    Expected ROI = EV / stake
    """
    return ev / stake
