# src/oddsjam_ev/metrics/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

DedupMethod = Literal["earliest", "latest", "largest_stake"]
DedupScope = Literal["within_filter", "global"]


@dataclass(frozen=True)
class DedupeReport:
    rows_in: int
    rows_out: int
    rows_dropped: int
    dedupe_key: list[str]
    method: DedupMethod
    scope: DedupScope


def standardize_timestamp(
    df: pd.DataFrame,
    *,
    src_col: str = "created_at",
    out_col: str = "created_at_est",
    target_tz: str = "America/New_York",
    assume_tz_for_naive: str | None = "America/New_York",
) -> pd.DataFrame:
    """
    Create a timezone-aware timestamp column in the target timezone.

    IMPORTANT (updated for your OddsJam exports):
    - Your OddsJam raw CSV timestamps are already standardized to EST/ET.
      That means tz-naive timestamps should be treated as America/New_York,
      not UTC.

    Behavior:
    - If src_col is tz-naive:
        - localize using assume_tz_for_naive (default America/New_York)
    - If src_col is tz-aware:
        - convert to target_tz
    """
    out = df.copy()

    if src_col not in out.columns:
        raise KeyError(f"Missing timestamp column: {src_col}")

    ts = pd.to_datetime(out[src_col], errors="coerce")

    if not pd.api.types.is_datetime64tz_dtype(ts):
        if assume_tz_for_naive is None:
            # fallback: preserve old "safe default" behavior if caller explicitly sets None
            ts = ts.dt.tz_localize("UTC").dt.tz_convert(target_tz)
        else:
            ts = ts.dt.tz_localize(assume_tz_for_naive).dt.tz_convert(target_tz)
    else:
        ts = ts.dt.tz_convert(target_tz)

    out[out_col] = ts
    return out


def _pick_first_present(cols: list[str], df_cols: pd.Index) -> str | None:
    for c in cols:
        if c in df_cols:
            return c
    return None


def infer_entity_cols(df: pd.DataFrame) -> list[str]:
    """
    Best-effort inference for a stable "bet identity" when no unique id exists.

    We purposely exclude columns that vary across re-exports or timestamps
    (created_at/profit/payout), since the whole point is deduping across time.
    """
    candidates = [
        "event_id",
        "game_id",
        "match_id",
        "fixture_id",
        "market_id",
        "selection_id",
        "sport",
        "league",
        "event",
        "matchup",
        "team",
        "player",
        "market",
        "market_name",
        "bet_type",
        "prop_type",
        "side",
        "selection",
        "name",
        "outcome",
        "odds",
        "line",
        "sportsbook",
        "book",
        "platform",
    ]

    present = [c for c in candidates if c in df.columns]
    if len(present) < 4:
        return present
    return present


def dedupe_bets(
    df: pd.DataFrame,
    *,
    method: DedupMethod = "earliest",
    scope: DedupScope = "within_filter",
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
    stake_col: str = "stake",
    id_cols: list[str] | None = None,
    entity_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, DedupeReport]:
    """
    Deduplicate bets that appear multiple times across timestamps / exports.
    """
    out = df.copy()

    if time_col not in out.columns:
        raise KeyError(f"Missing time column '{time_col}'. Run standardize_timestamp() first.")

    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    if out[time_col].isna().all():
        raise ValueError(f"Column '{time_col}' could not be parsed to datetimes (all NaT).")

    if id_cols is None:
        id_cols = ["bet_id", "wager_id", "ticket_id", "id", "uuid"]

    id_col = _pick_first_present(id_cols, out.columns)

    key: list[str] = []
    if scope == "within_filter":
        if filter_col not in out.columns:
            raise KeyError(
                f"scope='within_filter' requires '{filter_col}' column, but it is missing."
            )
        key.append(filter_col)

    if id_col is not None:
        key.append(id_col)
    else:
        if entity_cols is None:
            entity_cols = infer_entity_cols(out)

        if not entity_cols:
            raise ValueError(
                "Unable to infer entity_cols for dedupe (no id column and too few stable columns). "
                "Pass entity_cols explicitly."
            )

        entity_cols = [c for c in entity_cols if c in out.columns]
        if len(entity_cols) < 4:
            raise ValueError(
                f"entity_cols too small ({len(entity_cols)}). Pass a better entity_cols list."
            )
        key.extend(entity_cols)

    if method in {"earliest", "latest"}:
        asc = method == "earliest"
        out = out.sort_values(key + [time_col], ascending=[True] * len(key) + [asc])
        df_keep = out.drop_duplicates(subset=key, keep="first")
    elif method == "largest_stake":
        if stake_col not in out.columns:
            raise KeyError(f"method='largest_stake' requires '{stake_col}' column.")
        out[stake_col] = pd.to_numeric(out[stake_col], errors="coerce").fillna(-np.inf)
        out = out.sort_values(key + [stake_col], ascending=[True] * len(key) + [False])
        df_keep = out.drop_duplicates(subset=key, keep="first")
    else:
        raise ValueError(f"Unknown dedupe method: {method}")

    report = DedupeReport(
        rows_in=int(len(df)),
        rows_out=int(len(df_keep)),
        rows_dropped=int(len(df) - len(df_keep)),
        dedupe_key=key,
        method=method,
        scope=scope,
    )
    return df_keep.reset_index(drop=True), report


def validate_core_fields(
    df: pd.DataFrame,
    *,
    stake_col: str = "stake",
    odds_col: str = "odds",
    profit_col: str = "bet_profit",
) -> pd.DataFrame:
    """
    Return a compact validation table (counts + rates) for stake/odds/profit.
    """
    out = df.copy()

    def _is_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    stake = _is_num(out[stake_col]) if stake_col in out.columns else pd.Series([], dtype=float)
    odds = _is_num(out[odds_col]) if odds_col in out.columns else pd.Series([], dtype=float)
    profit = _is_num(out[profit_col]) if profit_col in out.columns else pd.Series([], dtype=float)

    n = len(out)

    def _rate(x: int) -> float:
        return float(x) / float(n) if n else 0.0

    rows = []

    if stake_col in out.columns:
        rows += [
            ("stake_missing", int(stake.isna().sum()), _rate(int(stake.isna().sum()))),
            ("stake_le_0", int((stake <= 0).sum()), _rate(int((stake <= 0).sum()))),
        ]
    else:
        rows.append(("stake_missing_column", n, 1.0))

    if odds_col in out.columns:
        rows += [
            ("odds_missing", int(odds.isna().sum()), _rate(int(odds.isna().sum()))),
            ("odds_eq_0", int((odds == 0).sum()), _rate(int((odds == 0).sum()))),
        ]
    else:
        rows.append(("odds_missing_column", n, 1.0))

    if profit_col in out.columns:
        rows += [
            ("profit_missing", int(profit.isna().sum()), _rate(int(profit.isna().sum()))),
        ]
    else:
        rows.append(("profit_missing_column", n, 1.0))

    return pd.DataFrame(rows, columns=["check", "count", "rate"])
