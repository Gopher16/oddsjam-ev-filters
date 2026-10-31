"""
visual_features.py
===============================================================================
Purpose
-------
Prepare unique visual rows with source-date, exact EV and optional derivation audits.

What this module provides
-------------------------
Pinned-cohort checks and source-quality states without timezone or capacity guesses.

Design Notes
------------
Source calendar dates are retrospective labels, not verified placement/settlement time.
===============================================================================
"""

import re
from decimal import Decimal

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.continuous_ev import parse_ev

MISSING = {"", "<missing>", "<conflict>", "<invalid>"}


def band(value, edges):
    """Return exact decimal left-closed display membership, retaining overflow bins."""
    if value is None or pd.isna(value):
        return "<unavailable>"
    number = Decimal(str(value))
    bounds = [Decimal(str(v)) for v in edges]
    for low, high in zip([None, *bounds], [*bounds, None], strict=True):
        if (low is None or number >= low) and (high is None or number < high):
            return (
                f"[{'-inf' if low is None else str(low)}, {'inf' if high is None else str(high)})"
            )
    raise ValueError("Unreachable visual bucket")


def _date(raw):
    text = "" if pd.isna(raw) else str(raw).strip()
    if text in MISSING:
        return None, None, "missing_or_conflicting"
    # No locale guessing or timezone conversion for source wall-calendar labels.
    if not re.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]|$)", text):
        return None, None, "unsupported_source_format"
    try:
        stamp = pd.Timestamp(text)
        if pd.isna(stamp):
            return None, None, "invalid"
        stamp.as_unit("ns")  # Reject dates outside supported calendar arithmetic.
        has_clock = bool(re.search(r"[T ]\d{2}:\d{2}", text))
        return (
            stamp.date().isoformat(),
            stamp.hour if has_clock else None,
            "source_wall_calendar_timezone_not_inferred",
        )
    except (ValueError, TypeError):
        return None, None, "invalid"


def _time(raw, zone):
    try:
        if raw is None or pd.isna(raw) or str(raw).strip() in MISSING:
            return None
        stamp = pd.Timestamp(raw)
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is None:
            if zone is None:
                return None
            stamp = stamp.tz_localize(zone, ambiguous="NaT", nonexistent="NaT")
        return stamp.tz_convert("UTC") if pd.notna(stamp) else None
    except (TypeError, ValueError, KeyError):
        return None


def prepare_visuals(canonical, dimensions, diagnostics, cfg):
    """Reconcile inputs and prepare chart features with explicit derivation states.

    Parameters
    ----------
    canonical : dict
        Verified canonical bundle tables.
    dimensions, diagnostics : pandas.DataFrame
        Matching atlas wager dimensions and prepared sizing rows.
    cfg : VisualConfig
        Explicit display and source-evidence policy.

    Returns
    -------
    pandas.DataFrame
        One row per canonical candidate, including economic exclusions and quality flags.
    """
    facts = canonical["bet_facts"]
    if facts.empty:
        raise ValueError("Visual publication requires at least one source candidate")
    for frame in (facts, dimensions, diagnostics):
        if (
            frame.bet_id.isna().any()
            or frame.bet_id.duplicated().any()
            or set(frame.bet_id) != set(facts.bet_id)
        ):
            raise ValueError("Visual input candidate identities do not reconcile")
    facts, dimensions, diagnostics = (
        frame.set_index("bet_id").sort_index() for frame in (facts, dimensions, diagnostics)
    )
    for frame in (dimensions, diagnostics):
        for key in facts.columns.intersection(frame.columns):
            left, right = facts[key], frame[key]
            if frame is dimensions and key in {"sportsbook", "sport", "league", "market_name"}:
                # The atlas contract normalizes only these source categories.
                left, right = (
                    values.fillna("").astype(str).str.strip().replace("", "<missing>")
                    for values in (left, right)
                )
            try:
                pd.testing.assert_series_equal(left, right, check_names=False, check_dtype=False)
            except AssertionError as exc:
                raise ValueError(f"Visual source field drift: {key}") from exc
    for key, source in (("stake", "reported_stake_usd"), ("profit", "reported_profit_usd")):
        if not np.allclose(
            diagnostics[key],
            pd.to_numeric(facts[source], errors="coerce"),
            rtol=0,
            atol=0,
            equal_nan=True,
        ):
            raise ValueError("Visual economics changed from canonical source")
    columns = [
        "namespace",
        "game_id",
        "execution_status",
        "product",
        "sportsbook",
        "sport",
        "league",
        "percentage",
        "odds",
        "created_at",
        "event_start_date",
        "timestamp_timezone",
        "liquidity",
        "clv",
        "clv_source",
    ]
    work = facts[columns].copy()
    work["market_family"] = dimensions.market_family
    for key in (
        "diagnostic_eligible",
        "diagnostic_exclusions_json",
        "stake",
        "profit",
        "unit_return",
        "sizing_regime",
    ):
        work[key] = diagnostics[key]
    eligible = work.diagnostic_eligible
    if not eligible.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("Boolean visual economic eligibility required")
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = work.profit / work.stake
    if (
        not np.isfinite(work.loc[eligible, ["stake", "profit", "unit_return"]]).all().all()
        or (work.loc[eligible, "stake"] <= 0).any()
        or not np.allclose(returns[eligible], work.loc[eligible, "unit_return"], rtol=0, atol=0)
    ):
        raise ValueError("Invalid eligible visual economics")
    ev = parse_ev(work.percentage)
    for column in ev:
        work[column] = ev[column].to_numpy()
    work["advertised_ev_fraction"] = (
        pd.to_numeric(work.advertised_ev_percent, errors="coerce") / 100
    )
    work["ev_bucket"] = work.advertised_ev_percent_exact.map(lambda v: band(v, cfg.ev_edges))
    dates = work.created_at.map(_date)
    work["source_date"] = dates.map(lambda v: v[0])
    work["source_hour"] = dates.map(lambda v: v[1]).astype("Int64")
    work["source_date_status"] = dates.map(lambda v: v[2])
    work["source_clock_status"] = np.where(
        work.source_hour.notna(),
        "source_wall_clock_timezone_not_inferred",
        "missing_or_invalid_clock",
    )
    work["source_weekday"] = pd.to_datetime(work.source_date, errors="coerce").dt.dayofweek.astype(
        "Int64"
    )
    work["time_basis"] = "source_wall_calendar_retrospective_not_settlement_or_bankroll"
    numeric = pd.to_numeric(work.liquidity, errors="coerce")
    valid = np.isfinite(numeric) & numeric.ge(0)
    work["liquidity_source_numeric"] = valid
    work["liquidity_usd"] = numeric.where(valid & cfg.liquidity_usd_verified)
    work["liquidity_status"] = np.where(
        work.liquidity_usd.notna(),
        "verified_source_usd_quote_not_fill",
        "source_untrusted_or_unavailable",
    )
    work["liquidity_bucket"] = work.liquidity_usd.map(lambda v: band(v, cfg.liquidity_edges))
    hours = []
    for row in work.itertuples():
        zone = cfg.naive_timezone or (
            row.timestamp_timezone if row.timestamp_timezone not in MISSING | {"unknown"} else None
        )
        start = _time(row.event_start_date, zone) if cfg.event_time_verified else None
        placed = _time(row.created_at, zone) if cfg.placement_time_verified else None
        hours.append(
            (start - placed).total_seconds() / 3600
            if start is not None and placed is not None
            else np.nan
        )
    work["time_to_event_hours"] = hours
    work["time_to_event_bucket"] = work.time_to_event_hours.map(
        lambda v: "after_start" if pd.notna(v) and v < 0 else band(v, cfg.time_edges_hours)
    )
    work["timing_status"] = np.where(
        work.time_to_event_hours.notna(),
        "verified_source_time_derivation",
        "placement_event_semantics_or_timezone_unverified",
    )
    odds, closing = (pd.to_numeric(work[k], errors="coerce") for k in ("odds", "clv"))
    known = (
        cfg.closing_fair_american_verified
        & work.clv_source.isin(cfg.closing_sources)
        & np.isfinite(odds)
        & np.isfinite(closing)
        & odds.abs().ge(100)
        & closing.abs().ge(100)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        decimal_odds = np.where(odds > 0, 1 + odds / 100, 1 + 100 / odds.abs())
        probability = np.where(
            closing > 0, 100 / (closing + 100), closing.abs() / (closing.abs() + 100)
        )
        estimate = pd.Series(decimal_odds * probability - 1, index=work.index)
    work["closing_price_ev_fraction"] = estimate.where(known & np.isfinite(estimate))
    work["closing_status"] = np.where(
        work.closing_price_ev_fraction.notna(),
        "verified_fair_closing_reference_estimate_not_realized_profit",
        "closing_semantics_source_or_timing_unverified",
    )
    return work.reset_index().sort_values("bet_id").reset_index(drop=True)
