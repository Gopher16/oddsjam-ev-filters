"""
source_csv.py
===============================================================================
Purpose
-------
Read Bet Tracker CSV bytes without coercing or dropping original field values.

What this module provides
-------------------------
Strict CSV parsing, schema drift diagnostics, and deterministic fingerprints.

Design Notes
------------
Source bytes remain authoritative. Empty strings and literal NA values differ.
===============================================================================
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math

import pandas as pd

REQUIRED = ("created_at", "sportsbook", "market_name", "bet_name", "odds", "stake", "status")
EXPECTED = (
    "created_at sportsbook sport league event_name game_id event_start_date market_name "
    "bet_name odds clv clv_source stake potential_payout status bet_type bet_profit notes "
    "is_live_bet is_free_bet is_odds_boost percentage liquidity market_width saved_filter_names "
    "parlay_id tags"
).split()


def fingerprint(value: object) -> str:
    """Return a deterministic SHA-256 hash of a JSON-compatible value.

    Parameters
    ----------
    value : object
        JSON-compatible payload.

    Returns
    -------
    str
        Hexadecimal digest; field names and values are encoded unambiguously.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_source(data: bytes) -> tuple[pd.DataFrame, dict]:
    """Validate CSV structure and preserve every field as a string.

    Parameters
    ----------
    data : bytes
        UTF-8 CSV, optionally with a byte order mark.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Source rows and schema/duplicate diagnostics.

    Notes
    -----
    Unknown columns are retained. Missing required columns, repeated headers,
    reserved metadata names, malformed CSV, and empty datasets fail explicitly.
    """
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Invalid UTF-8 CSV: {exc}") from exc
    if len(rows) < 2:
        raise ValueError("CSV must contain a header and at least one data row")
    header, values = rows[0], rows[1:]
    if len(set(header)) != len(header) or any(not h.strip() for h in header):
        raise ValueError("CSV has duplicate or blank column headers")
    if any(h.startswith("_ingest_") for h in header):
        raise ValueError("CSV uses reserved _ingest_ column prefix")
    missing = sorted(set(REQUIRED) - set(header))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    for number, row in enumerate(values, start=1):
        if len(row) != len(header):
            raise ValueError(
                f"CSV logical data row {number}: expected {len(header)} fields, got {len(row)}"
            )
    frame = pd.DataFrame(values, columns=header, dtype=str)
    invalid_numerics = {}
    for column in ("odds", "stake", "bet_profit", "percentage", "potential_payout", "clv"):
        if column in frame:
            invalid = 0
            for value in frame[column]:
                if not value:
                    continue
                try:
                    invalid += int(not math.isfinite(float(value)))
                except ValueError:
                    invalid += 1
            invalid_numerics[column] = invalid
    diagnostics = {
        "invalid_numeric_values": invalid_numerics,
        "created_at_raw_min": frame["created_at"].min(),
        "created_at_raw_max": frame["created_at"].max(),
        "watermark_semantics": "lexical raw timestamp bounds; not a safe incremental cursor",
        "status_counts": {str(k): int(v) for k, v in frame["status"].value_counts().items()},
        "columns": header,
        "added_columns": sorted(set(header) - set(EXPECTED)),
        "missing_optional_columns": sorted(set(EXPECTED) - set(header)),
        "row_count": len(frame),
        "duplicate_rows_beyond_first": int(frame.duplicated().sum()),
    }
    return frame, diagnostics
