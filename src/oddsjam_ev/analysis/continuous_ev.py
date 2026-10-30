"""
continuous_ev.py
===============================================================================
Purpose
-------
Define lossless source EV parsing and exact continuous threshold semantics.

What this module provides
-------------------------
Typed percent-point previews, exact decimal strings and explicit quality statuses.

Design Notes
------------
Advertised EV is a source theoretical estimate, not realized ROI or verified edge.
Float previews never determine exact threshold membership. Bands are independent.
===============================================================================
"""

import math
import operator
import re
from decimal import Decimal, InvalidOperation

import pandas as pd

OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}
DECIMAL_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
CONTRACT = {
    "contract_version": 1,
    "source_column": "percentage",
    "units": "percentage_points",
    "example": "3.125 means 3.125 percent; fraction 0.03125",
    "meaning": "source_advertised_theoretical_ev_not_realized_return",
    "source_semantics_status": "declared_export_interpretation_not_independently_verified",
    "temporal_role": "source_current_unverified",
    "decision_time_verified": False,
    "threshold_representation": "exact_decimal_percent_string_no_rounding_or_tolerance",
    "numeric_preview": "nullable_float64_approximate_not_threshold_authority",
    "ticket_ev": "unavailable_without_explicit_ticket_model",
}


def parse_ev(values: pd.Series) -> pd.DataFrame:
    """Parse source percent-point values without inferring units from magnitude.

    Parameters
    ----------
    values : pandas.Series
        Source strings or already typed values. Typed input cannot recover its
        original lexical representation; provenance records this distinction.

    Returns
    -------
    pandas.DataFrame
        Original text, exact decimal percent/fraction, numeric preview and status.
        Missing, invalid and conflicting values never become zero.
    """
    rows = []
    for value in values:
        missing = value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value))
        raw = None if missing else str(value)
        text = "" if raw is None else raw.strip()
        status = "missing" if not text else "conflict" if text == "<conflict>" else "invalid"
        decimal_percent, decimal_fraction, preview = None, None, None
        if (
            text
            and status != "conflict"
            and not isinstance(value, bool)
            and len(text) <= 4096
            and DECIMAL_TEXT.fullmatch(text)
        ):
            try:
                number = Decimal(text)
                number_float = float(number)
                if (
                    number.is_finite()
                    and math.isfinite(number_float)
                    and (number == 0 or number_float != 0)
                ):
                    # Tuple scaling is exact even beyond the Decimal context precision.
                    sign, digits, exponent = number.as_tuple()
                    decimal_percent = str(number)
                    decimal_fraction = str(Decimal((sign, digits, exponent - 2)))
                    preview, status = number_float, "valid"
            except (InvalidOperation, ValueError, OverflowError):
                pass
        rows.append(
            {
                "source_ev_raw": raw,
                "source_representation": "text"
                if isinstance(value, str)
                else "typed_input_original_lexeme_unavailable",
                "advertised_ev_percent_exact": decimal_percent,
                "advertised_ev_fraction_exact": decimal_fraction,
                "advertised_ev_percent": preview,
                "ev_status": status,
            }
        )
    result = pd.DataFrame(
        rows,
        index=values.index,
        columns=[
            "source_ev_raw",
            "source_representation",
            "advertised_ev_percent_exact",
            "advertised_ev_fraction_exact",
            "advertised_ev_percent",
            "ev_status",
        ],
    )
    result["advertised_ev_percent"] = result.advertised_ev_percent.astype("Float64")
    return result


def ev_threshold(values: pd.Series, comparison: str, endpoint: str) -> pd.Series:
    """Evaluate a declared percent-point threshold using exact decimal arithmetic.

    Parameters
    ----------
    values : pandas.Series
        Original row-level source EV, never aggregate band labels.
    comparison : str
        One of >, >=, <, <=, ==.
    endpoint : str
        Finite decimal percent-point endpoint, serialized as text (e.g. '3').

    Returns
    -------
    pandas.Series
        Boolean membership; missing, invalid and conflicting inputs are False.
        Economic eligibility and frozen hypothesis families are separate concerns.
    """
    if (
        comparison not in OPERATORS
        or not isinstance(endpoint, str)
        or not endpoint.strip()
        or len(endpoint) > 256
        or not DECIMAL_TEXT.fullmatch(endpoint.strip())
    ):
        raise ValueError("Explicit operator and finite decimal endpoint string required")
    try:
        cutoff = Decimal(endpoint)
        if not cutoff.is_finite():
            raise InvalidOperation
    except InvalidOperation:
        raise ValueError("Finite decimal endpoint required") from None
    parsed = parse_ev(values)
    return parsed.apply(
        lambda r: r.ev_status == "valid"
        and OPERATORS[comparison](Decimal(r.advertised_ev_percent_exact), cutoff),
        axis=1,
    ).astype(bool)


def build_ev_tables(tables: dict) -> dict:
    """Create additive EV views without changing canonical economic eligibility.

    Parameters
    ----------
    tables : dict
        Verified canonical source_features and bet_facts tables.

    Returns
    -------
    dict
        Current source-observation and wager-candidate EV views. Ticket EV is
        explicitly unavailable, even if source legs report identical values.
    """
    source, bets = tables["source_features"].copy(), tables["bet_facts"]
    source_column_present = "percentage" in source
    if not source_column_present:
        source["percentage"] = None
    source_columns = [
        "_ingest_observation_id",
        "_ingest_record_id",
        "_ingest_import_id",
        "_ingest_namespace",
        "_ingest_timestamp_timezone",
        "bet_id",
        "percentage",
    ]
    wager_columns = [
        "bet_id",
        "source_observation_id",
        "fact_grain",
        "product",
        "execution_status",
        "reported_roi_eligible",
        "percentage",
    ]
    for frame, columns, identity in (
        (source, source_columns, "_ingest_observation_id"),
        (bets, wager_columns, "bet_id"),
    ):
        if (
            set(columns) - set(frame)
            or frame[identity].isna().any()
            or frame[identity].duplicated().any()
        ):
            raise ValueError("Missing EV provenance or duplicate canonical identities")
    views = {}
    for name, frame, columns in (
        ("source_ev", source, source_columns),
        ("wager_ev", bets, wager_columns),
    ):
        parsed = parse_ev(frame.percentage)
        parsed["source_percentage_column_present"] = source_column_present
        if not source_column_present:
            parsed["ev_status"] = "unavailable_source_column"
        if name == "wager_ev":
            unsupported = frame.fact_grain.ne("source_single_candidate")
            parsed.loc[
                unsupported,
                [
                    "advertised_ev_percent_exact",
                    "advertised_ev_fraction_exact",
                    "advertised_ev_percent",
                ],
            ] = None
            parsed.loc[unsupported, "ev_status"] = "unavailable_ticket_model"
        views[name] = pd.concat([frame[columns].copy(), parsed], axis=1)
    return views
