from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DEFAULT_EXCHANGE_BOOKS: tuple[str, ...] = (
    "Novig",
    "Prophet X",
    "Prophet Exchange",
    "4Cx",
    "Kalshi",
    "Robinhood",
)

REGIME_EXCHANGE = "exchange"
REGIME_SPORTSBOOK = "sportsbook"


def normalize_book_name(value: object) -> str:
    """Return a stripped sportsbook name as a string."""
    if value is None:
        return ""
    return str(value).strip()


def build_exchange_book_set(exchange_books: Iterable[str] | None = None) -> set[str]:
    """Build a normalized set of exchange books."""
    books = exchange_books if exchange_books is not None else DEFAULT_EXCHANGE_BOOKS
    return {normalize_book_name(book) for book in books}


def infer_regime(
    sportsbook_series: pd.Series,
    exchange_books: Iterable[str] | None = None,
) -> pd.Series:
    """
    Infer betting regime from sportsbook name.

    Parameters
    ----------
    sportsbook_series
        Series containing sportsbook / platform names.
    exchange_books
        Iterable of names that should be treated as exchanges.

    Returns
    -------
    pd.Series
        Series containing 'exchange' or 'sportsbook'.
    """
    exchange_book_set = build_exchange_book_set(exchange_books)
    normalized = sportsbook_series.fillna("").map(normalize_book_name)
    return normalized.isin(exchange_book_set).map({True: REGIME_EXCHANGE, False: REGIME_SPORTSBOOK})


def add_regime_columns(
    df: pd.DataFrame,
    sportsbook_col: str = "sportsbook",
    regime_col: str = "regime",
    has_liquidity_col: str = "has_liquidity",
    exchange_books: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Add regime-aware columns to a dataframe.

    Adds:
    - regime
    - has_liquidity (True for exchanges, False for sportsbooks)

    Parameters
    ----------
    df
        Input dataframe.
    sportsbook_col
        Column containing sportsbook / platform names.
    regime_col
        Output regime column name.
    has_liquidity_col
        Output boolean column indicating whether liquidity is expected.
    exchange_books
        Iterable of exchange platform names.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with added columns.
    """
    if sportsbook_col not in df.columns:
        raise KeyError(f"Missing sportsbook column: {sportsbook_col}")

    out = df.copy()
    out[regime_col] = infer_regime(out[sportsbook_col], exchange_books=exchange_books)
    out[has_liquidity_col] = out[regime_col].eq(REGIME_EXCHANGE)
    return out
