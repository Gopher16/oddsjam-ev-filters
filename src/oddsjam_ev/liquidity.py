# src/oddsjam_ev/liquidity.py
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

_TAG_BUCKET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("<=500", re.compile(r"liq\s*<=\s*\$?\s*500", re.IGNORECASE)),
    ("500-1k", re.compile(r"\$?\s*500\s*<\s*liq\s*<=\s*\$?\s*1k", re.IGNORECASE)),
    ("1k-2k", re.compile(r"\$?\s*1k\s*<\s*liq\s*<=\s*\$?\s*2k", re.IGNORECASE)),
    ("2k-3k", re.compile(r"\$?\s*2k\s*<\s*liq\s*<=\s*\$?\s*3k", re.IGNORECASE)),
    ("3k-4k", re.compile(r"\$?\s*3k\s*<\s*liq\s*<=\s*\$?\s*4k", re.IGNORECASE)),
    ("4k-5k", re.compile(r"\$?\s*4k\s*<\s*liq\s*<=\s*\$?\s*5k", re.IGNORECASE)),
    ("5k-6k", re.compile(r"\$?\s*5k\s*<\s*liq\s*<=\s*\$?\s*6k", re.IGNORECASE)),
    ("6k-7k", re.compile(r"\$?\s*6k\s*<\s*liq\s*<=\s*\$?\s*7k", re.IGNORECASE)),
    ("7k-8k", re.compile(r"\$?\s*7k\s*<\s*liq\s*<=\s*\$?\s*8k", re.IGNORECASE)),
    ("8k-9k", re.compile(r"\$?\s*8k\s*<\s*liq\s*<=\s*\$?\s*9k", re.IGNORECASE)),
    ("9k-10k", re.compile(r"\$?\s*9k\s*<\s*liq\s*<=\s*\$?\s*10k", re.IGNORECASE)),
    ("10k-25k", re.compile(r"\$?\s*10k\s*<\s*liq\s*<=\s*\$?\s*25k", re.IGNORECASE)),
    ("25k-50k", re.compile(r"\$?\s*25k\s*<\s*liq\s*<=\s*\$?\s*50k", re.IGNORECASE)),
    ("50k-100k", re.compile(r"\$?\s*50k\s*<\s*liq\s*<=\s*\$?\s*100k", re.IGNORECASE)),
    ("250k-500k", re.compile(r"\$?\s*250k\s*<\s*liq\s*<=\s*\$?\s*500k", re.IGNORECASE)),
]


def _bucket_from_tag(tags_val: Any) -> str | None:
    if tags_val is None or (isinstance(tags_val, float) and np.isnan(tags_val)):
        return None
    s = str(tags_val)
    for bucket, pat in _TAG_BUCKET_PATTERNS:
        if pat.search(s):
            return bucket
    return None


def _bucket_from_numeric(x: float | int | None) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    v = float(x)
    # Match tag scheme boundaries
    if v <= 500:
        return "<=500"
    if 500 < v <= 1000:
        return "500-1k"
    if 1000 < v <= 2000:
        return "1k-2k"
    if 2000 < v <= 3000:
        return "2k-3k"
    if 3000 < v <= 4000:
        return "3k-4k"
    if 4000 < v <= 5000:
        return "4k-5k"
    if 5000 < v <= 6000:
        return "5k-6k"
    if 6000 < v <= 7000:
        return "6k-7k"
    if 7000 < v <= 8000:
        return "7k-8k"
    if 8000 < v <= 9000:
        return "8k-9k"
    if 9000 < v <= 10000:
        return "9k-10k"
    if 10000 < v <= 25000:
        return "10k-25k"
    if 25000 < v <= 50000:
        return "25k-50k"
    if 50000 < v <= 100000:
        return "50k-100k"
    if 250000 < v <= 500000:
        return "250k-500k"
    # Anything outside tagging scheme: keep None rather than invent bins
    return None


def add_liquidity_bucket(
    df: pd.DataFrame,
    *,
    liquidity_col: str = "liquidity",
    tags_col: str = "tags",
    out_bucket_col: str = "liquidity_bucket",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Populate liquidity_bucket using:
      1) numeric liquidity when present
      2) else infer from tags (i.e. "Liq <= $500" style tags)

    Returns:
      (df_out, meta)
    meta includes:
      - numeric_used_n / rate
      - tag_used_n / rate
      - still_missing_n / rate
    """
    out = df.copy()

    liq_num = (
        pd.to_numeric(out[liquidity_col], errors="coerce")
        if liquidity_col in out.columns
        else pd.Series([np.nan] * len(out))
    )
    bucket_num = liq_num.apply(_bucket_from_numeric)

    bucket_tag = (
        out[tags_col].apply(_bucket_from_tag)
        if tags_col in out.columns
        else pd.Series([None] * len(out))
    )

    # Choose numeric if available, else tag
    final_bucket = bucket_num.where(bucket_num.notna(), bucket_tag)

    out[out_bucket_col] = final_bucket.astype("object")

    numeric_used = int(bucket_num.notna().sum())
    tag_used = int((bucket_num.isna() & bucket_tag.notna()).sum())
    still_missing = int((final_bucket.isna()).sum())
    n = int(len(out))

    meta = {
        "n_rows": n,
        "numeric_used_n": numeric_used,
        "numeric_used_rate": (numeric_used / n) if n else 0.0,
        "tag_used_n": tag_used,
        "tag_used_rate": (tag_used / n) if n else 0.0,
        "still_missing_n": still_missing,
        "still_missing_rate": (still_missing / n) if n else 0.0,
    }
    return out, meta
