from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def finite_mask(*series: pd.Series) -> np.ndarray:
    """Return boolean mask where all series are finite (not NaN/inf)."""
    mask = None
    for s in series:
        s_num = pd.to_numeric(s, errors="coerce")
        m = np.isfinite(s_num.to_numpy())
        mask = m if mask is None else (mask & m)
    if mask is None:
        raise ValueError("finite_mask() requires at least one series.")
    return mask


def bubble_sizes(
    n: pd.Series,
    *,
    max_size: float = 2000.0,
    min_frac: float = 0.05,
    sqrt: bool = True,
) -> np.ndarray:
    """Convert a count-like series into matplotlib scatter sizes."""
    n_num = pd.to_numeric(n, errors="coerce").fillna(0.0)
    n_num = n_num.clip(lower=0.0)

    scaled = np.sqrt(n_num) if sqrt else n_num
    mx = float(scaled.max()) if float(scaled.max()) > 0 else 1.0
    sizes = (scaled / mx).clip(min_frac, 1.0) * max_size
    return sizes.to_numpy()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class LabelConfig:
    label_sort_col: str = "total_stake"
    label_n: int = 12
