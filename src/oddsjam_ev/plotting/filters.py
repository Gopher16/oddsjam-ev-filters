from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def _fmt_dollars(x: float, _pos: int) -> str:
    return f"${x:,.0f}"


def _normalize_filters(filters: object) -> list[str]:
    """
    Ensure filters is a list[str] and NOT a single string.

    This prevents the common bug where passing a single string results in iterating
    over characters (and you get one plot / weird behavior).
    """
    if filters is None:
        return []
    if isinstance(filters, str):
        raise TypeError("filters must be a list/Index of filter names, not a single string.")
    if isinstance(filters, (pd.Index, np.ndarray, tuple, set)):
        return [str(x) for x in list(filters)]
    if isinstance(filters, list):
        return [str(x) for x in filters]
    if isinstance(filters, Iterable):
        return [str(x) for x in list(filters)]
    raise TypeError(f"Unsupported filters type: {type(filters)}")


# -----------------------------------------------------------------------------
# ROI bubble/scatter (filter-summary) — matches your original plot
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RoiScatterConfig:
    """
    Filter-level ROI plot config (EXPECTED vs ACTUAL), matching your original plot.

    Defaults reproduce:
      x = total_ev_roi
      y = total_actual_roi
      bubble size = n_bets
      annotate top N filters by total_stake
      parity line y=x and breakeven line y=0
    """

    filter_col: str = "saved_filter_names"  # if missing, uses df.index as names

    # Axes
    x_col: str = "total_ev_roi"
    y_col: str = "total_actual_roi"

    # Bubble sizes
    size_col: str = "n_bets"
    size_scale: float = 2000.0
    size_min_frac: float = 0.05  # clip(size/max, size_min_frac..1.0)

    # Styling
    figsize: tuple[int, int] = (10, 8)
    alpha: float = 0.7
    edgecolors: str = "black"
    linewidths: float = 0.5

    # Reference lines
    show_parity_line: bool = True
    parity_label: str = "Parity (y = x)"
    parity_color: str = "gray"
    parity_linestyle: str = "--"
    parity_linewidth: float = 1.5

    show_breakeven_line: bool = True
    breakeven_label: str = "Breakeven Profit Line"
    breakeven_color: str = "red"
    breakeven_linestyle: str = ":"
    breakeven_linewidth: float = 1.5

    # Annotations
    annotate_top_n: int = 12
    annotate_by_col: str = "total_stake"
    annotate_fontsize: int = 8
    annotate_xytext: tuple[int, int] = (5, 3)

    # Titles/labels
    title: str = "Filter-level ROI: Expected vs Actual (bubble size = #bets)"
    xlabel: str = "Total EV ROI"
    ylabel: str = "Total Actual ROI"

    # Grid / legend
    grid: bool = True
    legend: bool = True


def _resolve_filter_names(df: pd.DataFrame, filter_col: str) -> pd.Series:
    if filter_col in df.columns:
        return df[filter_col].astype(str)
    return df.index.astype(str)


def plot_filter_roi_scatter(df: pd.DataFrame, cfg: RoiScatterConfig | None = None) -> plt.Axes:
    """
    Reproduces your original plot.

    Parameters
    ----------
    df : pd.DataFrame
        Filter summary df (index or column contains filter names).
        Must include cfg.x_col, cfg.y_col, cfg.size_col, cfg.annotate_by_col.
    cfg : RoiScatterConfig

    Returns
    -------
    ax : matplotlib Axes
    """
    cfg = cfg or RoiScatterConfig()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = [cfg.x_col, cfg.y_col, cfg.size_col, cfg.annotate_by_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. Available columns: {list(df.columns)}"
        )

    fs = df.copy()
    fs["_filter_name"] = _resolve_filter_names(fs, cfg.filter_col)

    x = pd.to_numeric(fs[cfg.x_col], errors="coerce")
    y = pd.to_numeric(fs[cfg.y_col], errors="coerce")
    n = pd.to_numeric(fs[cfg.size_col], errors="coerce")

    sub = fs.assign(_x=x, _y=y, _n=n).dropna(subset=["_x", "_y", "_n"])
    if sub.empty:
        raise ValueError("No valid rows after coercing x/y/size columns to numeric.")

    max_n = float(sub["_n"].max()) if float(sub["_n"].max()) > 0 else 1.0
    size = np.clip(sub["_n"] / max_n, cfg.size_min_frac, 1.0) * cfg.size_scale

    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.scatter(
        sub["_x"],
        sub["_y"],
        s=size,
        alpha=cfg.alpha,
        edgecolors=cfg.edgecolors,
        linewidths=cfg.linewidths,
    )

    if cfg.show_parity_line:
        mn = float(min(sub["_x"].min(), sub["_y"].min()))
        mx = float(max(sub["_x"].max(), sub["_y"].max()))
        ax.plot(
            [mn, mx],
            [mn, mx],
            color=cfg.parity_color,
            linestyle=cfg.parity_linestyle,
            linewidth=cfg.parity_linewidth,
            label=cfg.parity_label,
        )

    if cfg.show_breakeven_line:
        ax.axhline(
            0,
            color=cfg.breakeven_color,
            linestyle=cfg.breakeven_linestyle,
            linewidth=cfg.breakeven_linewidth,
            label=cfg.breakeven_label,
        )

    label_n = int(cfg.annotate_top_n) if cfg.annotate_top_n is not None else 0
    if label_n > 0:
        to_label = sub.sort_values(cfg.annotate_by_col, ascending=False).head(label_n)
        for _, row in to_label.iterrows():
            ax.annotate(
                str(row["_filter_name"]),
                (float(row["_x"]), float(row["_y"])),
                xytext=cfg.annotate_xytext,
                textcoords="offset points",
                fontsize=cfg.annotate_fontsize,
            )

    ax.set_title(cfg.title, fontsize=14, fontweight="bold")
    ax.set_xlabel(cfg.xlabel, fontsize=12)
    ax.set_ylabel(cfg.ylabel, fontsize=12)

    if cfg.grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    if cfg.legend:
        ax.legend(frameon=False, loc="best")

    return ax


# -----------------------------------------------------------------------------
# Profit-by-group / Profit-by-sport (multi-plot per filter supported)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfitByGroupPlotConfig:
    """
    Profit-by-group horizontal bar chart config.

    IMPORTANT:
      - default combine_filters=False => ONE CHART PER FILTER
    """

    figsize: tuple[int, int] = (12, 6)
    title_prefix: str = "Profit by Group — Selected Filters"
    grid_alpha: float = 0.3
    zero_line: bool = True
    annotate_counts: bool = True
    min_height: int = 5

    min_bets_per_group: int = 0
    top_k: int | None = None

    combine_filters: bool = False


def _profit_by_group_agg(
    df: pd.DataFrame,
    *,
    group_col: str,
    profit_col: str,
    stake_col: str,
) -> pd.DataFrame:
    return (
        df.groupby(group_col, dropna=False)
        .agg(
            total_profit=(profit_col, "sum"),
            total_stake=(stake_col, "sum"),
            n_bets=(profit_col, "size"),
        )
        .reset_index()
    )


def _apply_min_and_topk(
    agg: pd.DataFrame,
    *,
    min_bets_per_group: int,
    top_k: int | None,
) -> pd.DataFrame:
    if min_bets_per_group > 0:
        agg = agg[agg["n_bets"] >= min_bets_per_group]

    if agg.empty:
        return agg

    agg = agg.sort_values("total_profit", ascending=True)

    if top_k is not None:
        keep_idx = (
            agg.reindex(agg["total_profit"].abs().sort_values(ascending=False).index)
            .head(top_k)
            .index
        )
        agg = agg.loc[keep_idx].sort_values("total_profit", ascending=True)

    return agg


def _plot_profit_by_group_barh(
    agg: pd.DataFrame,
    *,
    title: str,
    group_label: str,
    cfg: ProfitByGroupPlotConfig | None = None,
) -> None:
    cfg = cfg or ProfitByGroupPlotConfig()
    if agg.empty:
        print("No groups remain after filtering.")
        return

    height = max(cfg.min_height, int(0.4 * len(agg)))
    fig, ax = plt.subplots(figsize=(cfg.figsize[0], height))

    ax.barh(agg[group_label].astype(str), agg["total_profit"])

    if cfg.zero_line:
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Profit ($)", fontsize=12)
    ax.set_ylabel(group_label, fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_dollars))
    ax.grid(axis="x", linestyle="--", alpha=cfg.grid_alpha)

    if cfg.annotate_counts:
        offset = max(float(agg["total_profit"].abs().max()) * 0.01, 1.0)
        for i, (profit, n) in enumerate(zip(agg["total_profit"], agg["n_bets"], strict=False)):
            x = float(profit) + (offset if profit >= 0 else -offset)
            ha = "left" if profit >= 0 else "right"
            ax.text(x, i, f"{int(n)} bets", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_profit_by_group_for_filters(
    df: pd.DataFrame,
    *,
    group_col: str,
    filters: list[str] | pd.Index | np.ndarray,
    cfg: ProfitByGroupPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    profit_col: str = "bet_profit",
    stake_col: str = "stake",
) -> None:
    cfg = cfg or ProfitByGroupPlotConfig()
    filters_list = _normalize_filters(filters)
    if not filters_list:
        raise ValueError("Provide a non-empty list of filters.")

    for c in [filter_col, group_col, profit_col, stake_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    sub = df[df[filter_col].isin(filters_list)].copy()
    if sub.empty:
        print("No rows match the provided filters.")
        return

    sub[profit_col] = pd.to_numeric(sub[profit_col], errors="coerce")
    sub[stake_col] = pd.to_numeric(sub[stake_col], errors="coerce")
    sub = sub.dropna(subset=[profit_col])

    if cfg.combine_filters:
        agg = _profit_by_group_agg(
            sub, group_col=group_col, profit_col=profit_col, stake_col=stake_col
        )
        agg = _apply_min_and_topk(agg, min_bets_per_group=cfg.min_bets_per_group, top_k=cfg.top_k)

        title_filters = ", ".join(filters_list[:3]) + (" …" if len(filters_list) > 3 else "")
        title = f"{cfg.title_prefix} ({title_filters})"
        _plot_profit_by_group_barh(agg, title=title, group_label=group_col, cfg=cfg)
        return

    # ONE CHART PER FILTER (this is what you want for top_filters n=3)
    for f in filters_list:
        fdf = sub[sub[filter_col] == f].copy()
        if fdf.empty:
            continue

        agg = _profit_by_group_agg(
            fdf, group_col=group_col, profit_col=profit_col, stake_col=stake_col
        )
        agg = _apply_min_and_topk(agg, min_bets_per_group=cfg.min_bets_per_group, top_k=cfg.top_k)

        title = f"{cfg.title_prefix} ({f})"
        _plot_profit_by_group_barh(agg, title=title, group_label=group_col, cfg=cfg)


@dataclass(frozen=True)
class ProfitBySportPlotConfig(ProfitByGroupPlotConfig):
    title_prefix: str = "Profit by Sport — Selected Filters"


def plot_profit_by_sport_for_filters(
    df: pd.DataFrame,
    filters: list[str] | pd.Index | np.ndarray,
    *,
    cfg: ProfitBySportPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    sport_col: str = "sport",
    profit_col: str = "bet_profit",
    stake_col: str = "stake",
) -> None:
    cfg = cfg or ProfitBySportPlotConfig()
    plot_profit_by_group_for_filters(
        df,
        group_col=sport_col,
        filters=filters,
        cfg=cfg,
        filter_col=filter_col,
        profit_col=profit_col,
        stake_col=stake_col,
    )
