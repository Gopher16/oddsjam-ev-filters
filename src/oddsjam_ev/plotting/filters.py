from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def _fmt_dollars(x: float, _pos: int) -> str:
    return f"${x:,.0f}"


def _normalize_filters(df: pd.DataFrame, filters: object, *, filter_col: str) -> list[str]:
    """
    Normalize 'filters' into list[str].

    Supported:
      - filters="some name" => [that name]
      - filters=list/Index/ndarray/etc => list[str]
      - filters=None is intentionally NOT resolved here (handled by caller for global_view logic)
    """
    if filters is None:
        return []

    if isinstance(filters, str):
        return [str(filters)]

    if isinstance(filters, (pd.Index, np.ndarray, tuple, set, list)):
        return [str(x) for x in list(filters)]

    if isinstance(filters, Iterable):
        return [str(x) for x in list(filters)]

    raise TypeError(f"Unsupported filters type: {type(filters)}")


# -----------------------------------------------------------------------------
# ROI bubble/scatter (filter-summary)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RoiScatterConfig:
    filter_col: str = "saved_filter_names"
    x_col: str = "total_ev_roi"
    y_col: str = "total_actual_roi"
    size_col: str = "n_bets"

    size_scale: float = 2000.0
    size_min_frac: float = 0.05

    figsize: tuple[int, int] = (10, 8)
    alpha: float = 0.7
    edgecolors: str = "black"
    linewidths: float = 0.5

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

    display_top_n: int | None = None
    display_by_col: str = "total_stake"
    display_filters: Sequence[str] | None = None

    annotate_top_n: int = 12
    annotate_by_col: str = "total_stake"
    annotate_filters: Sequence[str] | None = None

    annotate_fontsize: int = 8
    annotate_xytext: tuple[int, int] = (5, 3)

    annotate_connectors: bool = True
    connector_alpha: float = 0.6
    connector_linewidth: float = 0.8
    annotate_point_marker: bool = True
    point_marker_size: float = 18.0

    title: str = "Filter-level ROI: Expected vs Actual (bubble size = #bets)"
    xlabel: str = "Total EV ROI"
    ylabel: str = "Total Actual ROI"

    grid: bool = True
    legend: bool = True

    remove_outliers: bool = False
    outlier_quantile: float = 0.99
    outlier_method: str = "distance"  # "distance" or "axis_quantile"
    outlier_apply: str = "both"  # "both" or "either"


def _resolve_filter_names(df: pd.DataFrame, filter_col: str) -> pd.Series:
    if filter_col in df.columns:
        return df[filter_col].astype(str)
    return df.index.astype(str)


def _resolve_sort_col(sort_key: str, cfg: RoiScatterConfig) -> str:
    key = str(sort_key)
    alias_map = {
        "total_stake": "total_stake",
        "total_bets": cfg.size_col,
        "total_roi": cfg.y_col,
        "total_ev": cfg.x_col,
    }
    return alias_map.get(key, key)


def _robust_iqr_scale(s: pd.Series) -> float:
    q75 = float(s.quantile(0.75))
    q25 = float(s.quantile(0.25))
    scale = q75 - q25
    return scale if scale > 0 else 1.0


def _trim_outliers_by_distance(
    sub: pd.DataFrame, *, keep_quantile: float
) -> tuple[pd.DataFrame, int]:
    q = float(keep_quantile)
    if not (0.5 < q < 1.0):
        raise ValueError("outlier_quantile must be in (0.5, 1.0).")

    n = int(len(sub))
    drop_n = int(np.ceil((1.0 - q) * n))
    if drop_n <= 0:
        return sub, 0

    cx = float(sub["_x"].median())
    cy = float(sub["_y"].median())
    sx = _robust_iqr_scale(sub["_x"])
    sy = _robust_iqr_scale(sub["_y"])

    zx = (sub["_x"] - cx) / sx
    zy = (sub["_y"] - cy) / sy
    dist = np.sqrt(zx * zx + zy * zy)

    keep_n = max(n - drop_n, 1)
    keep_idx = dist.nsmallest(keep_n).index
    trimmed = sub.loc[keep_idx].copy()
    dropped = n - int(len(trimmed))
    return trimmed, dropped


def _trim_outliers_by_axis_quantiles(
    sub: pd.DataFrame, *, keep_quantile: float, apply: str
) -> tuple[pd.DataFrame, int]:
    q = float(keep_quantile)
    if not (0.5 < q < 1.0):
        raise ValueError("outlier_quantile must be in (0.5, 1.0).")

    x_lo, x_hi = sub["_x"].quantile(1 - q), sub["_x"].quantile(q)
    y_lo, y_hi = sub["_y"].quantile(1 - q), sub["_y"].quantile(q)

    in_x = sub["_x"].between(x_lo, x_hi)
    in_y = sub["_y"].between(y_lo, y_hi)

    if apply == "either":
        keep_mask = in_x | in_y
    elif apply == "both":
        keep_mask = in_x & in_y
    else:
        raise ValueError("outlier_apply must be 'either' or 'both'.")

    dropped = int((~keep_mask).sum())
    trimmed = sub.loc[keep_mask].copy()
    return trimmed, dropped


def plot_filter_roi_scatter(df: pd.DataFrame, cfg: RoiScatterConfig | None = None) -> plt.Axes:
    """
    Scatter/bubble plot for filter-level summary tables.

    Expected input: a filter_summary dataframe where each row represents a filter
    and includes columns like:
      - total_ev_roi
      - total_actual_roi
      - n_bets
      - total_stake (for labeling)
    """
    cfg = cfg or RoiScatterConfig()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    display_sort_col = _resolve_sort_col(cfg.display_by_col, cfg)
    annotate_sort_col = _resolve_sort_col(cfg.annotate_by_col, cfg)

    required = [cfg.x_col, cfg.y_col, cfg.size_col, annotate_sort_col]
    if cfg.display_top_n is not None:
        required.append(display_sort_col)

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

    if cfg.display_filters is not None:
        wanted = set(str(v) for v in cfg.display_filters)
        sub = sub[sub["_filter_name"].astype(str).isin(wanted)].copy()
    elif cfg.display_top_n is not None:
        top_n = int(cfg.display_top_n)
        if top_n <= 0:
            raise ValueError("display_top_n must be a positive integer when provided.")
        sub = sub.sort_values(display_sort_col, ascending=False).head(top_n).copy()

    if sub.empty:
        raise ValueError("No rows remain after display filtering (display_* settings).")

    dropped_outliers = 0
    if cfg.remove_outliers:
        if cfg.outlier_method == "distance":
            sub, dropped_outliers = _trim_outliers_by_distance(
                sub, keep_quantile=cfg.outlier_quantile
            )
        elif cfg.outlier_method == "axis_quantile":
            sub, dropped_outliers = _trim_outliers_by_axis_quantiles(
                sub, keep_quantile=cfg.outlier_quantile, apply=cfg.outlier_apply
            )
        else:
            raise ValueError("outlier_method must be 'distance' or 'axis_quantile'.")

        if sub.empty:
            raise ValueError(
                "Outlier removal dropped all points. Loosen outlier_quantile or disable."
            )

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
        zorder=2,
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
            zorder=1,
        )

    if cfg.show_breakeven_line:
        ax.axhline(
            0,
            color=cfg.breakeven_color,
            linestyle=cfg.breakeven_linestyle,
            linewidth=cfg.breakeven_linewidth,
            label=cfg.breakeven_label,
            zorder=1,
        )

    to_label = pd.DataFrame()
    if cfg.annotate_filters is not None:
        wanted = set(str(v) for v in cfg.annotate_filters)
        to_label = sub[sub["_filter_name"].astype(str).isin(wanted)].copy()
    else:
        label_n = int(cfg.annotate_top_n) if cfg.annotate_top_n is not None else 0
        if label_n > 0:
            to_label = sub.sort_values(annotate_sort_col, ascending=False).head(label_n).copy()

    if not to_label.empty:
        if cfg.annotate_point_marker:
            ax.scatter(
                to_label["_x"],
                to_label["_y"],
                s=cfg.point_marker_size,
                edgecolors="black",
                linewidths=0.8,
                alpha=1.0,
                zorder=3,
            )

        for _, row in to_label.iterrows():
            arrowprops = None
            if cfg.annotate_connectors:
                arrowprops = dict(
                    arrowstyle="-", linewidth=cfg.connector_linewidth, alpha=cfg.connector_alpha
                )

            ax.annotate(
                str(row["_filter_name"]),
                (float(row["_x"]), float(row["_y"])),
                xytext=cfg.annotate_xytext,
                textcoords="offset points",
                fontsize=cfg.annotate_fontsize,
                arrowprops=arrowprops,
                zorder=4,
            )

    title = cfg.title
    bits = []
    if cfg.display_filters is not None:
        bits.append(f"displayed: {len(sub)}")
    elif cfg.display_top_n is not None:
        bits.append(f"displayed top {cfg.display_top_n} by {cfg.display_by_col}")
    if cfg.remove_outliers and dropped_outliers > 0:
        bits.append(f"outliers removed: {dropped_outliers}")
    if bits:
        title = f"{title} ({', '.join(bits)})"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(cfg.xlabel, fontsize=12)
    ax.set_ylabel(cfg.ylabel, fontsize=12)

    if cfg.grid:
        ax.grid(True, linestyle="--", alpha=0.3)
    if cfg.legend:
        ax.legend(frameon=False, loc="best")

    return ax


# -----------------------------------------------------------------------------
# Profit-by-group / Profit-by-sport
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProfitByGroupPlotConfig:
    figsize: tuple[int, int] = (12, 6)
    title_prefix: str = "Profit by Group"
    grid_alpha: float = 0.3
    zero_line: bool = True
    annotate_counts: bool = True
    min_height: int = 5

    min_bets_per_group: int = 0
    top_k: int | None = None

    # if True and filters is list/str: combine filters into one plot
    combine_filters: bool = False


def _profit_by_group_agg(
    df: pd.DataFrame, *, group_col: str, profit_col: str, stake_col: str
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
    agg: pd.DataFrame, *, min_bets_per_group: int, top_k: int | None
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
    agg: pd.DataFrame, *, title: str, group_label: str, cfg: ProfitByGroupPlotConfig
) -> None:
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
    filters: object = None,
    global_view: bool = True,
    cfg: ProfitByGroupPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    profit_col: str = "bet_profit",
    stake_col: str = "stake",
) -> None:
    """
    Exactly matches your notebook usage pattern:

    1) Single unfiltered GLOBAL plot:
         plot_profit_by_group_for_filters(df, group_col="sport", filters=None, global_view=True)

    2) Single plot for a specific filter:
         plot_profit_by_group_for_filters(df, group_col="sport", filters="FILTER NAME")

    3) Multiple per-filter plots (only if you explicitly pass a list):
         plot_profit_by_group_for_filters(df, group_col="sport", filters=[...])
    """
    cfg = cfg or ProfitByGroupPlotConfig()

    for c in [group_col, profit_col, stake_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # ---- GLOBAL single plot across all filters
    if filters is None and global_view:
        sub = df.copy()
        sub[profit_col] = pd.to_numeric(sub[profit_col], errors="coerce")
        sub[stake_col] = pd.to_numeric(sub[stake_col], errors="coerce")
        sub = sub.dropna(subset=[profit_col])
        if sub.empty:
            print("No rows remain after coercing profit.")
            return

        agg = _profit_by_group_agg(
            sub, group_col=group_col, profit_col=profit_col, stake_col=stake_col
        )
        agg = _apply_min_and_topk(agg, min_bets_per_group=cfg.min_bets_per_group, top_k=cfg.top_k)
        _plot_profit_by_group_barh(
            agg, title=f"{cfg.title_prefix} — ALL FILTERS", group_label=group_col, cfg=cfg
        )
        return

    if filter_col not in df.columns:
        raise KeyError(f"Missing required column: {filter_col}")

    filters_list = _normalize_filters(df, filters, filter_col=filter_col)
    if not filters_list:
        raise ValueError(
            "filters must be a filter name or list of filter names when global_view=False."
        )

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
        _plot_profit_by_group_barh(
            agg, title=f"{cfg.title_prefix} — {title_filters}", group_label=group_col, cfg=cfg
        )
        return

    for f in filters_list:
        fdf = sub[sub[filter_col] == f].copy()
        if fdf.empty:
            continue

        agg = _profit_by_group_agg(
            fdf, group_col=group_col, profit_col=profit_col, stake_col=stake_col
        )
        agg = _apply_min_and_topk(agg, min_bets_per_group=cfg.min_bets_per_group, top_k=cfg.top_k)
        _plot_profit_by_group_barh(
            agg, title=f"{cfg.title_prefix} — {f}", group_label=group_col, cfg=cfg
        )


@dataclass(frozen=True)
class ProfitBySportPlotConfig(ProfitByGroupPlotConfig):
    title_prefix: str = "Profit by Sport"


def plot_profit_by_sport_for_filters(
    df: pd.DataFrame,
    filters: object = None,
    *,
    global_view: bool = True,
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
        global_view=global_view,
        cfg=cfg,
        filter_col=filter_col,
        profit_col=profit_col,
        stake_col=stake_col,
    )
