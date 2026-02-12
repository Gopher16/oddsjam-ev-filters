from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import FuncFormatter


def _fmt_dollars(x: float, _pos: int) -> str:
    return f"${x:,.0f}"


def _to_plot_time(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(dt):
        return dt.dt.tz_localize(None)
    return dt


def _normalize_filters(df: pd.DataFrame, filters: object, *, filter_col: str) -> list[str]:
    if filters is None:
        return []
    if isinstance(filters, str):
        return [str(filters)]
    if isinstance(filters, (pd.Index, np.ndarray, tuple, set, list)):
        return [str(x) for x in list(filters)]
    if isinstance(filters, Iterable):
        return [str(x) for x in list(filters)]
    raise TypeError(f"Unsupported filters type: {type(filters)}")


def top_n_filters_by_metric(
    filter_summary: pd.DataFrame, *, metric: str, n: int, ascending: bool = False
) -> list[str]:
    if metric not in filter_summary.columns:
        raise KeyError(f"metric '{metric}' not found in filter_summary columns")
    return (
        filter_summary.sort_values(metric, ascending=ascending).head(n).index.astype(str).tolist()
    )


# -----------------------------------------------------------------------------
# Cumulative profit over time (one line per filter)
# NOTE: This plot already has a natural "global" view: filters=None => all filters.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CumProfitPlotConfig:
    figsize: tuple[int, int] = (12, 6)
    linewidth: float = 2.0
    zero_line: bool = True
    grid: bool = True
    legend_ncol: int = 2
    y_pad_frac: float = 0.08
    min_points_per_line: int = 2
    resample: str | None = None  # None, "D", "W", "M"
    title_prefix: str = "Cumulative Actual Profit Over Time — "


def plot_cum_profit_top_filters(
    df: pd.DataFrame,
    *,
    filter_summary: pd.DataFrame,
    top_n: int = 6,
    sort_col: str = "total_actual_profit",
    cfg: CumProfitPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
    profit_col: str = "bet_profit",
) -> None:
    cfg = cfg or CumProfitPlotConfig()
    filters = top_n_filters_by_metric(filter_summary, metric=sort_col, n=top_n, ascending=False)
    plot_cumulative_profit_over_time(
        df,
        filters=filters,
        cfg=cfg,
        filter_col=filter_col,
        time_col=time_col,
        profit_col=profit_col,
        title=f"{cfg.title_prefix}Top {top_n} Filters",
    )


def plot_cumulative_profit_over_time(
    df: pd.DataFrame,
    *,
    filters: object = None,
    cfg: CumProfitPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
    profit_col: str = "bet_profit",
    title: str | None = None,
) -> None:
    cfg = cfg or CumProfitPlotConfig()

    for c in [filter_col, time_col, profit_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # filters=None => all filters (existing behavior)
    if filters is None:
        filters_list = sorted(df[filter_col].dropna().astype(str).unique().tolist())
    else:
        filters_list = _normalize_filters(df, filters, filter_col=filter_col)

    if not filters_list:
        print("No filters found to plot.")
        return

    sub = df[df[filter_col].isin(filters_list)].copy()
    if sub.empty:
        print("No rows match provided filters.")
        return

    sub["plot_time"] = _to_plot_time(sub[time_col])
    sub[profit_col] = pd.to_numeric(sub[profit_col], errors="coerce")
    sub = sub.dropna(subset=["plot_time", profit_col])
    if sub.empty:
        print("No rows remain after parsing time/profit.")
        return

    fig, ax = plt.subplots(figsize=cfg.figsize)

    if cfg.resample is None:
        sub = sub.sort_values([filter_col, "plot_time"])
        sub["cum_profit"] = sub.groupby(filter_col)[profit_col].cumsum()

        for f in filters_list:
            g = sub[sub[filter_col] == f]
            if len(g) < cfg.min_points_per_line:
                continue
            ax.plot(g["plot_time"], g["cum_profit"], label=str(f), linewidth=cfg.linewidth)
    else:
        res = (
            sub.set_index("plot_time")
            .groupby(filter_col)[profit_col]
            .resample(cfg.resample)
            .sum()
            .groupby(level=0)
            .cumsum()
            .rename("cum_profit")
            .reset_index()
        )
        for f in filters_list:
            g = res[res[filter_col] == f]
            if len(g) < cfg.min_points_per_line:
                continue
            ax.plot(g["plot_time"], g["cum_profit"], label=str(f), linewidth=cfg.linewidth)

    if cfg.zero_line:
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

    ax.set_title(
        title
        or f"{cfg.title_prefix}{'ALL Filters' if filters is None else f'{len(filters_list)} Filters'}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Profit ($)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_dollars))

    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    fig.autofmt_xdate()

    if cfg.grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(title="Saved Filter", frameon=False, ncol=1, loc="best")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Cumulative profit by group (ONE FIGURE global; ONE FIGURE per specified filter)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CumProfitByGroupPlotConfig(CumProfitPlotConfig):
    pass


CumProfitByGroupConfig = CumProfitByGroupPlotConfig  # backwards-compatible alias


def plot_cum_profit_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    filters: object = None,
    global_view: bool = True,
    cfg: CumProfitByGroupPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
    profit_col: str = "bet_profit",
    min_points_per_line: int | None = None,
) -> None:
    """
    Desired behavior:
      1) Global unfiltered plot:
           filters=None, global_view=True -> ONE figure (all filters combined), lines by group_col
      2) Single-filter plot:
           filters="Filter A" -> ONE figure, lines by group_col within that filter
      3) Multiple explicit filters:
           filters=[...] -> one figure per filter (existing behavior)
    """
    cfg = cfg or CumProfitByGroupPlotConfig()
    mpl = min_points_per_line if min_points_per_line is not None else cfg.min_points_per_line

    for c in [group_col, time_col, profit_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    if filters is None and global_view:
        # --- one global figure, all filters combined
        sub = df.copy()
        sub["plot_time"] = _to_plot_time(sub[time_col])
        sub[profit_col] = pd.to_numeric(sub[profit_col], errors="coerce")
        sub = sub.dropna(subset=["plot_time", profit_col, group_col])
        if sub.empty:
            print("No rows remain after parsing time/profit/group.")
            return

        fig, ax = plt.subplots(figsize=cfg.figsize)

        if cfg.resample is None:
            sub = sub.sort_values([group_col, "plot_time"])
            sub["cum_profit"] = sub.groupby(group_col)[profit_col].cumsum()
            for grp, g in sub.groupby(group_col):
                if len(g) < mpl:
                    continue
                ax.plot(g["plot_time"], g["cum_profit"], label=str(grp), linewidth=cfg.linewidth)
        else:
            res = (
                sub.set_index("plot_time")
                .groupby(group_col)[profit_col]
                .resample(cfg.resample)
                .sum()
                .groupby(level=0)
                .cumsum()
                .rename("cum_profit")
                .reset_index()
            )
            for grp, g in res.groupby(group_col):
                if len(g) < mpl:
                    continue
                ax.plot(g["plot_time"], g["cum_profit"], label=str(grp), linewidth=cfg.linewidth)

        if cfg.zero_line:
            ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

        ax.set_title(f"{cfg.title_prefix}ALL FILTERS (by {group_col})", fontsize=13)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Profit ($)")
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_dollars))

        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        fig.autofmt_xdate()

        if cfg.grid:
            ax.grid(True, linestyle="--", alpha=0.3)

        ax.legend(title=group_col, frameon=False, ncol=cfg.legend_ncol, loc="best")
        plt.tight_layout()
        plt.show()
        return

    # --- filtered mode(s)
    if filter_col not in df.columns:
        raise KeyError(f"Missing required column: {filter_col}")

    filters_list = _normalize_filters(df, filters, filter_col=filter_col)
    if not filters_list:
        raise ValueError("Provide a filter name or list of filter names when global_view=False.")

    sub = df[df[filter_col].isin(filters_list)].copy()
    if sub.empty:
        print("No rows match provided filters.")
        return

    sub["plot_time"] = _to_plot_time(sub[time_col])
    sub[profit_col] = pd.to_numeric(sub[profit_col], errors="coerce")
    sub = sub.dropna(subset=["plot_time", profit_col])
    if sub.empty:
        print("No rows remain after parsing time/profit.")
        return

    for f in filters_list:
        fdf = sub[sub[filter_col] == f].copy()
        if fdf.empty:
            continue

        fig, ax = plt.subplots(figsize=cfg.figsize)

        if cfg.resample is None:
            fdf = fdf.sort_values([group_col, "plot_time"])
            fdf["cum_profit"] = fdf.groupby(group_col)[profit_col].cumsum()
            for grp, g in fdf.groupby(group_col):
                if len(g) < mpl:
                    continue
                ax.plot(g["plot_time"], g["cum_profit"], label=str(grp), linewidth=cfg.linewidth)
        else:
            res = (
                fdf.set_index("plot_time")
                .groupby(group_col)[profit_col]
                .resample(cfg.resample)
                .sum()
                .groupby(level=0)
                .cumsum()
                .rename("cum_profit")
                .reset_index()
            )
            for grp, g in res.groupby(group_col):
                if len(g) < mpl:
                    continue
                ax.plot(g["plot_time"], g["cum_profit"], label=str(grp), linewidth=cfg.linewidth)

        if cfg.zero_line:
            ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

        ax.set_title(
            f"{cfg.title_prefix}{f}" + (f" (Resampled: {cfg.resample})" if cfg.resample else ""),
            fontsize=13,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Profit ($)")
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_dollars))

        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        fig.autofmt_xdate()

        if cfg.grid:
            ax.grid(True, linestyle="--", alpha=0.3)

        ax.legend(title=group_col, frameon=False, ncol=cfg.legend_ncol, loc="best")
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Bet distribution/count over time (ONE FIGURE global; ONE FIGURE per specified filter)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BetDistributionOverTimePlotConfig:
    figsize: tuple[int, int] = (12, 4)
    resample: str = "D"  # "D", "W", "M"
    kind: str = "line"  # "line" or "bar"
    grid: bool = True
    title_prefix: str = "Bet Volume Over Time — "
    y_label: str = "Bets"
    min_points: int = 1


def plot_bet_distribution_over_time(
    df: pd.DataFrame,
    *,
    filters: object = None,
    global_view: bool = True,
    cfg: BetDistributionOverTimePlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
) -> None:
    """
    Desired behavior:
      - Global plot (all filters combined): filters=None, global_view=True -> ONE figure
      - Single filter plot: filters="Filter A" -> ONE figure
    """
    cfg = cfg or BetDistributionOverTimePlotConfig()

    for c in [time_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    sub = df.copy()
    if filter_col in sub.columns and filters is not None:
        filters_list = _normalize_filters(sub, filters, filter_col=filter_col)
        sub = sub[sub[filter_col].isin(filters_list)].copy()

    sub["plot_time"] = _to_plot_time(sub[time_col])
    sub = sub.dropna(subset=["plot_time"])
    if sub.empty:
        print("No rows remain after parsing time.")
        return

    if filters is None and global_view:
        s = sub.set_index("plot_time").resample(cfg.resample).size().rename("n_bets").reset_index()
        if len(s) < cfg.min_points:
            return
        fig, ax = plt.subplots(figsize=cfg.figsize)
        if cfg.kind == "bar":
            ax.bar(s["plot_time"], s["n_bets"])
        else:
            ax.plot(s["plot_time"], s["n_bets"], linewidth=2.0)
        ax.set_title(f"{cfg.title_prefix}ALL FILTERS (Resampled: {cfg.resample})", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel(cfg.y_label)

        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        fig.autofmt_xdate()

        if cfg.grid:
            ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # filtered mode: ONE figure already (either a single filter string, or a list combined by selection above)
    label = "SELECTED FILTERS" if filters is not None else "ALL FILTERS"
    if isinstance(filters, str):
        label = str(filters)

    s = sub.set_index("plot_time").resample(cfg.resample).size().rename("n_bets").reset_index()
    if len(s) < cfg.min_points:
        return

    fig, ax = plt.subplots(figsize=cfg.figsize)
    if cfg.kind == "bar":
        ax.bar(s["plot_time"], s["n_bets"])
    else:
        ax.plot(s["plot_time"], s["n_bets"], linewidth=2.0)

    ax.set_title(f"{cfg.title_prefix}{label} (Resampled: {cfg.resample})", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(cfg.y_label)

    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    fig.autofmt_xdate()

    if cfg.grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


BetCountOverTimePlotConfig = BetDistributionOverTimePlotConfig


def plot_bet_count_over_time(
    df: pd.DataFrame,
    *,
    filters: object = None,
    global_view: bool = True,
    cfg: BetCountOverTimePlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
) -> None:
    cfg = cfg or BetCountOverTimePlotConfig()
    plot_bet_distribution_over_time(
        df=df,
        filters=filters,
        global_view=global_view,
        cfg=cfg,
        filter_col=filter_col,
        time_col=time_col,
    )


# -----------------------------------------------------------------------------
# Average bets by time bucket (ONE FIGURE global; ONE FIGURE per specified filter)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AvgBetsByTimeBucketPlotConfig:
    bucket: str = "dow"  # "dow", "month", "hour"
    figsize: tuple[int, int] = (10, 4)
    kind: str = "bar"  # "bar" or "line"
    grid: bool = True
    title_prefix: str = "Average Bets by Time Bucket — "
    y_label: str = "Avg Bets"
    sort: bool = True


def plot_avg_bets_by_time_bucket(
    df: pd.DataFrame,
    *,
    filters: object = None,
    global_view: bool = True,
    cfg: AvgBetsByTimeBucketPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
    bucket: str | None = None,
) -> None:
    cfg = cfg or AvgBetsByTimeBucketPlotConfig()

    for c in [time_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    use_bucket = bucket or cfg.bucket
    if use_bucket not in {"dow", "month", "hour"}:
        raise ValueError("bucket must be one of: 'dow', 'month', 'hour'")

    sub = df.copy()
    label = "ALL FILTERS"
    if filter_col in sub.columns and filters is not None:
        filters_list = _normalize_filters(sub, filters, filter_col=filter_col)
        sub = sub[sub[filter_col].isin(filters_list)].copy()
        if isinstance(filters, str):
            label = str(filters)
        else:
            label = "SELECTED FILTERS"

    sub["plot_time"] = _to_plot_time(sub[time_col])
    sub = sub.dropna(subset=["plot_time"])
    if sub.empty:
        print("No rows remain after parsing time.")
        return

    sub["day"] = sub["plot_time"].dt.floor("D")

    # Build daily counts (for dow/month) or hourly counts (for hour)
    if use_bucket in {"dow", "month"}:
        daily = sub.groupby("day", dropna=False).size().rename("n_bets").reset_index()

        if use_bucket == "dow":
            daily["dow"] = daily["day"].dt.dayofweek
            out = (
                daily.groupby("dow", dropna=False)["n_bets"]
                .mean()
                .reset_index()
                .rename(columns={"n_bets": "avg_bets"})
            )
            dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            out["bucket_label"] = out["dow"].map(dow_map)
            if cfg.sort:
                out = out.sort_values("dow")
            x = out["bucket_label"]
            title = f"{cfg.title_prefix}{label} (bucket=dow)"
        else:
            daily["month"] = daily["day"].dt.to_period("M").astype(str)
            out = (
                daily.groupby("month", dropna=False)["n_bets"]
                .mean()
                .reset_index()
                .rename(columns={"n_bets": "avg_bets"})
            )
            if cfg.sort:
                out = out.sort_values("month")
            out["bucket_label"] = out["month"]
            x = out["bucket_label"]
            title = f"{cfg.title_prefix}{label} (bucket=month)"

    else:
        sub["hour"] = sub["plot_time"].dt.hour
        hourly_counts = sub.groupby("hour", dropna=False).size().rename("n_bets").reset_index()
        n_days = max(int(sub["day"].nunique()), 1)
        out = hourly_counts.copy()
        out["avg_bets"] = out["n_bets"] / float(n_days)
        out["bucket_label"] = out["hour"].astype(int).astype(str).str.zfill(2) + ":00"
        if cfg.sort:
            out = out.sort_values("hour")
        x = out["bucket_label"]
        title = f"{cfg.title_prefix}{label} (bucket=hour)"

    fig, ax = plt.subplots(figsize=cfg.figsize)
    if cfg.kind == "line":
        ax.plot(x, out["avg_bets"], linewidth=2.0)
    else:
        ax.bar(x, out["avg_bets"])

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Bucket")
    ax.set_ylabel(cfg.y_label)

    if cfg.grid:
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    plt.tight_layout()
    plt.show()
