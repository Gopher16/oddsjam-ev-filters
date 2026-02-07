from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import FuncFormatter


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fmt_dollars(x: float, _pos: int) -> str:
    return f"${x:,.0f}"


def _to_plot_time(s: pd.Series) -> pd.Series:
    """
    Convert a datetime-like series to tz-naive timestamps for matplotlib.
    """
    dt = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(dt):
        return dt.dt.tz_localize(None)
    return dt


def _normalize_filters(filters: object) -> list[str]:
    """
    Ensure filters is a list[str] and NOT a single string.
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


def top_n_filters_by_metric(
    filter_summary: pd.DataFrame,
    *,
    metric: str,
    n: int,
    ascending: bool = False,
) -> list[str]:
    """
    Return top N filter names from a filter_summary dataframe (index = filter names).
    """
    if metric not in filter_summary.columns:
        raise KeyError(f"metric '{metric}' not found in filter_summary columns")
    return (
        filter_summary.sort_values(metric, ascending=ascending).head(n).index.astype(str).tolist()
    )


# -----------------------------------------------------------------------------
# Cumulative profit over time (one line per filter)
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
    filters: list[str] | pd.Index | np.ndarray,
    cfg: CumProfitPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
    profit_col: str = "bet_profit",
    title: str | None = None,
) -> None:
    cfg = cfg or CumProfitPlotConfig()
    filters_list = _normalize_filters(filters)
    if not filters_list:
        raise ValueError("Provide a non-empty list of filters.")

    for c in [filter_col, time_col, profit_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

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
        title or f"{cfg.title_prefix}Top {len(filters_list)} Filters",
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

    lines = ax.get_lines()
    if lines:
        ys = np.concatenate([ln.get_ydata() for ln in lines if len(ln.get_ydata())])
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    else:
        ymin, ymax = 0.0, 0.0

    pad = max(abs(ymin), abs(ymax)) * cfg.y_pad_frac + 1e-9
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(title="Saved Filter", frameon=False, ncol=1, loc="best")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Cumulative profit by group (one figure per filter; line per group)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CumProfitByGroupPlotConfig(CumProfitPlotConfig):
    pass


CumProfitByGroupConfig = CumProfitByGroupPlotConfig  # backwards-compatible alias


def plot_cum_profit_by_group(
    df: pd.DataFrame,
    *,
    filters: list[str] | pd.Index | np.ndarray,
    group_col: str,
    cfg: CumProfitByGroupPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
    profit_col: str = "bet_profit",
    min_points_per_line: int | None = None,
) -> None:
    cfg = cfg or CumProfitByGroupPlotConfig()
    filters_list = _normalize_filters(filters)
    if not filters_list:
        raise ValueError("Provide a non-empty list of filters.")

    for c in [filter_col, group_col, time_col, profit_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

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

    mpl = min_points_per_line if min_points_per_line is not None else cfg.min_points_per_line

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

        lines = ax.get_lines()
        if lines:
            ys = np.concatenate([ln.get_ydata() for ln in lines if len(ln.get_ydata())])
            ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        else:
            ymin, ymax = 0.0, 0.0

        pad = max(abs(ymin), abs(ymax)) * cfg.y_pad_frac + 1e-9
        ax.set_ylim(ymin - pad, ymax + pad)

        ax.legend(title=group_col, frameon=False, ncol=cfg.legend_ncol, loc="best")
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Bet distribution over time (one figure per filter)
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
    filters: list[str] | pd.Index | np.ndarray,
    cfg: BetDistributionOverTimePlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
) -> None:
    cfg = cfg or BetDistributionOverTimePlotConfig()
    filters_list = _normalize_filters(filters)
    if not filters_list:
        raise ValueError("Provide a non-empty list of filters.")

    for c in [filter_col, time_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    sub = df[df[filter_col].isin(filters_list)].copy()
    if sub.empty:
        print("No rows match provided filters.")
        return

    sub["plot_time"] = _to_plot_time(sub[time_col])
    sub = sub.dropna(subset=["plot_time"])
    if sub.empty:
        print("No rows remain after parsing time.")
        return

    for f in filters_list:
        fdf = sub[sub[filter_col] == f].copy()
        if fdf.empty:
            continue

        s = fdf.set_index("plot_time").resample(cfg.resample).size().rename("n_bets").reset_index()
        if len(s) < cfg.min_points:
            continue

        fig, ax = plt.subplots(figsize=cfg.figsize)

        if cfg.kind == "bar":
            ax.bar(s["plot_time"], s["n_bets"])
        else:
            ax.plot(s["plot_time"], s["n_bets"], linewidth=2.0)

        ax.set_title(f"{cfg.title_prefix}{f} (Resampled: {cfg.resample})", fontsize=12)
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


# Backwards-compatible public names (your __init__.py expects these)
BetCountOverTimePlotConfig = BetDistributionOverTimePlotConfig


def plot_bet_count_over_time(
    df: pd.DataFrame,
    *,
    filters: list[str] | pd.Index | np.ndarray,
    cfg: BetCountOverTimePlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
) -> None:
    cfg = cfg or BetCountOverTimePlotConfig()
    plot_bet_distribution_over_time(
        df=df, filters=filters, cfg=cfg, filter_col=filter_col, time_col=time_col
    )


# -----------------------------------------------------------------------------
# Average bets by time bucket (dow/month/hour) — one figure per filter
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AvgBetsByTimeBucketPlotConfig:
    """
    Plot average number of bets by:
      - bucket="dow": average daily bet count by day-of-week (Mon..Sun)
      - bucket="month": average daily bet count by month (YYYY-MM)
      - bucket="hour": average bet count by hour of day (00:00..23:00)

    Default is bucket="dow".
    """

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
    filters: list[str] | pd.Index | np.ndarray,
    cfg: AvgBetsByTimeBucketPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    time_col: str = "created_at_est",
    bucket: str | None = None,
) -> None:
    cfg = cfg or AvgBetsByTimeBucketPlotConfig()
    filters_list = _normalize_filters(filters)
    if not filters_list:
        raise ValueError("Provide a non-empty list of filters.")

    for c in [filter_col, time_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    use_bucket = bucket or cfg.bucket
    if use_bucket not in {"dow", "month", "hour"}:
        raise ValueError("bucket must be one of: 'dow', 'month', 'hour'")

    sub = df[df[filter_col].isin(filters_list)].copy()
    if sub.empty:
        print("No rows match provided filters.")
        return

    sub["plot_time"] = _to_plot_time(sub[time_col])
    sub = sub.dropna(subset=["plot_time"])
    if sub.empty:
        print("No rows remain after parsing time.")
        return

    # FIX: always create day series once (prevents KeyError: 'day')
    sub["day"] = sub["plot_time"].dt.floor("D")

    for f in filters_list:
        fdf = sub[sub[filter_col] == f].copy()
        if fdf.empty:
            continue

        if use_bucket in {"dow", "month"}:
            daily = fdf.groupby("day", dropna=False).size().rename("n_bets").reset_index()

            if use_bucket == "dow":
                daily["dow"] = daily["day"].dt.dayofweek  # 0=Mon..6=Sun
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
                title = f"{cfg.title_prefix}{f} (bucket=dow)"

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
                title = f"{cfg.title_prefix}{f} (bucket=month)"

        else:
            fdf["hour"] = fdf["plot_time"].dt.hour
            hourly_counts = fdf.groupby("hour", dropna=False).size().rename("n_bets").reset_index()

            # For hour, "avg bets" is the average number of bets in that hour across the dataset window.
            # That means: total bets in hour / number of unique days represented.
            n_days = max(int(fdf["day"].nunique()), 1)
            out = hourly_counts.copy()
            out["avg_bets"] = out["n_bets"] / float(n_days)
            out["bucket_label"] = out["hour"].astype(int).astype(str).str.zfill(2) + ":00"

            if cfg.sort:
                out = out.sort_values("hour")

            x = out["bucket_label"]
            title = f"{cfg.title_prefix}{f} (bucket=hour)"

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

        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

        plt.tight_layout()
        plt.show()
