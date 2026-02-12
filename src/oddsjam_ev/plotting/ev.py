from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from oddsjam_ev.metrics.odds import clv_ev_pct, ensure_odds_multiplier, ensure_prob_clv

GREEN = "#2E8B57"
RED = "#D9534F"


# -----------------------------------------------------------------------------
# Bucketing helpers
# -----------------------------------------------------------------------------
def make_ev_bucket(
    series: pd.Series,
    *,
    bin_size: float,
    min_x: float,
    max_x: float,
    edge: str = "right",
) -> pd.Series:
    bins = np.arange(min_x, max_x + bin_size + 1e-9, bin_size)
    labels = np.round(bins[:-1], 10)
    binned = pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=(edge == "right"),
    )
    return binned.astype(float)


def _format_dollars_signed(p: float) -> str:
    if pd.isna(p):
        return ""
    if float(p) < 0:
        return f"(${abs(float(p)):,.0f})"
    return f"${float(p):,.0f}"


def _format_total_profit_parts(total_profit: float) -> tuple[str, str, str]:
    """
    Return (prefix_text, amount_text, amount_color).
    Example:
      +1234 -> ("Total Profit", "$1,234", GREEN)
      -1234 -> ("Total Profit", "($1,234)", RED)
    """
    tp = float(total_profit) if pd.notna(total_profit) else 0.0
    if tp < 0:
        return "Total Profit", f"(${abs(tp):,.0f})", RED
    return "Total Profit", f"${tp:,.0f}", GREEN


def _add_clv_fields(sub: pd.DataFrame, *, odds_col: str, clv_col: str) -> pd.DataFrame:
    sub = ensure_prob_clv(sub, clv_col=clv_col, out_col="prob_clv")
    sub = ensure_odds_multiplier(sub, odds_col=odds_col, out_col="odds_multiplier")
    if sub["prob_clv"].notna().any() and sub["odds_multiplier"].notna().any():
        sub["clv_ev_pct"] = clv_ev_pct(sub["prob_clv"], sub["odds_multiplier"])
    else:
        sub["clv_ev_pct"] = np.nan
    return sub


# -----------------------------------------------------------------------------
# Ordering helpers (liquidity_bucket, time_to_event_bucket)
# -----------------------------------------------------------------------------
def _parse_liquidity_bucket_lower_bound(s: str) -> float:
    """
    Convert liquidity bucket label to a numeric sort key (lower bound).

    Fixes ordering where "<=500" and "500-1k" both map to 500:
    we slightly nudge "<=X" buckets *below* X so they sort before "X-...".
    """
    if s is None:
        return np.inf

    t = str(s).strip().lower().replace(" ", "")

    m = re.match(r"^<=([\d\.]+)(k)?$", t)
    if m:
        val = float(m.group(1))
        if m.group(2) == "k":
            val *= 1000.0
        return val - 1e-6

    m = re.match(r"^>([\d\.]+)(k)?$", t)
    if m:
        val = float(m.group(1))
        if m.group(2) == "k":
            val *= 1000.0
        return val + 1e-6

    m = re.match(r"^([\d\.]+)(k)?-([\d\.]+)(k)?$", t)
    if m:
        a = float(m.group(1)) * (1000.0 if m.group(2) == "k" else 1.0)
        return a

    m = re.search(r"([\d\.]+)", t)
    if m:
        return float(m.group(1))

    return np.inf


_TIME_TO_EVENT_ORDER = [
    "<= 1 hour",
    "2 - 4 hours",
    "4 - 12 hours",
    "12 - 24 hours",
    "1 - 2 days",
    "2 - 3 days",
    "3 - 7 days",
    "7 - 14 days",
    "> 14 days",
    "after_start",
]


def _time_to_event_sort_key(s: str) -> int:
    if s is None:
        return 10_000
    t = str(s).strip()
    try:
        return _TIME_TO_EVENT_ORDER.index(t)
    except ValueError:
        return 9_999


def _infer_categorical_sorter(col_name: str) -> Callable[[str], float | int] | None:
    if col_name == "liquidity_bucket":
        return _parse_liquidity_bucket_lower_bound
    if col_name == "time_to_event_bucket":
        return _time_to_event_sort_key
    return None


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CombinedEVPlotConfig:
    bin_size: float = 1.0
    min_x_pct: float = 0.0
    max_x_pct: float = 10.0
    bucket_edge: str = "left"

    y_lim_pct: float = 10.0
    min_bets_per_bucket: int = 1
    show_ev_dollar_labels: bool = True
    clv_weight: str = "stake"

    bar_color_by: str = "roi"  # "roi" | "profit"
    show_theoretical_ev_line: bool = True


CombinedEvPlotConfig = CombinedEVPlotConfig  # backwards-compatible alias


# -----------------------------------------------------------------------------
# Core bucketing + summarization
# -----------------------------------------------------------------------------
def _bucketize_x(
    sub: pd.DataFrame,
    *,
    x_bucket: str,
    cfg: CombinedEVPlotConfig,
    theo_ev_col: str,
    group_col: str | None,
) -> tuple[pd.Series, str]:
    if x_bucket == "theoretical_ev":
        key = make_ev_bucket(
            sub[theo_ev_col],
            bin_size=cfg.bin_size,
            min_x=cfg.min_x_pct,
            max_x=cfg.max_x_pct,
            edge=cfg.bucket_edge,
        )
        return key, "Theoretical EV Bucket (%)"

    if x_bucket == "group_col":
        if not group_col:
            raise ValueError("group_col must be provided when x_bucket='group_col'.")
        if group_col not in sub.columns:
            raise KeyError(f"Missing column: {group_col}")
        return sub[group_col].astype(str), group_col

    raise ValueError("x_bucket must be 'theoretical_ev' or 'group_col'.")


def _prepare_bucket_summary(
    sub: pd.DataFrame,
    *,
    cfg: CombinedEVPlotConfig,
    x_bucket: str,
    group_col: str | None,
    filter_ev_range_when_theoretical: bool,
    theo_ev_col: str,
    stake_col: str,
    profit_col: str,
    odds_col: str,
    clv_col: str,
) -> tuple[pd.DataFrame, str, str]:
    sub = sub.copy()

    for c in [theo_ev_col, stake_col, profit_col]:
        if c not in sub.columns:
            raise KeyError(f"Missing column: {c}")

    sub[stake_col] = pd.to_numeric(sub[stake_col], errors="coerce")
    sub[profit_col] = pd.to_numeric(sub[profit_col], errors="coerce")
    sub[theo_ev_col] = pd.to_numeric(sub[theo_ev_col], errors="coerce")

    sub = sub.dropna(subset=[stake_col, profit_col])
    sub = sub[sub[stake_col] > 0].copy()
    if sub.empty:
        return pd.DataFrame(), "", ""

    sub["realized_roi_pct"] = (sub[profit_col] / sub[stake_col]) * 100.0
    sub = _add_clv_fields(sub, odds_col=odds_col, clv_col=clv_col)

    bucket_key, x_axis_label = _bucketize_x(
        sub, x_bucket=x_bucket, cfg=cfg, theo_ev_col=theo_ev_col, group_col=group_col
    )
    sub["_bucket_key"] = bucket_key

    if x_bucket == "theoretical_ev" and filter_ev_range_when_theoretical:
        sub = sub.dropna(subset=["_bucket_key"]).copy()
        sub = sub[(sub[theo_ev_col] >= cfg.min_x_pct) & (sub[theo_ev_col] <= cfg.max_x_pct)].copy()

    sub = sub.dropna(subset=["_bucket_key"]).copy()
    if sub.empty:
        return pd.DataFrame(), "", ""

    g = sub.groupby("_bucket_key", dropna=False)

    bucket = pd.DataFrame(
        {
            "bucket_key": g.size().index,
            "avg_ev_pct": g[theo_ev_col].mean(),
            "mean_roi_pct": g["realized_roi_pct"].mean(),
            "n_bets": g.size(),
            "total_stake": g[stake_col].sum(),
            "net_profit": g[profit_col].sum(),
        }
    ).reset_index(drop=True)

    clv_label = "Realized EV% (CLV)"
    if sub["clv_ev_pct"].notna().any():
        if cfg.clv_weight == "equal":
            clv_series = g["clv_ev_pct"].mean()
            clv_label = "Realized EV% (CLV, equal-weighted)"
        else:
            clv_series = (
                sub.dropna(subset=["clv_ev_pct"])
                .groupby("_bucket_key", sort=False)[["clv_ev_pct", stake_col]]
                .apply(lambda x: np.average(x["clv_ev_pct"], weights=x[stake_col]))
            )
            clv_label = "Realized EV% (CLV, stake-weighted)"

        clv_df = clv_series.rename("clv_ev_pct").reset_index()
        clv_df = clv_df.rename(columns={"_bucket_key": "bucket_key"})
        bucket = bucket.merge(clv_df, on="bucket_key", how="left")

    bucket = bucket[bucket["n_bets"] >= cfg.min_bets_per_bucket].copy()
    return bucket, clv_label, x_axis_label


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _add_total_profit_box_aligned_to_legend(
    ax: plt.Axes, *, legend: plt.Legend, total_profit: float
) -> None:
    """
    Place "Total Profit <amount>" directly under the legend, aligned to the
    legend's left edge (in axes coordinates). This cleans up the 'uncentered'
    look that happens when using a fixed bbox_to_anchor.
    """
    # Need a renderer to get legend bounds (in display coords)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Legend bbox in display coords -> axes fraction coords
    leg_bbox_disp = legend.get_window_extent(renderer=renderer)
    inv = ax.transAxes.inverted()
    (x0, y0) = inv.transform((leg_bbox_disp.x0, leg_bbox_disp.y0))
    (x1, _) = inv.transform((leg_bbox_disp.x1, leg_bbox_disp.y0))

    # left-aligned with legend; centered *horizontally* within legend width if desired:
    # we'll keep left-align (cleanest match with legend entries)
    x_left = float(x0)

    # small padding beneath legend
    y_below = float(y0) - 0.02

    prefix, amount, amount_color = _format_total_profit_parts(total_profit)

    left = TextArea(
        f"{prefix} ",
        textprops={"color": "black", "fontsize": 11, "fontweight": "bold"},
    )
    right = TextArea(
        amount,
        textprops={"color": amount_color, "fontsize": 11, "fontweight": "bold"},
    )
    line = HPacker(children=[left, right], align="center", pad=0, sep=0)

    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=line,
        frameon=False,
        bbox_to_anchor=(x_left, y_below),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
        pad=0.0,
    )
    ax.add_artist(anchored)


def _plot_bucket_summary(
    bucket: pd.DataFrame,
    *,
    cfg: CombinedEVPlotConfig,
    title: str,
    clv_label: str,
    x_axis_label: str,
    x_bucket: str,
) -> None:
    if bucket.empty:
        print("No buckets after filtering.")
        return

    # Order buckets
    if x_bucket == "theoretical_ev":
        x_vals = pd.to_numeric(bucket["bucket_key"], errors="coerce")
        bucket = bucket.assign(_x=x_vals).dropna(subset=["_x"]).sort_values("_x").copy()
        x_for_plot = bucket["_x"].to_numpy()
        xtick_labels = [
            f"{left_edge:.1f}–{(left_edge + cfg.bin_size):.1f}%" for left_edge in x_for_plot
        ]
    else:
        bucket = bucket.copy()
        bucket["bucket_key_str"] = bucket["bucket_key"].astype(str)

        sorter = _infer_categorical_sorter(x_axis_label)
        if sorter is not None:
            bucket["_order"] = bucket["bucket_key_str"].apply(sorter)
            bucket = bucket.sort_values(["_order", "bucket_key_str"]).copy()
        else:
            bucket = bucket.sort_values("bucket_key_str").copy()

        x_for_plot = np.arange(len(bucket))
        xtick_labels = bucket["bucket_key_str"].tolist()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6.8), sharex=True, constrained_layout=True
    )

    # top lines
    ax_top.plot(
        x_for_plot,
        bucket["mean_roi_pct"],
        marker="o",
        linewidth=2,
        linestyle=(0, (5, 3)),
        label="Realized ROI%",
    )

    if "clv_ev_pct" in bucket.columns and bucket["clv_ev_pct"].notna().any():
        ax_top.plot(
            x_for_plot,
            bucket["clv_ev_pct"],
            marker="o",
            linewidth=2,
            linestyle="--",
            label=clv_label,
        )

    if cfg.show_theoretical_ev_line and "avg_ev_pct" in bucket.columns:
        if bucket["avg_ev_pct"].notna().any():
            ax_top.plot(
                x_for_plot,
                bucket["avg_ev_pct"],
                marker="o",
                linewidth=2,
                linestyle=":",
                label="Theoretical EV% (mean)",
            )

    ax_top.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax_top.set_ylabel("Realized EV %")
    ax_top.set_ylim(-cfg.y_lim_pct, cfg.y_lim_pct)
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_top.legend(frameon=False, loc="best")

    # bar coloring
    if cfg.bar_color_by == "profit":
        bar_colors = np.where(bucket["net_profit"] >= 0, GREEN, RED)
    else:
        bar_colors = np.where(bucket["mean_roi_pct"] >= 0, GREEN, RED)

    ax_bot.bar(
        x_for_plot,
        bucket["n_bets"],
        width=0.9 if x_bucket != "theoretical_ev" else cfg.bin_size * 0.9,
        color=bar_colors,
        alpha=0.85,
        edgecolor="none",
    )
    ax_bot.set_ylim(0, float(bucket["n_bets"].max()) * 1.22)

    # profit labels per bucket
    if cfg.show_ev_dollar_labels:
        for x, n, p in zip(x_for_plot, bucket["n_bets"], bucket["net_profit"], strict=False):
            label = _format_dollars_signed(float(p))
            color = GREEN if float(p) >= 0 else RED
            ax_bot.annotate(
                label,
                (float(x), float(n)),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                clip_on=False,
            )

    # Legend in upper-right
    legend = ax_bot.legend(
        handles=[
            Patch(facecolor=GREEN, alpha=0.85, label="Positive (by setting)"),
            Patch(facecolor=RED, alpha=0.85, label="Negative (by setting)"),
        ],
        frameon=False,
        loc="upper right",
    )

    # Total profit aligned under legend
    total_profit = float(pd.to_numeric(bucket["net_profit"], errors="coerce").fillna(0.0).sum())
    _add_total_profit_box_aligned_to_legend(ax_bot, legend=legend, total_profit=total_profit)

    ax_bot.set_ylabel("# Bets")
    ax_bot.set_xlabel(x_axis_label)
    ax_bot.set_xticks(x_for_plot)
    ax_bot.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax_bot.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.show()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def combined_ev_summary_plot(
    df: pd.DataFrame,
    filter_name: str | None,
    *,
    cfg: CombinedEVPlotConfig | None = None,
    x_bucket: str = "theoretical_ev",  # "theoretical_ev" | "group_col"
    group_col: str | None = None,  # required if x_bucket="group_col"
    filter_col: str = "saved_filter_names",
    theo_ev_col: str = "percentage",
    stake_col: str = "stake",
    profit_col: str = "bet_profit",
    odds_col: str = "odds",
    clv_col: str = "clv",
    only_non_null_group: bool = True,  # applies if x_bucket="group_col"
) -> None:
    cfg = cfg or CombinedEvPlotConfig()

    if filter_name is not None:
        if filter_col not in df.columns:
            raise KeyError(f"Missing column: {filter_col}")
        sub = df.loc[df[filter_col] == filter_name].copy()
        if sub.empty:
            print(f"No data for filter: {filter_name}")
            return
        base_title = filter_name
    else:
        sub = df.copy()
        base_title = "ALL FILTERS"

    if x_bucket == "group_col":
        if not group_col:
            raise ValueError("group_col must be provided when x_bucket='group_col'.")
        if group_col not in sub.columns:
            raise KeyError(f"Missing column: {group_col}")
        if only_non_null_group:
            sub = sub[sub[group_col].notna()].copy()

    bucket, clv_label, x_axis_label = _prepare_bucket_summary(
        sub,
        cfg=cfg,
        x_bucket=x_bucket,
        group_col=group_col,
        filter_ev_range_when_theoretical=True,
        theo_ev_col=theo_ev_col,
        stake_col=stake_col,
        profit_col=profit_col,
        odds_col=odds_col,
        clv_col=clv_col,
    )
    if bucket.empty:
        print("No buckets after filtering.")
        return

    title = f"EV & ROI Summary — {base_title}"
    _plot_bucket_summary(
        bucket,
        cfg=cfg,
        title=title,
        clv_label=clv_label,
        x_axis_label=x_axis_label,
        x_bucket=x_bucket,
    )


def combined_ev_summary_plot_faceted_by_group(
    df: pd.DataFrame,
    filter_name: str | None,
    *,
    facet_col: str,
    cfg: CombinedEVPlotConfig | None = None,
    x_bucket: str = "theoretical_ev",
    group_col_for_x: str | None = None,
    filter_col: str = "saved_filter_names",
    theo_ev_col: str = "percentage",
    stake_col: str = "stake",
    profit_col: str = "bet_profit",
    odds_col: str = "odds",
    clv_col: str = "clv",
    only_non_null_facet: bool = True,
) -> None:
    cfg = cfg or CombinedEvPlotConfig()

    if facet_col not in df.columns:
        raise KeyError(f"Missing facet column: {facet_col}")

    if filter_name is not None:
        if filter_col not in df.columns:
            raise KeyError(f"Missing column: {filter_col}")
        base = df.loc[df[filter_col] == filter_name].copy()
        if base.empty:
            print(f"No data for filter: {filter_name}")
            return
        base_title = filter_name
    else:
        base = df.copy()
        base_title = "ALL FILTERS"

    if only_non_null_facet:
        base = base[base[facet_col].notna()].copy()

    facet_vals = base[facet_col].dropna().astype(str).unique().tolist()
    if not facet_vals:
        print(f"No non-null values in facet_col: {facet_col}")
        return

    facet_sorter = _infer_categorical_sorter(facet_col)
    vals = sorted(facet_vals, key=facet_sorter) if facet_sorter is not None else sorted(facet_vals)

    for v in vals:
        sub = base[base[facet_col].astype(str) == v].copy()
        bucket, clv_label, x_axis_label = _prepare_bucket_summary(
            sub,
            cfg=cfg,
            x_bucket=x_bucket,
            group_col=group_col_for_x if x_bucket == "group_col" else None,
            filter_ev_range_when_theoretical=True,
            theo_ev_col=theo_ev_col,
            stake_col=stake_col,
            profit_col=profit_col,
            odds_col=odds_col,
            clv_col=clv_col,
        )
        if bucket.empty:
            print(f"[{facet_col}={v}] No buckets after filtering.")
            continue

        title = f"EV & ROI Summary — {base_title} — {facet_col}: {v}"
        _plot_bucket_summary(
            bucket,
            cfg=cfg,
            title=title,
            clv_label=clv_label,
            x_axis_label=x_axis_label,
            x_bucket=x_bucket,
        )
