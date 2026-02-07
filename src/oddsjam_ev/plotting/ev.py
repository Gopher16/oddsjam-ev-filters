from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from oddsjam_ev.metrics.odds import clv_ev_pct, ensure_odds_multiplier, ensure_prob_clv

GREEN = "#2E8B57"
RED = "#D9534F"


def make_ev_bucket(
    series: pd.Series,
    *,
    bin_size: float,
    min_x: float,
    max_x: float,
    edge: str = "right",
) -> pd.Series:
    """
    Bin EV% into labeled buckets.

    edge='left'  -> left-closed [a,b)  (2.50 -> 2.5–3.0)
    edge='right' -> right-closed (a,b] (2.50 -> 2.0–2.5)
    """
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


@dataclass(frozen=True)
class CombinedEVPlotConfig:
    bin_size: float = 1.0
    min_x_pct: float = 0.0
    max_x_pct: float = 10.0
    y_lim_pct: float = 10.0
    min_bets_per_bucket: int = 1
    show_ev_dollar_labels: bool = True
    clv_weight: str = "stake"  # "stake" or "equal"
    bucket_edge: str = "left"  # "left" or "right"


# Backwards-compatible alias (older versions used MixedCase Ev)
CombinedEvPlotConfig = CombinedEVPlotConfig


def combined_ev_summary_plot(
    df: pd.DataFrame,
    filter_name: str,
    *,
    cfg: CombinedEVPlotConfig | None = None,
    filter_col: str = "saved_filter_names",
    theo_ev_col: str = "percentage",
    stake_col: str = "stake",
    profit_col: str = "bet_profit",
    odds_col: str = "odds",
    clv_col: str = "clv",
) -> None:
    """
    Two-panel summary by theoretical EV buckets for one filter.

    Top:
      - Realized ROI% (mean per bucket)
      - Realized EV% from CLV (optional; stake- or equal-weighted)
      - Theoretical EV% (mean per bucket)

    Bottom:
      - # Bets per bucket (colored by mean ROI sign)
      - Optional profit dollars label per bucket
    """
    cfg = cfg or CombinedEVPlotConfig()
    if filter_col not in df.columns:
        raise KeyError(f"Missing column: {filter_col}")

    sub = df.loc[df[filter_col] == filter_name].copy()
    if sub.empty:
        print(f"No data for filter: {filter_name}")
        return

    for c in [theo_ev_col, stake_col, profit_col]:
        if c not in sub.columns:
            raise KeyError(f"Missing column: {c}")

    sub[stake_col] = pd.to_numeric(sub[stake_col], errors="coerce")
    sub[profit_col] = pd.to_numeric(sub[profit_col], errors="coerce")
    sub[theo_ev_col] = pd.to_numeric(sub[theo_ev_col], errors="coerce")

    sub = sub.dropna(subset=[theo_ev_col, stake_col, profit_col])
    sub = sub[sub[stake_col] > 0].copy()

    # Optional CLV helpers
    sub = ensure_prob_clv(sub, clv_col=clv_col, out_col="prob_clv")
    sub = ensure_odds_multiplier(sub, odds_col=odds_col, out_col="odds_multiplier")

    sub["realized_roi_pct"] = (sub[profit_col] / sub[stake_col]) * 100.0
    if sub["prob_clv"].notna().any() and sub["odds_multiplier"].notna().any():
        sub["clv_ev_pct"] = clv_ev_pct(sub["prob_clv"], sub["odds_multiplier"])
    else:
        sub["clv_ev_pct"] = np.nan

    sub["ev_bucket"] = make_ev_bucket(
        sub[theo_ev_col],
        bin_size=cfg.bin_size,
        min_x=cfg.min_x_pct,
        max_x=cfg.max_x_pct,
        edge=cfg.bucket_edge,
    )

    g = sub.groupby("ev_bucket", dropna=False)

    bucket = (
        pd.DataFrame(
            {
                "avg_ev_pct": g[theo_ev_col].mean(),
                "mean_roi_pct": g["realized_roi_pct"].mean(),
                "n_bets": g.size(),
                "total_stake": g[stake_col].sum(),
                "net_profit": g[profit_col].sum(),
            }
        )
        .reset_index()
        .dropna(subset=["ev_bucket"])
    )

    # Optional CLV EV% line (weighting)
    clv_label = "Realized EV% (CLV)"
    if sub["clv_ev_pct"].notna().any():
        if cfg.clv_weight == "equal":
            clv_series = g["clv_ev_pct"].mean()
            clv_label = "Realized EV% (CLV, equal-weighted)"
        else:
            # Avoid pandas FutureWarning by selecting columns explicitly (no grouping cols)
            clv_series = (
                sub.dropna(subset=["clv_ev_pct"])
                .groupby("ev_bucket", sort=False)[["clv_ev_pct", stake_col]]
                .apply(lambda x: np.average(x["clv_ev_pct"], weights=x[stake_col]))
            )
            clv_label = "Realized EV% (CLV, stake-weighted)"

        bucket = bucket.merge(
            clv_series.rename("clv_ev_pct").reset_index(),
            on="ev_bucket",
            how="left",
        )

    bucket = bucket[(bucket["ev_bucket"] >= cfg.min_x_pct) & (bucket["ev_bucket"] <= cfg.max_x_pct)]
    bucket = bucket[bucket["n_bets"] >= cfg.min_bets_per_bucket]
    if bucket.empty:
        print("No buckets after filtering.")
        return

    centers = bucket["ev_bucket"].to_numpy()
    xtick_labels = [f"{left_edge:.1f}–{(left_edge + cfg.bin_size):.1f}%" for left_edge in centers]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6.8), sharex=True, constrained_layout=True
    )

    ax_top.plot(
        bucket["ev_bucket"],
        bucket["mean_roi_pct"],
        marker="o",
        linewidth=2,
        linestyle=(0, (5, 3)),
        label="Realized ROI%",
    )

    if "clv_ev_pct" in bucket.columns and bucket["clv_ev_pct"].notna().any():
        ax_top.plot(
            bucket["ev_bucket"],
            bucket["clv_ev_pct"],
            marker="o",
            linewidth=2,
            linestyle="--",
            label=clv_label,
        )

    ax_top.plot(
        bucket["ev_bucket"],
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

    bar_colors = np.where(bucket["mean_roi_pct"] >= 0, GREEN, RED)
    ax_bot.bar(
        bucket["ev_bucket"],
        bucket["n_bets"],
        width=cfg.bin_size * 0.9,
        color=bar_colors,
        alpha=0.85,
        edgecolor="none",
    )

    ax_bot.set_ylim(0, float(bucket["n_bets"].max()) * 1.22)

    if cfg.show_ev_dollar_labels:
        for x, n, p, mean_roi in zip(
            bucket["ev_bucket"],
            bucket["n_bets"],
            bucket["net_profit"],
            bucket["mean_roi_pct"],
            strict=False,
        ):
            color = GREEN if mean_roi >= 0 else RED
            label = f"${p:,.0f}" if mean_roi >= 0 else f"(${abs(p):,.0f})"
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

    ax_bot.legend(
        handles=[
            Patch(facecolor=GREEN, alpha=0.85, label="Bucket Mean ROI% ≥ 0"),
            Patch(facecolor=RED, alpha=0.85, label="Bucket Mean ROI% < 0"),
        ],
        frameon=False,
        loc="best",
    )

    ax_bot.set_ylabel("# Bets")
    ax_bot.set_xlabel("Theoretical EV Bucket (%)")
    ax_bot.set_xticks(centers)
    ax_bot.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax_bot.grid(axis="y", linestyle="--", alpha=0.25)
    ax_bot.set_xlim(cfg.min_x_pct, cfg.max_x_pct)

    fig.suptitle(
        f"EV & ROI by Theoretical EV Bucket — {filter_name}", fontsize=13, fontweight="bold"
    )
    plt.show()
