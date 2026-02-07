from __future__ import annotations

# EV plots
from .ev import (
    CombinedEVPlotConfig,
    CombinedEvPlotConfig,  # backwards-compatible alias
    combined_ev_summary_plot,
)

# “Filters” plots (ROI scatter + profit-by-group)
from .filters import (
    ProfitByGroupPlotConfig,
    ProfitBySportPlotConfig,
    RoiScatterConfig,
    plot_filter_roi_scatter,
    plot_profit_by_group_for_filters,
    plot_profit_by_sport_for_filters,
)

# Time series plots
from .timeseries import (
    AvgBetsByTimeBucketPlotConfig,
    # Bet activity plots (public names)
    BetCountOverTimePlotConfig,
    BetDistributionOverTimePlotConfig,
    CumProfitByGroupPlotConfig,
    CumProfitPlotConfig,
    plot_avg_bets_by_time_bucket,
    plot_bet_count_over_time,
    plot_bet_distribution_over_time,
    plot_cum_profit_by_group,
    plot_cum_profit_top_filters,
    plot_cumulative_profit_over_time,
    top_n_filters_by_metric,
)

__all__ = [
    # ROI scatter
    "RoiScatterConfig",
    "plot_filter_roi_scatter",
    # EV bucket chart
    "CombinedEVPlotConfig",
    "CombinedEvPlotConfig",
    "combined_ev_summary_plot",
    # time series
    "top_n_filters_by_metric",
    "CumProfitPlotConfig",
    "plot_cumulative_profit_over_time",
    "plot_cum_profit_top_filters",
    "CumProfitByGroupPlotConfig",
    "plot_cum_profit_by_group",
    # bet activity
    "BetCountOverTimePlotConfig",
    "plot_bet_count_over_time",
    "BetDistributionOverTimePlotConfig",
    "plot_bet_distribution_over_time",
    "AvgBetsByTimeBucketPlotConfig",
    "plot_avg_bets_by_time_bucket",
    # profit by group / sport
    "ProfitByGroupPlotConfig",
    "plot_profit_by_group_for_filters",
    "ProfitBySportPlotConfig",
    "plot_profit_by_sport_for_filters",
]
