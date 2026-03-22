"""
analysis package
===============================================================================

Reusable analysis-layer utilities for filter evaluation, scorecards, and
exchange-specific capacity / liquidity diagnostics.

Modules
-------
- filter_eval
    Canonical filter-level summary tables
- exchange_eval
    Exchange-only bucket-based liquidity and capacity metrics
- scorecards
    Merged prod-aware decision scorecards
===============================================================================
"""

from oddsjam_ev.analysis.exchange_eval import build_exchange_capacity_table
from oddsjam_ev.analysis.filter_eval import FilterEvalConfig, build_filter_evaluation_table
from oddsjam_ev.analysis.scorecards import build_exchange_scorecard

__all__ = [
    "FilterEvalConfig",
    "build_filter_evaluation_table",
    "build_exchange_capacity_table",
    "build_exchange_scorecard",
]
