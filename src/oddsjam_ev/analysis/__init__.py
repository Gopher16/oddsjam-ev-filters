"""
analysis package
===============================================================================

Reusable analysis-layer utilities for filter evaluation and exchange-specific
capacity / liquidity diagnostics.

Modules
-------
- filter_eval
    Canonical filter-level summary tables
- exchange_eval
    Exchange-only liquidity and capacity metrics
===============================================================================
"""

from oddsjam_ev.analysis.exchange_eval import build_exchange_capacity_table
from oddsjam_ev.analysis.filter_eval import FilterEvalConfig, build_filter_evaluation_table

__all__ = [
    "FilterEvalConfig",
    "build_filter_evaluation_table",
    "build_exchange_capacity_table",
]
