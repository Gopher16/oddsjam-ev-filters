"""
discovery_config.py
===============================================================================
Purpose
-------
Validate predeclared discovery and temporal evidence policies.

What this module provides
-------------------------
Serializable policy, explicit inference thresholds, and temporal windows.

Design Notes
------------
Thresholds are operational guardrails, not proof of statistical power.
===============================================================================
"""

import math
from dataclasses import dataclass

import pandas as pd

from oddsjam_ev.analysis.opportunity_atlas import DIMENSIONS


@dataclass(frozen=True)
class DiscoveryConfig:
    """Research policy.

    Attributes
    ----------
    min_event_groups : int
        Operational lower bound for conditional group inference.
    min_eligible_stake_usd : float, optional
        Required grouped-cohort stake floor; None blocks formal testing.
    confidence_level, alpha : float
        Bootstrap interval coverage and familywise testing level.
    bootstrap_resamples, seed : int
        Reproducible cluster resampling parameters.
    execution_modes : tuple[str, ...]
        Separate funded-assumption, paper, and unknown populations.
    views : tuple[tuple[str, ...], ...]
        Predeclared categorical predicates, at most two dimensions each.
    temporal_windows : tuple[dict, ...]
        Explicit UTC development/selection/test windows for rolling origins.
    embargo_days : float, optional
        Justified gap policy; None blocks chronological validation.
    contract_version : int
        Report contract version.
    """

    min_event_groups: int = 100
    min_eligible_stake_usd: float | None = None
    confidence_level: float = 0.95
    bootstrap_resamples: int = 5000
    seed: int = 20261022
    alpha: float = 0.05
    execution_modes: tuple = ("real_money", "paper", "unknown")
    views: tuple = tuple((d,) for d in DIMENSIONS) + (
        ("sportsbook", "market_family"),
        ("league", "market_family"),
        ("odds_band", "advertised_ev_band"),
    )
    temporal_windows: tuple = ()
    embargo_days: float | None = None
    contract_version: int = 1

    def __post_init__(self):
        if self.contract_version != 1:
            raise ValueError("Unsupported discovery contract")
        for name, minimum in [("min_event_groups", 2), ("bootstrap_resamples", 100), ("seed", 0)]:
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ["confidence_level", "alpha"]:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 < value < 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in ["min_eligible_stake_usd", "embargo_days"]:
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative or null")
        modes = tuple(self.execution_modes)
        if (
            not modes
            or len(set(modes)) != len(modes)
            or set(modes) - {"real_money", "paper", "unknown"}
        ):
            raise ValueError("Invalid execution modes")
        views = tuple(tuple(v) for v in self.views)
        if (
            not views
            or len({frozenset(v) for v in views}) != len(views)
            or any(
                not 1 <= len(v) <= 2 or len(set(v)) != len(v) or set(v) - set(DIMENSIONS)
                for v in views
            )
        ):
            raise ValueError("Invalid or duplicate discovery views")
        object.__setattr__(self, "execution_modes", modes)
        object.__setattr__(self, "views", views)
        windows = tuple(dict(x) for x in self.temporal_windows)
        ids = []
        for window in windows:
            if set(window) != {
                "fold_id",
                "development_start",
                "selection_start",
                "test_start",
                "test_end",
            }:
                raise ValueError("Temporal window fields are invalid")
            times = [
                pd.Timestamp(window[k])
                for k in ["development_start", "selection_start", "test_start", "test_end"]
            ]
            if (
                not isinstance(window["fold_id"], str)
                or not window["fold_id"]
                or any(pd.isna(t) or t.tzinfo is None for t in times)
                or any(a >= b for a, b in zip(times, times[1:], strict=False))
            ):
                raise ValueError("Temporal windows require increasing timezone-aware boundaries")
            ids.append(window["fold_id"])
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate temporal fold IDs")
        ordered = sorted(windows, key=lambda w: pd.Timestamp(w["test_start"]))
        for previous, following in zip(ordered, ordered[1:], strict=False):
            if pd.Timestamp(previous["test_end"]) > pd.Timestamp(following["test_start"]):
                raise ValueError("Test windows must not overlap across rolling origins")
            if pd.Timestamp(previous["development_start"]) != pd.Timestamp(
                following["development_start"]
            ):
                raise ValueError("Expanding windows require a fixed development start")
        object.__setattr__(self, "temporal_windows", tuple(ordered))
