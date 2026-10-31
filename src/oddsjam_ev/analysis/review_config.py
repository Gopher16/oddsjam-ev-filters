"""
review_config.py
===============================================================================
Purpose
-------
Freeze bounded global catalogue and deterministic research presentation policy.

What this module provides
-------------------------
Outcome-independent views, declared display ranges, overlap caps and resource bounds.

Design Notes
------------
No inferential thresholds are changed. Global candidates are not formally tested.
===============================================================================
"""

from dataclasses import dataclass

from oddsjam_ev.analysis.visual_config import VisualConfig


@dataclass(frozen=True)
class ReviewConfig:
    """Versioned catalogue and presentation settings, independent of inferential gates.

    Attributes
    ----------
    global_views : tuple
        Declared single/pair feature searches across each recorded population.
    ev_edges, odds_edges : tuple
        Finite display cutpoints, including explicit outer bins in the catalogue.
    headline_limit : int
        Maximum distinct ideas per actionable queue, never a quota.
    overlap_limit : float
        Wager or event Jaccard threshold for headline diversity, not test suppression.
    max_candidates, max_memberships, max_pairs : int
        Abort budgets; a partial family must never become a publication.
    contract_version : int
        Presentation contract; no prospective validation is supported here.
    """

    global_views: tuple = (
        ("market_family", "sportsbook"),
        ("league", "market_family"),
        ("advertised_ev_band", "odds_band"),
    )
    ev_edges: tuple = (0, 2.5, 3, 5, 10, 20)
    odds_edges: tuple = (-250, -150, -100, 100, 150, 250)
    headline_limit: int = 4
    overlap_limit: float = 0.8
    max_candidates: int = 3000
    max_memberships: int = 3000000
    max_pairs: int = 2000000
    contract_version: int = 1

    def __post_init__(self):
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("Unsupported review contract")
        supported = {
            "sportsbook",
            "sport",
            "league",
            "market_family",
            "odds_band",
            "advertised_ev_band",
        }
        views = tuple(tuple(v) for v in self.global_views)
        if (
            not views
            or len(set(views)) != len(views)
            or any(
                not 1 <= len(v) <= 2
                or len(set(v)) != len(v)
                or set(v) - supported
                or tuple(sorted(v)) != v
                for v in views
            )
        ):
            raise ValueError("Unique sorted bounded supported review views required")
        object.__setattr__(self, "global_views", views)
        edges = VisualConfig(ev_edges=self.ev_edges, liquidity_edges=(0, 1))
        object.__setattr__(self, "ev_edges", edges.ev_edges)
        edges = VisualConfig(ev_edges=self.odds_edges, liquidity_edges=(0, 1))
        object.__setattr__(self, "odds_edges", edges.ev_edges)
        for name in ("headline_limit", "max_candidates", "max_memberships", "max_pairs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive review limits required")
        if type(self.overlap_limit) not in (int, float) or not 0 < self.overlap_limit <= 1:
            raise ValueError("Finite overlap fraction required")
