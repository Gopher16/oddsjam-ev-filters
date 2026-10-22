"""
atlas_config.py
===============================================================================
Purpose
-------
Validate descriptive atlas policies without introducing strategy thresholds.

What this module provides
-------------------------
Explicit numeric bands and exact market-label mappings.

Design Notes
------------
Band boundaries are display choices, not evidence of a betting edge.
===============================================================================
"""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AtlasConfig:
    """Descriptive policies.

    Attributes
    ----------
    odds_edges, ev_edges : tuple[float, ...]
        Finite, strictly increasing display boundaries; intervals are left closed.
    market_families : dict[str, str]
        Exact source label mapping. Missing mappings are never guessed.
    contract_version : int
        Atlas schema version.
    """

    odds_edges: tuple = (-200, -150, -110, 100, 150, 200)
    ev_edges: tuple = (0, 1, 2.5, 5, 10, 20)
    market_families: dict = field(default_factory=dict)
    contract_version: int = 1

    def __post_init__(self):
        if self.contract_version != 1:
            raise ValueError("Unsupported atlas contract")
        for edges in (self.odds_edges, self.ev_edges):
            if (
                not edges
                or any(not math.isfinite(x) for x in edges)
                or any(a >= b for a, b in zip(edges, edges[1:], strict=False))
            ):
                raise ValueError("Band edges must be finite and strictly increasing")
        if any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in self.market_families.items()
        ):
            raise ValueError("Market mappings must contain nonempty strings")
