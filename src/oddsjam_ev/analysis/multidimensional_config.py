"""
multidimensional_config.py
===============================================================================
Purpose
-------
Declare bounded, outcome-independent within-filter descriptive search families.

What this module provides
-------------------------
Canonical predicates, exact EV endpoints, dimension allowlist and hard work budgets.

Design Notes
------------
Resource limits are not statistical support gates. Formal inference and ranking
remain blocked pending reviewed support, multiplicity and chronology policies.
===============================================================================
"""

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from oddsjam_ev.analysis.continuous_ev import ev_threshold
from oddsjam_ev.analysis.opportunity_atlas import DIMENSIONS

EV_DIMENSION = "advertised_ev_percent"


def _endpoint(value):
    ev_threshold(pd.Series([], dtype=object), ">", value)
    number = Decimal(value)
    if number == 0:
        return "0"
    sign, digits, exponent = number.as_tuple()
    while digits and digits[-1] == 0:
        digits = digits[:-1]
        exponent += 1
    return str(Decimal((sign, digits, exponent)))


@dataclass(frozen=True)
class MultidimensionalConfig:
    """Predeclared family and operational limits, not an optimization policy.

    Attributes
    ----------
    views : tuple
        Explicit dimension combinations, canonicalized independent of order.
    ev_thresholds : tuple
        Operator/endpoint/percentage-point unit definitions; no learned cutpoints.
    max_depth, max_parents, max_candidates, max_memberships, max_scan_rows : int
        Fail-closed resource budgets; no partial family is published on overflow.
    contract_version : int
        Descriptive candidate/comparison contract version.
    """

    views: tuple = (
        ("sportsbook", "market_family"),
        ("league", "market_family", EV_DIMENSION),
        ("sportsbook", "sport", "market_family", EV_DIMENSION),
    )
    ev_thresholds: tuple = ({"operator": ">", "endpoint": "3", "units": "percentage_points"},)
    max_depth: int = 4
    max_parents: int = 200
    max_candidates: int = 5000
    max_memberships: int = 1000000
    max_scan_rows: int = 50000000
    contract_version: int = 1

    def __post_init__(self):
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("Unsupported multidimensional contract")
        for key, maximum in [
            ("max_depth", 4),
            ("max_parents", 1000),
            ("max_candidates", 20000),
            ("max_memberships", 5000000),
            ("max_scan_rows", 200000000),
        ]:
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Invalid multidimensional work budget")
        if not isinstance(self.views, (list, tuple)) or not self.views:
            raise ValueError("Explicit predicate views required")
        views = []
        for view in self.views:
            if (
                not isinstance(view, (list, tuple))
                or not 1 <= len(view) <= self.max_depth
                or len(set(view)) != len(view)
                or set(view) - set((*DIMENSIONS, EV_DIMENSION))
            ):
                raise ValueError("Unsupported or over-depth predicate view")
            views.append(tuple(sorted(view)))
        if len(set(views)) != len(views):
            raise ValueError("Duplicate predicate view")
        thresholds = []
        if not isinstance(self.ev_thresholds, (list, tuple)) or len(self.ev_thresholds) > 32:
            raise ValueError("At most 32 declared EV thresholds are supported")
        for item in self.ev_thresholds:
            if (
                not isinstance(item, dict)
                or set(item) != {"operator", "endpoint", "units"}
                or item["units"] != "percentage_points"
            ):
                raise ValueError("Explicit EV predicate units required")
            ev_threshold(pd.Series([], dtype=object), item["operator"], item["endpoint"])
            thresholds.append(
                {
                    "operator": item["operator"],
                    "endpoint": _endpoint(item["endpoint"]),
                    "units": "percentage_points",
                }
            )
        if any(EV_DIMENSION in view for view in views) and not thresholds:
            raise ValueError("EV views require declared thresholds")
        if len({(t["operator"], t["endpoint"]) for t in thresholds}) != len(thresholds):
            raise ValueError("Duplicate EV threshold")
        object.__setattr__(self, "views", tuple(sorted(views)))
        object.__setattr__(
            self,
            "ev_thresholds",
            tuple(sorted(thresholds, key=lambda t: (t["operator"], t["endpoint"]))),
        )
