"""
canonical_config.py
===============================================================================
Purpose
-------
Validate explicit canonical economic policy without changing legacy defaults.

What this module provides
-------------------------
A small immutable configuration contract and exact product classification.

Design Notes
------------
Venue mappings are declared policy, not proof of fees, funding, or execution.
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalConfig:
    """User-confirmed currency and conservative product policy.

    Attributes
    ----------
    products : dict
        Disjoint exact venue lists for fixed_odds, exchange, and pickem.
    pickem_filter_ids : tuple
        Registry IDs that provide negative standalone-wager eligibility evidence.
    profit_tolerance_usd : float
        Explicit arithmetic consistency tolerance; source amounts are not rounded.
    currency_evidence : str
        Provenance for USD, independently of unverified fee inclusion.
    """

    products: dict[str, list[str]]
    pickem_filter_ids: tuple[str, ...] = ()
    profit_tolerance_usd: float = 0.01
    currency_evidence: str = "user_confirmed_2026-10-22"
    schema_version: int = 1
    currency: str = "USD"
    fee_inclusion: str = "unverified"
    refunds_in_roi: bool = False

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.currency != "USD"
            or self.fee_inclusion != "unverified"
            or self.refunds_in_roi is not False
        ):
            raise ValueError("Unsupported canonical schema or economic policy")
        if not isinstance(self.currency_evidence, str) or not self.currency_evidence.strip():
            raise ValueError("Currency provenance is required")
        tolerance = self.profit_tolerance_usd
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("profit_tolerance_usd must be finite and nonnegative")
        if not isinstance(self.products, dict) or set(self.products) != {
            "fixed_odds",
            "exchange",
            "pickem",
        }:
            raise ValueError("Explicit fixed_odds, exchange, and pickem mappings are required")
        seen = set()
        for names in self.products.values():
            if not isinstance(names, list):
                raise ValueError("Product venue mappings must be lists")
            for name in names:
                if (
                    not isinstance(name, str)
                    or not name.strip()
                    or name != name.strip()
                    or name in seen
                ):
                    raise ValueError("Product venues must be nonempty, exact, and disjoint")
                seen.add(name)
        if not isinstance(self.pickem_filter_ids, (tuple, list)) or any(
            not isinstance(v, str) or not v for v in self.pickem_filter_ids
        ):
            raise ValueError("pickem_filter_ids must contain registry IDs")

    def product(self, venue: str) -> str:
        """Return an exact declared product or unknown.

        Parameters
        ----------
        venue : str
            Source venue label.

        Returns
        -------
        str
            fixed_odds, exchange, pickem, or unknown.
        """
        return next((kind for kind, names in self.products.items() if venue in names), "unknown")
