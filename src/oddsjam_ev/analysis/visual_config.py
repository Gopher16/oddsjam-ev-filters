"""
visual_config.py
===============================================================================
Purpose
-------
Declare visual buckets, source semantics and bounded descriptive publication policy.

What this module provides
-------------------------
Validated opt-in chronology/liquidity/closing-reference evidence and display contracts.

Design Notes
------------
Defaults preserve unknown source semantics. Chart readiness is not strategy validation.
===============================================================================
"""

import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class VisualConfig:
    """Visual publication settings and explicit source-evidence assertions.

    Attributes
    ----------
    ev_edges, liquidity_edges, time_edges_hours : tuple
        Finite increasing left-closed display edges, independent of exact predicates.
    naive_timezone : str, optional
        Explicit supported source timezone, never inferred from legacy preprocessing.
    placement_time_verified, event_time_verified : bool
        Source-field meanings verified by the operator/reference below.
    chronology_evidence, liquidity_evidence, closing_evidence : str, optional
        Required provenance for enabling the corresponding optional derived series.
    liquidity_usd_verified, closing_fair_american_verified : bool
        Explicit supported source meaning; closing odds must be fair-reference odds.
    closing_sources : tuple
        Exact source values permitted for the verified closing-price transform.
    min_performance_wagers : int
        Display flag only; rows are preserved and activity periods are not suppressed.
    fixed_unit_usd : float
        Notional amount for a fixed-unit descriptive comparison.
    max_scope_links, max_rows, max_calendar_days : int
        Fail-closed resource bounds; never truncate outputs.
    contract_version : int
        Visual-data contract version.
    """

    ev_edges: tuple = (0, 2.5, 3, 3.5, 4, 5, 7.5, 10, 20)
    liquidity_edges: tuple = (0, 500, 1000, 2000, 5000, 10000)
    time_edges_hours: tuple = (0, 1, 2, 4, 12, 24, 48, 72, 168, 336)
    naive_timezone: str | None = None
    placement_time_verified: bool = False
    event_time_verified: bool = False
    chronology_evidence: str | None = None
    liquidity_usd_verified: bool = False
    liquidity_evidence: str | None = None
    closing_fair_american_verified: bool = False
    closing_sources: tuple = ()
    closing_evidence: str | None = None
    min_performance_wagers: int = 100
    fixed_unit_usd: float = 1.0
    max_scope_links: int = 1000000
    max_rows: int = 2000000
    max_calendar_days: int = 7305
    contract_version: int = 1

    def __post_init__(self):
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("Unsupported visual contract")
        for name in ("ev_edges", "liquidity_edges", "time_edges_hours"):
            values = tuple(getattr(self, name))
            if (
                len(values) < 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or any(a >= b for a, b in zip(values, values[1:], strict=False))
            ):
                raise ValueError("Finite increasing visual edges required")
            object.__setattr__(self, name, values)
        if self.liquidity_edges[0] < 0 or self.time_edges_hours[0] != 0:
            raise ValueError("Nonnegative liquidity and zero-start time edges required")
        for name in (
            "placement_time_verified",
            "event_time_verified",
            "liquidity_usd_verified",
            "closing_fair_american_verified",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError("Explicit boolean visual-source assertions required")
        for name in ("min_performance_wagers", "max_scope_links", "max_rows", "max_calendar_days"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer visual limits required")
        if (
            type(self.fixed_unit_usd) not in (int, float)
            or not math.isfinite(self.fixed_unit_usd)
            or self.fixed_unit_usd <= 0
        ):
            raise ValueError("Positive finite notional unit required")
        if self.naive_timezone is not None:
            ZoneInfo(self.naive_timezone)
        for enabled, reference in (
            (
                self.placement_time_verified
                or self.event_time_verified
                or self.naive_timezone is not None,
                self.chronology_evidence,
            ),
            (self.liquidity_usd_verified, self.liquidity_evidence),
            (self.closing_fair_american_verified, self.closing_evidence),
        ):
            if enabled and (not isinstance(reference, str) or not reference.strip()):
                raise ValueError("Verified visual derivations require explicit evidence")
        if not isinstance(self.closing_sources, (tuple, list)) or not all(
            isinstance(v, str) and v.strip() for v in self.closing_sources
        ):
            raise ValueError("Explicit closing source list required")
        object.__setattr__(self, "closing_sources", tuple(self.closing_sources))
        if self.closing_fair_american_verified and not self.closing_sources:
            raise ValueError("Verified closing transform requires permitted source references")
