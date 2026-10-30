"""
prioritization_config.py
===============================================================================
Purpose
-------
Declare the user-approved initial exploratory statistical screening policy.

What this module provides
-------------------------
Scale-independent arm gates, joint event bootstrap and complete-family correction.

Design Notes
------------
Operational support gates are not power guarantees. No validated/production label
is supported while chronology, event independence and fillability are unverified.
===============================================================================
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrioritizationConfig:
    """Approved exploratory gates and bounded presentation/computation policy.

    Attributes
    ----------
    min_eligible_wagers, min_event_groups : int
        Applied separately to subgroup and complement.
    min_event_coverage : float
        Complete event coverage required in this contract.
    max_wager_stake_share, max_event_stake_share : float
        Arm-specific exposure-concentration caps.
    min_eligible_stake_usd : float
        Explicit zero additional dollar floor; positive finite stakes still required.
    bootstrap_resamples, seed : int
        Paired namespace/event bootstrap budget and deterministic seed.
    confidence_level, alpha : float
        Percentile interval level and Holm familywise threshold.
    max_bootstrap_cells, max_pair_comparisons : int
        Fail-closed resource bounds; no partial family publication.
    max_shortlist_per_category : int
        Presentation cap, applied after full-family correction.
    approval_reference : str
        Provenance of the reviewed policy, never evidence of statistical validity.
    contract_version : int
        Exploratory test/presentation contract.
    """

    min_eligible_wagers: int = 100
    min_event_groups: int = 100
    min_event_coverage: float = 1.0
    max_wager_stake_share: float = 0.05
    max_event_stake_share: float = 0.10
    min_eligible_stake_usd: float = 0.0
    bootstrap_resamples: int = 5000
    seed: int = 20261030
    confidence_level: float = 0.95
    alpha: float = 0.05
    max_bootstrap_cells: int = 5000000000
    max_pair_comparisons: int = 250000
    max_shortlist_per_category: int = 10
    approval_reference: str = (
        "User approved initial exploratory policy in Codex conversation 2026-10-30"
    )
    contract_version: int = 1

    def __post_init__(self):
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("Unsupported prioritization contract")
        for key, lo, hi in [
            ("min_eligible_wagers", 2, 1000000),
            ("min_event_groups", 2, 1000000),
            ("bootstrap_resamples", 100, 10000),
            ("seed", 0, 2**32 - 1),
            ("max_bootstrap_cells", 1, 5000000000),
            ("max_pair_comparisons", 1, 1000000),
            ("max_shortlist_per_category", 1, 1000),
        ]:
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError("Invalid prioritization count/budget")
        for key in ("max_wager_stake_share", "max_event_stake_share", "confidence_level", "alpha"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 < value < 1
            ):
                raise ValueError("Invalid prioritization fraction")
        if self.min_event_coverage != 1.0 or isinstance(self.min_event_coverage, bool):
            raise ValueError("Complete event coverage required")
        if self.min_eligible_stake_usd != 0 or isinstance(self.min_eligible_stake_usd, bool):
            raise ValueError(
                "Contract uses scale-independent positive stakes with no extra dollar floor"
            )
        if not isinstance(self.approval_reference, str) or not self.approval_reference.strip():
            raise ValueError("Policy approval provenance required")
