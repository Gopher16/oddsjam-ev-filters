"""
experiment_config.py
===============================================================================
Purpose
-------
Validate explicit paper experiment protocols and timing evidence.

What this module provides
-------------------------
Paper-only policy, protocol requirements and aware timestamp parsing.

Design Notes
------------
Unknown availability is permitted for signal screening, never proof of execution.
===============================================================================
"""

from dataclasses import dataclass

import pandas as pd

from oddsjam_ev.analysis.recommendations import protocol_blockers


@dataclass(frozen=True)
class ExperimentPolicy:
    """Paper-only experiment constraints.

    Attributes
    ----------
    funding : str
        Only paper_only is supported.
    max_funded_stake_usd : int
        Must remain zero.
    allowed_types : tuple[str, ...]
        Signal screening and sampled execution verification.
    exchange_order_behavior : str
        User-confirmed taker-only workflow; never evidence of a fill.
    contract_version : int
        Experiment contract version.
    """

    funding: str = "paper_only"
    max_funded_stake_usd: int = 0
    allowed_types: tuple = ("signal_screening", "execution_verification")
    exchange_order_behavior: str = "take_existing_orders_only"
    contract_version: int = 1

    def __post_init__(self):
        if (
            self.contract_version != 1
            or self.funding != "paper_only"
            or self.exchange_order_behavior != "take_existing_orders_only"
            or type(self.max_funded_stake_usd) not in (int, float)
            or self.max_funded_stake_usd != 0
            or set(self.allowed_types) != {"signal_screening", "execution_verification"}
        ):
            raise ValueError("Only zero-funded-risk paper experiments are supported")
        object.__setattr__(self, "allowed_types", tuple(self.allowed_types))


def aware(value) -> pd.Timestamp:
    """Parse a finite explicitly timezone-aware timestamp.

    Parameters
    ----------
    value : str or pandas.Timestamp
        Evidence time, never a naive source timestamp.

    Returns
    -------
    pandas.Timestamp
        UTC timestamp.
    """
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError("Explicit timezone-aware time required")
    return t.tz_convert("UTC")


def validate_protocol(protocol: dict, policy: ExperimentPolicy) -> None:
    """Validate operational detail beyond an approved draft protocol.

    Parameters
    ----------
    protocol : dict
        Complete explicitly specified prospective protocol.
    policy : ExperimentPolicy
        Paper-only constraints.

    Returns
    -------
    None
    """
    if protocol_blockers(protocol):
        raise ValueError("Incomplete or funded experiment protocol")
    if protocol.get("experiment_type") not in policy.allowed_types:
        raise ValueError("Unknown experiment type")
    for field in [
        "name",
        "filter_id",
        "comparison_plan",
        "overlap_policy",
        "expected_liquidity",
        "stop_rules",
    ]:
        if not isinstance(protocol.get(field), str) or not protocol[field].strip():
            raise ValueError(f"{field} is required")
    if (
        not isinstance(protocol.get("filter_config"), dict)
        or not protocol["filter_config"]
        or not isinstance(protocol.get("predicate"), dict)
        or not protocol["predicate"]
    ):
        raise ValueError("Exact filter config and predicate are required")
    if (
        type(protocol.get("settlement_followup_days")) is not int
        or protocol["settlement_followup_days"] < 0
    ):
        raise ValueError("Settlement follow-up must be explicit")
    start, end = aware(protocol.get("planned_start_utc")), aware(protocol.get("planned_end_utc"))
    if end - start != pd.Timedelta(days=protocol["duration_days"]):
        raise ValueError("Enrollment window must match the approved duration")
    if protocol["experiment_type"] == "execution_verification" and (
        not isinstance(protocol.get("sampling_rule"), str) or not protocol["sampling_rule"].strip()
    ):
        raise ValueError("Verification experiments require a predeclared sampling rule")
