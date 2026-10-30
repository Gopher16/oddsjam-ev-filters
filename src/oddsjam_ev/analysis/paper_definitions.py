"""
paper_definitions.py
===============================================================================
Purpose
-------
Freeze prospective paper-test intent separately from launch and outcome evidence.

What this module provides
-------------------------
Strict definition validation and editable candidate-bound draft templates.

Design Notes
------------
A definition is not experiment approval, enrollment, a fill, or validation.
===============================================================================
"""

import math
import re

from oddsjam_ev.analysis.experiment_config import aware

KEY = re.compile(r"[a-z][a-z0-9_-]{2,79}\Z")
METRICS = {"stake_weighted_roi", "equal_stake_mean_return"}


def validate_definition(protocol: dict) -> None:
    """Require explicit prospective timing, sizing and review intent.

    Parameters
    ----------
    protocol : dict
        Human-completed definition; source rules are attached separately.

    Returns
    -------
    None
        Raises ValueError for incomplete or unsupported intent.
    """
    required = {
        "name",
        "hypothesis",
        "design",
        "baseline",
        "planned_start_utc",
        "planned_end_utc",
        "review_dates_utc",
        "sizing",
        "min_wagers_per_arm",
        "min_event_groups_per_arm",
        "primary_metric",
        "secondary_metric",
        "review_criteria",
        "stop_rules",
        "overlap_policy",
        "quote_availability_policy",
        "settlement_followup_days",
        "funding",
        "max_funded_stake_usd",
    }
    if not isinstance(protocol, dict) or set(protocol) - {"configuration_reference"} != required:
        raise ValueError("Exact definition fields required; unknown fields are not ignored")
    if "configuration_reference" in protocol:
        reference = protocol["configuration_reference"]
        if (
            not isinstance(reference, dict)
            or set(reference)
            != {"contract_version", "publication_id", "decision_id", "record_sha256"}
            or type(reference["contract_version"]) is not int
            or reference["contract_version"] != 1
            or not all(
                isinstance(reference[k], str) and reference[k].strip()
                for k in ("publication_id", "decision_id", "record_sha256")
            )
        ):
            raise ValueError("Exact versioned configuration reference required")
    for field in ("name", "hypothesis", "overlap_policy", "quote_availability_policy"):
        if not isinstance(protocol[field], str) or not protocol[field].strip():
            raise ValueError(field + " must be explicit")
    if protocol["design"] not in {"subgroup_tracking", "exclusion_comparison"}:
        raise ValueError("Unsupported paper comparison design")
    baseline = protocol["baseline"]
    if (
        not isinstance(baseline, dict)
        or set(baseline) != {"kind", "filter_config", "evidence_reference"}
        or baseline["kind"] != "unchanged_parent"
    ):
        raise ValueError("Explicit unchanged-parent baseline required")
    if (
        not isinstance(baseline["filter_config"], dict)
        or not baseline["filter_config"]
        or not isinstance(baseline["evidence_reference"], str)
        or not baseline["evidence_reference"].strip()
    ):
        raise ValueError(
            "Prospective baseline settings and evidence required; historical settings are not inferred"
        )
    if (
        protocol["funding"] != "paper_only"
        or type(protocol["max_funded_stake_usd"]) is not int
        or protocol["max_funded_stake_usd"] != 0
    ):
        raise ValueError("Definitions must be paper only with zero funded risk")
    sizing = protocol["sizing"]
    if (
        not isinstance(sizing, dict)
        or set(sizing) != {"method", "stake_usd", "applies_to"}
        or sizing["method"] != "fixed_paper_stake"
        or sizing["applies_to"] != "both_arms"
    ):
        raise ValueError("Explicit identical fixed paper stake required for both arms")
    stake = sizing["stake_usd"]
    if type(stake) not in (int, float) or not math.isfinite(stake) or stake <= 0:
        raise ValueError("Positive finite paper stake required")
    for field in ("min_wagers_per_arm", "min_event_groups_per_arm"):
        if type(protocol[field]) is not int or protocol[field] < 1:
            raise ValueError(field + " must be a positive integer")
    if (
        type(protocol["settlement_followup_days"]) is not int
        or protocol["settlement_followup_days"] < 0
    ):
        raise ValueError("Explicit nonnegative settlement follow-up required")
    if {protocol["primary_metric"], protocol["secondary_metric"]} != METRICS:
        raise ValueError("Weighted and equal-stake measures must both be predeclared")
    criteria = protocol["review_criteria"]
    if (
        not isinstance(criteria, dict)
        or set(criteria) != {"continue", "stop", "inconclusive"}
        or not all(isinstance(v, str) and v.strip() for v in criteria.values())
    ):
        raise ValueError("Continue, stop and inconclusive review criteria required")
    rules = protocol["stop_rules"]
    if (
        not isinstance(rules, list)
        or not rules
        or not all(isinstance(v, str) and v.strip() for v in rules)
    ):
        raise ValueError("Explicit stopping rules required")
    start, end = aware(protocol["planned_start_utc"]), aware(protocol["planned_end_utc"])
    reviews = protocol["review_dates_utc"]
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("Predeclared review dates required")
    dates = [aware(v) for v in reviews]
    import pandas as pd

    if (
        end <= start
        or dates != sorted(set(dates))
        or dates[0] < start
        or dates[-1] < end + pd.Timedelta(days=protocol["settlement_followup_days"])
    ):
        raise ValueError("Ordered unique review dates must include final settlement follow-up")


def draft_definition(candidate: dict) -> dict:
    """Create an intentionally incomplete draft, not a registered test.

    Parameters
    ----------
    candidate : dict
        Verified candidate identity and exact predicate from discovery.

    Returns
    -------
    dict
        Editable human-intent fields and a read-only source reminder. Dates and
        baseline settings remain empty until explicitly chosen by the operator.
    """
    return {
        "candidate_id": candidate["candidate_id"],
        "exact_predicate_json": candidate["predicate_json"],
        "protocol": {
            "name": "",
            "hypothesis": "",
            "design": "subgroup_tracking",
            "baseline": {"kind": "unchanged_parent", "filter_config": {}, "evidence_reference": ""},
            "planned_start_utc": None,
            "planned_end_utc": None,
            "review_dates_utc": [],
            "sizing": {"method": "fixed_paper_stake", "stake_usd": None, "applies_to": "both_arms"},
            "min_wagers_per_arm": 100,
            "min_event_groups_per_arm": 100,
            "primary_metric": "equal_stake_mean_return",
            "secondary_metric": "stake_weighted_roi",
            "review_criteria": {"continue": "", "stop": "", "inconclusive": ""},
            "stop_rules": [],
            "overlap_policy": "",
            "quote_availability_policy": "",
            "settlement_followup_days": 0,
            "funding": "paper_only",
            "max_funded_stake_usd": 0,
        },
    }
