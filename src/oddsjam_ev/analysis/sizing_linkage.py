"""
sizing_linkage.py
===============================================================================
Purpose
-------
Audit candidate sizing regimes without guessing placement times or policy use.

What this module provides
-------------------------
Half-open interval matching, overlap/contradiction quarantine and unknown coverage.

Design Notes
------------
Canonical created_at and export receipt are not verified placement evidence.
Interval membership alone never establishes application. No Kelly replay is run.
===============================================================================
"""

import json
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import combinations

import pandas as pd

from oddsjam_ev.analysis.sizing_config import aware

LINK_COLUMNS = [
    "bet_id",
    "product",
    "execution_status",
    "source_reported_stake_usd",
    "placement_at_verified",
    "candidate_policy_ids_json",
    "setting_policy_id",
    "applied_policy_id",
    "bankroll_amount",
    "kelly_multiplier",
    "currency",
    "linkage_status",
    "suggested_stake_recorded",
    "accepted_stake_verified",
    "stake_evidence_status",
    "deviation_reasons_json",
    "evidence_ids_json",
    "sizing_replay_blocked",
    "replay_block_reasons_json",
]
BLOCKS = [
    "settlement_order_unverified",
    "concurrent_exposure_unverified",
    "cash_flows_unverified",
    "source_sizing_algorithm_unverified",
    "fees_and_fillability_unverified",
]


def _trusted(policy):
    return (
        policy["effective_interval_verified"]
        and policy["confidence"] == "confirmed"
        and policy["evidence_type"] in {"user_confirmed", "contemporaneous_record"}
    )


def _same_scope(a, b):
    return a["namespace"] == b["namespace"] and all(
        a["scope"][key] is None or b["scope"][key] is None or a["scope"][key] == b["scope"][key]
        for key in ("product", "execution_status")
    )


def build_sizing_linkage(bets: pd.DataFrame, history: dict) -> tuple[pd.DataFrame, dict]:
    """Link verified placement evidence to settings; audit unresolved populations.

    Parameters
    ----------
    bets : pandas.DataFrame
        Verified unique canonical wager candidates, not expanded memberships.
    history : dict
        History already validated with its predecessor by the publication layer.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        One non-economic linkage row per candidate and overlap/coverage audit.
        Source reported stake is retained without interpreting paper as a fill.
    """
    required = {"bet_id", "namespace", "product", "execution_status", "reported_stake_usd"}
    if required - set(bets) or bets.bet_id.isna().any() or bets.bet_id.duplicated().any():
        raise ValueError("Unique canonical wager identities and sizing provenance required")
    policies = {p["policy_id"]: p for p in history["policies"]}
    bybet = defaultdict(list)
    known = set(bets.bet_id)
    for record in history["wager_evidence"]:
        if record["bet_id"] not in known:
            raise ValueError("Orphan sizing wager evidence")
        bybet[record["bet_id"]].append(record)
    overlaps = []
    trusted = [p for p in policies.values() if _trusted(p)]
    for a, b in combinations(trusted, 2):
        if (
            _same_scope(a, b)
            and (b["effective_to"] is None or aware(a["effective_from"]) < aware(b["effective_to"]))
            and (a["effective_to"] is None or aware(b["effective_from"]) < aware(a["effective_to"]))
        ):
            overlaps.append(sorted([a["policy_id"], b["policy_id"]]))
    rows = []
    for bet in bets.to_dict("records"):
        records = bybet[bet["bet_id"]]
        if any(r["accepted_verified"] for r in records) and bet["execution_status"] != "real_money":
            raise ValueError("Accepted fill evidence requires verified real-money classification")
        times = {aware(r["placement_at"]) for r in records if r["placement_verified"]}
        status = (
            "unknown_placement"
            if not times
            else "ambiguous_placement"
            if len(times) > 1
            else "unknown_policy_interval"
        )
        matches = []
        stamp = next(iter(times)) if len(times) == 1 else None
        if stamp is not None:
            matches = sorted(
                p["policy_id"]
                for p in trusted
                if p["namespace"] == bet["namespace"]
                and all(
                    p["scope"][key] is None or p["scope"][key] == bet[key]
                    for key in ("product", "execution_status")
                )
                and aware(p["effective_from"]) <= stamp
                and (p["effective_to"] is None or stamp < aware(p["effective_to"]))
            )
            if len(matches) > 1:
                status = "ambiguous_policy_interval"
            elif len(matches) == 1:
                status = "matched_setting_interval"
        setting = matches[0] if len(matches) == 1 else None
        applied_claims = {r["applied_policy_id"] for r in records if r["application_verified"]}
        applied = None
        if applied_claims:
            if setting is not None and applied_claims == {setting}:
                applied = setting
                status = "application_verified"
            else:
                status = "application_conflict"
        policy = policies[setting] if setting else None
        suggested = {r["suggested_stake"] for r in records if r["suggested_stake"] is not None}
        accepted = {r["accepted_stake"] for r in records if r["accepted_verified"]}
        # Numeric-equivalent representations are not contradictory evidence.
        conflict = (
            len({Decimal(v) for v in suggested}) > 1 or len({Decimal(v) for v in accepted}) > 1
        )
        rows.append(
            {
                "bet_id": bet["bet_id"],
                "product": bet["product"],
                "execution_status": bet["execution_status"],
                "source_reported_stake_usd": bet["reported_stake_usd"],
                "placement_at_verified": stamp.isoformat() if stamp else None,
                "candidate_policy_ids_json": json.dumps(matches),
                "setting_policy_id": setting,
                "applied_policy_id": applied,
                "bankroll_amount": policy["bankroll_amount"] if policy else None,
                "kelly_multiplier": policy["kelly_multiplier"] if policy else None,
                "currency": policy["currency"] if policy else None,
                "linkage_status": status,
                "suggested_stake_recorded": sorted(suggested)[0]
                if suggested and not conflict
                else None,
                "accepted_stake_verified": sorted(accepted)[0]
                if accepted and not conflict
                else None,
                "stake_evidence_status": "conflicting"
                if conflict
                else "accepted_verified"
                if accepted
                else "suggested_only"
                if suggested
                else "unknown",
                "deviation_reasons_json": json.dumps(
                    sorted({r["deviation_reason"] for r in records if r["deviation_reason"]})
                ),
                "evidence_ids_json": json.dumps(sorted(r["evidence_id"] for r in records)),
                "sizing_replay_blocked": True,
                "replay_block_reasons_json": json.dumps(BLOCKS),
            }
        )
    frame = pd.DataFrame(rows, columns=LINK_COLUMNS)
    audit = {
        "candidate_count": len(bets),
        "linkage_rows": len(frame),
        "linkage_status_counts": dict(sorted(Counter(frame.linkage_status).items())),
        "policy_count": len(policies),
        "undated_policy_count": sum(p["effective_from"] is None for p in policies.values()),
        "untrusted_policy_count": sum(not _trusted(p) for p in policies.values()),
        "overlapping_policy_pairs": sorted(overlaps),
        "stake_evidence_conflicts": int(frame.stake_evidence_status.eq("conflicting").sum()),
        "applied_policy_count": int(frame.applied_policy_id.notna().sum()),
        "sizing_replay_blocked": True,
        "replay_block_reasons": BLOCKS,
        "scope": "One row per canonical candidate, including paper/unknown and ineligible products; no economic eligibility or return metric changes.",
    }
    return frame, audit
