"""
sizing_config.py
===============================================================================
Purpose
-------
Validate optional, append-only evidence of historical bankroll and Kelly settings.

What this module provides
-------------------------
Strict versioned history records, aware effective intervals and revision validation.

Design Notes
------------
Unknown dates/settings stay unknown. A settings observation is not proof a policy
was applied to a wager; plans and reconstructed assumptions never establish it.
===============================================================================
"""

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from oddsjam_ev.io.source_csv import fingerprint

EVIDENCE_TYPES = {
    "user_confirmed",
    "contemporaneous_record",
    "reconstructed_assumption",
    "unknown",
    "planned",
}
DECIMAL_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def aware(value):
    """Parse a required timezone-aware ISO timestamp without assuming a timezone.

    Parameters
    ----------
    value : str
        Explicit offset-bearing timestamp.

    Returns
    -------
    datetime.datetime
        Aware timestamp; naive and malformed values are rejected.
    """
    if not isinstance(value, str):
        raise ValueError("Aware ISO timestamp required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("Aware ISO timestamp required") from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Aware ISO timestamp required")
    return result


def amount(value, *, maximum=None, positive=False):
    """Validate a nullable finite decimal string, without guessing missing amounts.

    Parameters
    ----------
    value : str or None
        Decimal text; floats are intentionally not a history serialization format.
    maximum : decimal.Decimal or None
        Optional supported upper bound.
    positive : bool
        Require strictly positive rather than nonnegative amounts.

    Returns
    -------
    decimal.Decimal or None
        Validated exact value.
    """
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 128 or not DECIMAL_TEXT.fullmatch(value):
        raise ValueError("Finite decimal amount string required")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError("Finite decimal amount string required") from None
    if (
        not number.is_finite()
        or number < 0
        or (positive and number == 0)
        or (maximum is not None and number > maximum)
    ):
        raise ValueError("Sizing amount outside supported range")
    return number


def _keys(row, required, optional=()):
    if (
        not isinstance(row, dict)
        or set(required) - set(row)
        or set(row) - set(required) - set(optional)
    ):
        raise ValueError("Missing or unsupported sizing history fields")


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty sizing evidence reference/identity required")


def validate_history(history: dict, previous: dict | None = None) -> str:
    """Validate immutable history content and optional exact-prefix append revision.

    Parameters
    ----------
    history : dict
        Contract 1 JSON with policies and independent per-wager evidence records.
    previous : dict or None
        Verified prior history payload. Revisions must preserve all prior records
        exactly, including unresolved/contradictory evidence.

    Returns
    -------
    str
        Content fingerprint; validation never edits or infers historical values.
    """
    _keys(
        history,
        (
            "contract_version",
            "knowledge_as_of",
            "previous_history_id",
            "policies",
            "wager_evidence",
        ),
    )
    if history["contract_version"] != 1 or type(history["contract_version"]) is not int:
        raise ValueError("Unsupported sizing history contract")
    cutoff = aware(history["knowledge_as_of"])
    if not isinstance(history["policies"], list) or not isinstance(history["wager_evidence"], list):
        raise ValueError("Sizing history records must be lists")
    if previous is None:
        if history["previous_history_id"] is not None:
            raise ValueError("Prior history required for this revision")
    else:
        if history["previous_history_id"] != fingerprint(previous) or cutoff < aware(
            previous["knowledge_as_of"]
        ):
            raise ValueError("Sizing history predecessor/cutoff mismatch")
        for key in ("policies", "wager_evidence"):
            if history[key][: len(previous[key])] != previous[key]:
                raise ValueError("Sizing history must be append-only")
    ids = set()
    for policy in history["policies"]:
        _keys(
            policy,
            (
                "policy_id",
                "namespace",
                "effective_from",
                "effective_to",
                "open_ended",
                "timezone",
                "recorded_at",
                "bankroll_amount",
                "currency",
                "kelly_multiplier",
                "evidence_type",
                "confidence",
                "source_reference",
                "effective_interval_verified",
                "scope",
                "constraints",
            ),
        )
        for key in ("policy_id", "namespace", "source_reference"):
            _text(policy[key])
        if policy["policy_id"] in ids:
            raise ValueError("Duplicate sizing policy identity")
        ids.add(policy["policy_id"])
        if (
            policy["currency"] != "USD"
            or policy["evidence_type"] not in EVIDENCE_TYPES
            or policy["confidence"] not in {"confirmed", "assumed", "unknown"}
        ):
            raise ValueError("Unsupported sizing currency or evidence status")
        for key in ("open_ended", "effective_interval_verified"):
            if type(policy[key]) is not bool:
                raise ValueError("Sizing verification flags must be booleans")
        try:
            zone = None if policy["timezone"] in (None, "unknown") else ZoneInfo(policy["timezone"])
        except (ZoneInfoNotFoundError, TypeError, ValueError):
            raise ValueError("Unknown sizing timezone") from None
        if policy["effective_interval_verified"] and zone is None:
            raise ValueError("Verified sizing intervals require a known timezone")
        if aware(policy["recorded_at"]) > cutoff:
            raise ValueError("Sizing record is beyond knowledge cutoff")
        start, end = policy["effective_from"], policy["effective_to"]
        if start is None:
            if end is not None or policy["open_ended"] or policy["effective_interval_verified"]:
                raise ValueError("Undated settings cannot establish an effective interval")
        else:
            start = aware(start)
            if policy["open_ended"] != (end is None):
                raise ValueError("Explicit open-ended interval required")
            if end is not None and aware(end) <= start:
                raise ValueError("Sizing interval must have positive duration")
            for text in (policy["effective_from"], end):
                if text is not None:
                    stamp = aware(text)
                    if zone is not None and stamp.utcoffset() != stamp.astimezone(zone).utcoffset():
                        raise ValueError("Sizing timestamp offset disagrees with named timezone")
        amount(policy["bankroll_amount"], positive=True)
        amount(policy["kelly_multiplier"], maximum=Decimal("1"))
        _keys(policy["scope"], ("product", "execution_status"))
        if policy["scope"]["product"] not in {
            None,
            "exchange",
            "fixed_odds",
            "pickem",
            "unknown",
        } or policy["scope"]["execution_status"] not in {None, "real_money", "paper", "unknown"}:
            raise ValueError("Unsupported sizing scope")
        if not isinstance(policy["constraints"], list):
            raise ValueError("Sizing constraints must be evidenced notes")
        for note in policy["constraints"]:
            _keys(note, ("kind", "description", "source_reference"))
            if note["kind"] not in {"cap", "liquidity", "rounding", "override"}:
                raise ValueError("Unsupported sizing constraint")
            _text(note["description"])
            _text(note["source_reference"])
    evidence_ids = set()
    for record in history["wager_evidence"]:
        _keys(
            record,
            (
                "evidence_id",
                "bet_id",
                "recorded_at",
                "source_reference",
                "placement_at",
                "placement_verified",
                "applied_policy_id",
                "application_verified",
                "suggested_stake",
                "accepted_stake",
                "accepted_verified",
                "deviation_reason",
            ),
        )
        for key in ("evidence_id", "bet_id", "source_reference"):
            _text(record[key])
        if record["evidence_id"] in evidence_ids:
            raise ValueError("Duplicate sizing wager evidence identity")
        evidence_ids.add(record["evidence_id"])
        if aware(record["recorded_at"]) > cutoff:
            raise ValueError("Wager evidence is beyond knowledge cutoff")
        for key in ("placement_verified", "application_verified", "accepted_verified"):
            if type(record[key]) is not bool:
                raise ValueError("Wager verification flags must be booleans")
        if record["placement_at"] is not None:
            if aware(record["placement_at"]) > aware(record["recorded_at"]):
                raise ValueError("Placement evidence cannot predate its event")
        elif record["placement_verified"]:
            raise ValueError("Verified placement timestamp required")
        if record["applied_policy_id"] is not None and record["applied_policy_id"] not in ids:
            raise ValueError("Unknown applied sizing policy")
        if record["application_verified"] and (
            record["applied_policy_id"] is None or not record["placement_verified"]
        ):
            raise ValueError("Applied policy requires explicit identity and placement evidence")
        amount(record["suggested_stake"])
        amount(record["accepted_stake"])
        if record["accepted_verified"] and record["accepted_stake"] is None:
            raise ValueError("Verified accepted stake amount required")
        if record["deviation_reason"] is not None:
            _text(record["deviation_reason"])
    return fingerprint(history)
