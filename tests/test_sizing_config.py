"""Strict synthetic sizing history, uncertainty and append-only evidence contracts."""

import copy

import pytest

from oddsjam_ev.analysis.sizing_config import validate_history


def policy(ident="p1", start="2026-01-01T00:00:00+00:00", end="2026-02-01T00:00:00+00:00"):
    return {
        "policy_id": ident,
        "namespace": "synthetic",
        "effective_from": start,
        "effective_to": end,
        "open_ended": False,
        "timezone": "UTC",
        "recorded_at": "2026-03-01T00:00:00Z",
        "bankroll_amount": "1000",
        "currency": "USD",
        "kelly_multiplier": "0.5",
        "evidence_type": "user_confirmed",
        "confidence": "confirmed",
        "source_reference": "synthetic statement",
        "effective_interval_verified": True,
        "scope": {"product": None, "execution_status": None},
        "constraints": [],
    }


def evidence(bet="b1", ident="e1", stamp="2026-01-15T00:00:00Z"):
    return {
        "evidence_id": ident,
        "bet_id": bet,
        "recorded_at": "2026-03-01T00:00:00Z",
        "source_reference": "synthetic execution record",
        "placement_at": stamp,
        "placement_verified": True,
        "applied_policy_id": None,
        "application_verified": False,
        "suggested_stake": None,
        "accepted_stake": None,
        "accepted_verified": False,
        "deviation_reason": None,
    }


def history():
    return {
        "contract_version": 1,
        "knowledge_as_of": "2026-03-02T00:00:00Z",
        "previous_history_id": None,
        "policies": [policy()],
        "wager_evidence": [],
    }


def test_append_only_and_unknown_values():
    old = history()
    original = copy.deepcopy(old)
    ident = validate_history(old)
    new = copy.deepcopy(old)
    new["previous_history_id"] = ident
    undated = policy("unknown", None, None)
    undated.update(
        open_ended=False,
        effective_interval_verified=False,
        bankroll_amount=None,
        kelly_multiplier=None,
        confidence="unknown",
        evidence_type="unknown",
    )
    new["policies"].append(undated)
    assert validate_history(new, old) != ident
    assert old == original
    new["policies"][0]["kelly_multiplier"] = "0.25"
    with pytest.raises(ValueError, match="append-only"):
        validate_history(new, old)


@pytest.mark.parametrize(
    "key,value",
    [
        ("bankroll_amount", "0"),
        ("bankroll_amount", "NaN"),
        ("bankroll_amount", 1000),
        ("kelly_multiplier", "1.1"),
        ("kelly_multiplier", "-0.1"),
        ("currency", "EUR"),
        ("effective_from", "2026-01-01"),
        ("effective_to", "2025-01-01T00:00:00Z"),
        ("recorded_at", "2027-01-01T00:00:00Z"),
        ("timezone", "America/Chicago"),
        ("open_ended", True),
        ("effective_interval_verified", "yes"),
    ],
)
def test_invalid_policy(key, value):
    data = history()
    data["policies"][0][key] = value
    with pytest.raises(ValueError):
        validate_history(data)


def test_duplicate_unknown_field_and_wager_checks():
    data = history()
    data["policies"].append(policy())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_history(data)
    data = history()
    data["extra"] = True
    with pytest.raises(ValueError):
        validate_history(data)
    data = history()
    record = evidence()
    record.update(
        applied_policy_id="p1",
        application_verified=True,
        placement_at=None,
        placement_verified=False,
    )
    data["wager_evidence"] = [record]
    with pytest.raises(ValueError):
        validate_history(data)


def test_dst_offsets_and_half_open_adjacent_intervals():
    data = history()
    p = data["policies"][0]
    p.update(
        timezone="America/New_York",
        effective_from="2026-03-08T01:30:00-05:00",
        effective_to="2026-03-08T03:30:00-04:00",
        recorded_at="2026-03-09T00:00:00Z",
    )
    data["knowledge_as_of"] = "2026-03-10T00:00:00Z"
    validate_history(data)
    p["effective_to"] = "2026-03-08T03:30:00-05:00"
    with pytest.raises(ValueError, match="timezone"):
        validate_history(data)
