"""Venue identity never substitutes for a product contract, access or permission."""

import copy

import pytest

from oddsjam_ev.analysis.venue_registry import (
    product_version,
    resolve_product,
    seed_from_policy,
    validate_registry,
)
from oddsjam_ev.io.source_csv import fingerprint

STAMP = "2026-11-24T00:00:00Z"


def policy():
    return dict(
        products={
            "fixed_odds": ["BookMaker", "BetOnline"],
            "exchange": ["Robinhood"],
            "pickem": ["DraftKings (Pick 3)", "DraftKings (Pick 6)"],
        }
    )


def ready_registry():
    r = seed_from_policy(policy(), STAMP)
    p = next(p for p in r["products"] if p["aliases"] == ["BookMaker"])
    # This fixture is invented contract/access evidence, not a claim about BookMaker.
    p.update(
        effective_from="2026-11-01T00:00:00Z",
        effective_to="2026-12-01T00:00:00Z",
        scope_verified=True,
    )
    p["contract"].update(
        evidence_state="confirmed",
        currency="USD",
        price_unit="decimal_odds",
        stake_unit="usd",
        payout_unit="gross_usd",
        fee_model="none",
        settlement_model="fixed_odds_win_loss",
        settlement_rules="synthetic rules with push/void handling",
        reference="synthetic contract",
    )
    r["access"] = [
        dict(
            evidence_id="access1",
            namespace="synthetic",
            product_id=p["product_id"],
            source_id="oddsjam_csv",
            state="confirmed",
            effective_from=p["effective_from"],
            effective_to=p["effective_to"],
            scope_verified=True,
            open_ended=False,
            recorded_at=STAMP,
            reference="synthetic access",
        )
    ]
    return r, p


def query(r, **kw):
    return resolve_product(
        r,
        "BookMaker",
        source_id="oddsjam_csv",
        namespace="synthetic",
        at="2026-11-10T00:00:00Z",
        **kw,
    )


def test_seed_is_deterministic_non_mutating_and_requests_prove_nothing():
    old = policy()
    before = copy.deepcopy(old)
    r = seed_from_policy(old, STAMP)
    assert old == before and r == seed_from_policy(old, STAMP)
    assert r["source_policy_id"] == fingerprint(old)
    assert len(r["products"]) == 5 and not r["access"]
    for name in ("Robinhood", "DraftKings", "FanDuel"):
        got = resolve_product(r, name, source_id="oddsjam_csv")
        assert not got["evidence_ready"] and not got["economic_contract_ready"]
        assert got["request_ids"] and not got["execution_authorized"]
    assert resolve_product(r, "DraftKings", source_id="oddsjam_csv")["status"] == "ambiguous"
    got = resolve_product(r, "DraftKings", source_id="oddsjam_csv", label="DraftKings (Pick 6)")
    assert got["product_type"] == "pickem" and "settlement_unsupported" in got["blockers"]
    assert resolve_product(r, "draftkings", source_id="oddsjam_csv")["status"] == "unknown"


def test_confirmed_contract_access_and_operation_have_separate_gates():
    r, p = ready_registry()
    validate_registry(r)
    before = copy.deepcopy(r)
    assert query(r)["evidence_ready"] and r == before
    assert not query(r, operation="orders_submit")["evidence_ready"]
    p["contract"]["fee_model"] = "unknown"
    assert "fees_unknown" in query(r)["blockers"]
    p["contract"]["fee_model"] = "rate"
    p["contract"]["fee_rate"] = 0.02
    assert "fees_unknown" in query(r)["blockers"]
    p["contract"]["fee_basis"] = "gross_profit"
    assert query(r)["evidence_ready"]
    p["contract"]["settlement_model"] = "ticket_paytable"
    assert "settlement_unsupported" in query(r)["blockers"]
    p["contract"]["settlement_model"] = "fixed_odds_win_loss"
    p["contract"]["price_unit"] = "usd_per_contract"
    assert "units_unsupported" in query(r)["blockers"]


def test_access_conflicts_unknown_dates_namespace_and_permission():
    r, p = ready_registry()
    r["access"].append({**r["access"][0], "evidence_id": "denial", "state": "denied"})
    assert "access_unverified_or_conflicting" in query(r)["blockers"]
    r["access"][1].update(effective_from=None, effective_to=None, scope_verified=False)
    assert "access_unverified_or_conflicting" in query(r)["blockers"]
    r["access"] = r["access"][:1]
    r["access"][0]["namespace"] = "other"
    assert not query(r)["evidence_ready"]
    p["operations"]["manual_csv_import"]["permission"] = "denied"
    assert "operation_not_permitted" in query(r)["blockers"]


def test_half_open_scope_and_preserved_adjacent_product_versions():
    r, p = ready_registry()
    nxt = copy.deepcopy(p)
    nxt.update(
        version_id="next", effective_from=p["effective_to"], effective_to="2027-01-01T00:00:00Z"
    )
    r["products"].append(nxt)
    validate_registry(r)
    kwargs = dict(source_id="oddsjam_csv", namespace="synthetic")
    assert resolve_product(r, "BookMaker", at="2026-11-01T00:00:00Z", **kwargs)[
        "candidate_version_ids"
    ] == [p["version_id"]]
    got = resolve_product(r, "BookMaker", at=p["effective_to"], **kwargs)
    assert got["candidate_version_ids"] == ["next"]
    assert not got["evidence_ready"]  # Original account-access interval has ended.
    assert resolve_product(r, "BookMaker", **kwargs)["status"] == "ambiguous"


def test_alias_collision_and_immutable_product_identity():
    r, p = ready_registry()
    other = copy.deepcopy(p)
    other.update(version_id="collision", product_id="different")
    r["products"].append(other)
    with pytest.raises(ValueError, match="alias collision"):
        validate_registry(r)
    other["product_id"] = p["product_id"]
    other["product_type"] = "exchange"
    with pytest.raises(ValueError, match="ownership/type"):
        validate_registry(r)
    r, p = ready_registry()
    r["venue_aliases"].append({**r["venue_aliases"][0], "venue_id": r["venues"][1]["venue_id"]})
    with pytest.raises(ValueError, match="alias collision"):
        validate_registry(r)


def test_append_only_knowledge_and_empty_registry():
    r, _ = ready_registry()
    new = copy.deepcopy(r)
    new["previous_registry_id"] = fingerprint(r)
    validate_registry(new, r)
    new["products"][0]["reference"] = "rewritten"
    with pytest.raises(ValueError, match="append-only"):
        validate_registry(new, r)
    with pytest.raises(ValueError, match="predecessor"):
        validate_registry({**r, "previous_registry_id": "bad"})
    with pytest.raises(ValueError, match="cutoff"):
        validate_registry({**r, "knowledge_as_of": "2026-01-01T00:00:00Z"})
    empty = seed_from_policy(dict(products={"fixed_odds": [], "exchange": [], "pickem": []}), STAMP)
    assert resolve_product(empty, "Book", source_id="none")["blockers"] == ["unknown_product"]


@pytest.mark.parametrize(
    "update",
    [
        {"effective_from": "2026-11-01T00:00:00"},
        {"scope_verified": 1},
        {"effective_to": None},
        {"product_type": "sportsbook"},
        {"aliases": [" "]},
        {"accepted_fill": True},
    ],
)
def test_invalid_product_metadata_rejected(update):
    r, p = ready_registry()
    p.update(update)
    with pytest.raises(ValueError):
        validate_registry(r)


def test_exchange_contract_requires_payout_units_and_positive_unit_value():
    r, p = ready_registry()
    p["product_type"] = "exchange"
    p["contract"].update(
        price_unit="usd_per_contract",
        stake_unit="contracts",
        payout_unit="usd_per_contract",
        settlement_model="binary_contract",
    )
    assert "unit_payout_missing" in query(r)["blockers"]
    p["contract"]["unit_payout"] = 1.0
    assert query(r)["economic_contract_ready"]
    p["contract"]["fee_rate"] = float("nan")
    with pytest.raises(ValueError, match="amount/rate"):
        validate_registry(r)


def test_factory_does_not_claim_support_or_access():
    p = product_version("v", "p", "venue", "source", "dfs", ["Contest"], STAMP, "synthetic")
    assert p["contract"]["fee_model"] == "unknown" and not p["scope_verified"]
    assert all(op["permission"] == "unknown" for op in p["operations"].values())


def test_explicit_version_pin_retains_unknown_history_without_latest_wins():
    r, p = ready_registry()
    old = copy.deepcopy(p)
    old.update(
        version_id="unknown-legacy", effective_from=None, effective_to=None, scope_verified=False
    )
    r["products"].append(old)
    assert query(r)["status"] == "ambiguous"
    assert query(r, version_id=p["version_id"])["evidence_ready"]
    assert not query(r, version_id="unknown-legacy")["evidence_ready"]
    assert query(r, version_id="missing")["status"] == "unknown"
