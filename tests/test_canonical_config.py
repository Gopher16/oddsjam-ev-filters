"""Canonical economic policy tests independent of the publication layer."""

import pytest

from oddsjam_ev.analysis.canonical_config import CanonicalConfig


def test_exact_product_mappings_and_unknowns():
    cfg = CanonicalConfig({"fixed_odds": ["Book"], "exchange": ["Exchange"], "pickem": ["Picks"]})
    assert cfg.product("Book") == "fixed_odds"
    assert cfg.product("Exchange") == "exchange"
    assert cfg.product("Picks") == "pickem"
    assert cfg.product("book") == cfg.product("Unlisted") == "unknown"
    assert cfg.currency == "USD"
    assert cfg.fee_inclusion == "unverified"


@pytest.mark.parametrize(
    "updates",
    [
        {"currency": "EUR"},
        {"fee_inclusion": "verified"},
        {"refunds_in_roi": True},
        {"profit_tolerance_usd": -1},
        {"profit_tolerance_usd": float("nan")},
        {"currency_evidence": ""},
        {"products": {}},
        {"products": {"fixed_odds": ["Book"], "exchange": ["Book"], "pickem": []}},
    ],
)
def test_invalid_policy(updates):
    with pytest.raises(ValueError):
        CanonicalConfig(**{"products": {"fixed_odds": [], "exchange": [], "pickem": []}, **updates})
