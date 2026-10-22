"""Attribution, opportunity proxy, feature role, and registry integrity contracts."""

import pandas as pd

from oddsjam_ev.analysis.feature_catalog import build_feature_catalog
from oddsjam_ev.analysis.opportunity_facts import build_opportunity_facts


def lineage_row(index, **updates):
    return {
        "bet_id": f"bet-{index}",
        "_ingest_observation_id": f"obs-{index}",
        "_ingest_record_id": f"r-{index}",
        "_ingest_namespace": "test",
        "_ingest_timestamp_timezone": "unknown",
        "game_id": "g",
        "market_name": "Moneyline",
        "bet_name": "Away",
        "sportsbook": "Book",
        "odds": "120",
        "created_at": "2026-01-01 12:00:00",
        **updates,
    }


def test_proxy_repeated_executions_preserved_and_no_money():
    opportunities, links = build_opportunity_facts(pd.DataFrame([lineage_row(0), lineage_row(1)]))
    assert len(opportunities) == 1 and len(links) == 2
    assert opportunities.iloc[0].linked_source_record_count == 2
    assert not opportunities.unique_opportunity_verified.any()
    assert {"stake", "bet_profit", "reported_roi"}.isdisjoint(opportunities.columns)


def test_selection_group_across_prices_and_venues():
    opportunities, links = build_opportunity_facts(
        pd.DataFrame([lineage_row(0), lineage_row(1, odds="130", sportsbook="Other")])
    )
    assert len(opportunities) == 2
    assert opportunities.selection_proxy_id.nunique() == 1
    assert links.bet_id.nunique() == 2


def test_missing_identity_gets_distinct_fallbacks():
    opportunities, _ = build_opportunity_facts(
        pd.DataFrame([lineage_row(0, game_id=""), lineage_row(1, game_id="")])
    )
    assert len(opportunities) == 2
    assert set(opportunities.identity_quality) == {"observation_fallback"}


def test_feature_catalog_preserves_empty_and_outcome_roles():
    catalog = build_feature_catalog(
        pd.DataFrame({"clv": ["120", ""], "percentage": ["5", ""], "liquidity": ["", ""]})
    ).set_index("source_column")
    assert catalog.loc["percentage", "nonempty_records"] == 1
    assert catalog.loc["liquidity", "nonempty_records"] == 0
    assert catalog.loc["clv", "temporal_role"] == "post_decision_or_unverified"
    assert not catalog.loc["settled_at", "column_present"]
    assert not catalog.decision_time_verified.any()
