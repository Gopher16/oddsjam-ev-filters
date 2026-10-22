"""Synthetic source-grain, ticket completeness, and reported-economics contracts."""

import json

import pandas as pd
import pytest

from oddsjam_ev.analysis.bet_facts import build_bet_facts
from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.analysis.payouts import qualify_economics, summarize_economics


def config():
    return CanonicalConfig(
        {"fixed_odds": ["Book"], "exchange": ["Exchange"], "pickem": ["Picks"]},
        pickem_filter_ids=("pickem",),
    )


def registry():
    return {
        "schema_version": 1,
        "snapshot_date": "2026-10-22",
        "filters": [
            {"id": "core", "name": "PROD - Core (Low Risk, Slow Build)"},
            {"id": "pickem", "name": "PROD - Pickem", "former_names": ["Pickem legacy"]},
            {"id": "other", "name": "TEST - Other"},
        ],
    }


def observation(index=0, **updates):
    return {
        "_ingest_observation_id": f"obs-{index}",
        "_ingest_record_id": f"record-{index}",
        "_ingest_namespace": "test",
        "_ingest_is_current": True,
        "_ingest_disposition": "new",
        "_ingest_reason": "",
        "_ingest_timestamp_timezone": "unknown",
        "_ingest_execution_status": "real_money",
        "_ingest_execution_evidence": "user_assumption",
        "stake": "10",
        "bet_profit": "12",
        "potential_payout": "22",
        "status": "won",
        "sportsbook": "Book",
        "odds": "120",
        "bet_type": "positive_ev",
        "parlay_id": "",
        "created_at": "2026-01-01 12:00:00",
        "game_id": "game",
        "market_name": "Moneyline",
        "bet_name": "Away",
        "event_start_date": "2026-01-02 12:00:00",
        "event_name": "Event",
        "saved_filter_names": "PROD - Core (Low Risk, Slow Build)",
        "is_live_bet": "False",
        "is_free_bet": "False",
        "is_odds_boost": "False",
        **updates,
    }


def build(rows):
    cfg = config()
    tables = build_bet_facts(pd.DataFrame(rows), cfg, registry())
    tables["bet_facts"] = qualify_economics(tables["bet_facts"], cfg)
    return tables


def test_singles_preserve_profit_and_unknown_time():
    tables = build([observation()])
    bets = tables["bet_facts"]
    assert bets.iloc[0].reported_profit_usd == 12
    assert bets.iloc[0].modeled_profit_usd == 12
    assert bets.iloc[0].reported_roi == pytest.approx(1.2)
    assert bets.iloc[0].currency == "USD"
    assert bets.iloc[0].fee_inclusion == "unverified"
    assert not bets.iloc[0].fee_verified_net_roi_eligible
    assert pd.isna(bets.iloc[0].created_at_utc)
    assert len(tables["source_audit"]) == 1


def test_ticket_money_once_and_leg_status_not_ticket_status():
    tables = build(
        [
            observation(i, parlay_id="ticket", bet_type="parlay", status=status, bet_name=str(i))
            for i, status in enumerate(["won", "lost", "pending"])
        ]
    )
    bets, legs = tables["bet_facts"], tables["bet_legs"]
    assert len(bets) == 1 and len(legs) == 3
    assert bets.reported_stake_usd.sum() == 10
    assert bets.reported_profit_usd.sum() == 12
    assert bets.iloc[0].settlement_status == "ticket_settlement_unverified"
    assert not bets.iloc[0].reported_roi_eligible
    assert {"stake", "bet_profit", "reported_profit_usd", "potential_payout"}.isdisjoint(
        legs.columns
    )


def test_unresolved_legs_propagate_to_ticket():
    tables = build(
        [
            observation(0, parlay_id="ticket"),
            observation(
                1, parlay_id="ticket", _ingest_is_current=False, _ingest_disposition="unresolved"
            ),
        ]
    )
    bet = tables["bet_facts"].iloc[0]
    assert bet.unresolved_leg_observations == 1
    assert bet.observed_leg_count == 1
    assert len(tables["bet_legs"]) == 2
    assert not bet.reported_roi_eligible
    assert "unresolved_ticket_members" in json.loads(bet.economic_exclusions_json)


def test_conflicting_ticket_economics_not_arbitrarily_chosen():
    tables = build(
        [
            observation(0, parlay_id="ticket"),
            observation(1, parlay_id="ticket", stake="20", bet_profit="30"),
        ]
    )
    bet = tables["bet_facts"].iloc[0]
    assert pd.isna(bet.reported_stake_usd) and pd.isna(bet.reported_profit_usd)
    assert not bet.ticket_economics_consistent


def test_missing_profit_is_not_zero_and_empty_cohort_is_null():
    tables = build([observation(bet_profit="")])
    bets = tables["bet_facts"]
    assert pd.isna(bets.iloc[0].reported_profit_usd)
    summary = summarize_economics(bets).iloc[0]
    assert summary.eligible_count == 0
    assert pd.isna(summary.eligible_stake_usd) and pd.isna(summary.reported_roi)


def test_single_leg_ticket_is_not_assumed_complete():
    bet = build([observation(parlay_id="ticket")])["bet_facts"].iloc[0]
    assert bet.fact_grain == "ticket_candidate"
    assert bet.ticket_completeness == "unverified"
    assert not bet.reported_roi_eligible


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"status": "pending", "bet_profit": "12"}, "unresolved_or_excluded_settlement"),
        ({"status": "refunded", "bet_profit": "0"}, "unresolved_or_excluded_settlement"),
        ({"status": "cancelled", "bet_profit": "0"}, "unresolved_or_excluded_settlement"),
        ({"sportsbook": "Picks"}, "pickem_candidate_leg"),
        ({"sportsbook": "Unlisted"}, "unknown_product"),
        ({"saved_filter_names": "Pickem legacy"}, "pickem_filter_evidence"),
        ({"bet_type": "parlay"}, "unsupported_bet_type"),
        ({"bet_type": "middle"}, "unsupported_bet_type"),
        ({"bet_type": "arbitrage"}, "unsupported_bet_type"),
        ({"is_free_bet": "True"}, "unsupported_or_unknown_is_free_bet"),
        ({"is_odds_boost": ""}, "unsupported_or_unknown_is_odds_boost"),
        ({"stake": "0"}, "invalid_stake"),
        ({"stake": "inf"}, "invalid_stake"),
        ({"bet_profit": "inf"}, "missing_or_nonfinite_profit"),
        ({"odds": "0"}, "invalid_odds"),
        ({"bet_profit": "10"}, "reported_profit_model_mismatch"),
    ],
)
def test_ineligible_rows_retained(updates, reason):
    bets = build([observation(**updates)])["bet_facts"]
    assert len(bets) == 1
    assert not bets.iloc[0].reported_roi_eligible
    assert reason in json.loads(bets.iloc[0].economic_exclusions_json)


@pytest.mark.parametrize(
    "odds,status,profit",
    [
        (120, "won", 12),
        (-200, "won", 5),
        (120, "lost", -10),
        (-200, "half_won", 2.5),
        (120, "half_won", 6),
        (120, "half_lost", -5),
    ],
)
def test_fixed_odds_settlements(odds, status, profit):
    bet = build([observation(odds=str(odds), status=status, bet_profit=str(profit))])[
        "bet_facts"
    ].iloc[0]
    assert bet.reported_roi_eligible
    assert bet.modeled_profit_usd == pytest.approx(profit)


def test_exchange_reported_profit_never_charged_estimated_fees():
    bet = build([observation(sportsbook="Exchange", bet_profit="11.5")])["bet_facts"].iloc[0]
    assert bet.reported_roi_eligible
    assert bet.reported_profit_usd == 11.5
    assert pd.isna(bet.modeled_profit_usd)
    assert not bet.fee_verified_net_roi_eligible


def test_history_does_not_add_wagers():
    tables = build(
        [
            observation(0),
            observation(
                1,
                _ingest_record_id="record-0",
                _ingest_is_current=False,
                _ingest_disposition="repeated_observation",
            ),
        ]
    )
    assert len(tables["bet_facts"]) == 1
    assert len(tables["source_audit"]) == 2
    assert tables["source_audit"].bet_id.nunique() == 1


def test_ticket_namespaces_do_not_merge():
    tables = build(
        [
            observation(0, parlay_id="ticket"),
            observation(1, parlay_id="ticket", _ingest_namespace="other"),
        ]
    )
    assert len(tables["bet_facts"]) == 2


def test_confirmed_timezone_converts_but_does_not_change_source():
    bet = build([observation(_ingest_timestamp_timezone="America/New_York")])["bet_facts"].iloc[0]
    assert bet.created_at == "2026-01-01 12:00:00"
    assert bet.created_at_utc == pd.Timestamp("2026-01-01 17:00:00Z")


def test_duplicate_current_identity_rejected():
    with pytest.raises(ValueError, match="unique"):
        build([observation(0), observation(1, _ingest_record_id="record-0")])


def test_roi_uses_same_eligible_population_for_profit_and_stake():
    bets = build(
        [
            observation(0),
            observation(1, status="lost", stake="20", bet_profit="-20"),
            observation(2, status="pending", stake="1000", bet_profit="999"),
        ]
    )["bet_facts"]
    summary = summarize_economics(bets).iloc[0]
    assert summary.eligible_count == 2
    assert summary.eligible_stake_usd == 30
    assert summary.eligible_reported_profit_usd == -8
    assert summary.reported_roi == pytest.approx(-8 / 30)


def test_empty_sources_remain_empty_with_valid_schema():
    empty = pd.DataFrame([observation()]).iloc[:0]
    tables = build_bet_facts(empty, config(), registry())
    assert tables["bet_facts"].empty
    assert qualify_economics(tables["bet_facts"], config()).empty
    assert tables["source_audit"].empty


def test_unresolved_only_ticket_preserves_audit_without_economics():
    tables = build(
        [
            observation(
                parlay_id="ticket", _ingest_is_current=False, _ingest_disposition="unresolved"
            )
        ]
    )
    bet = tables["bet_facts"].iloc[0]
    assert bet.unresolved_leg_observations == 1
    assert pd.isna(bet.reported_profit_usd)
    assert len(tables["bet_legs"]) == 1
    assert not bet.reported_roi_eligible


def test_ticket_conflicting_execution_and_time_are_unknown():
    tables = build(
        [
            observation(0, parlay_id="ticket"),
            observation(
                1,
                parlay_id="ticket",
                _ingest_execution_status="paper",
                created_at="2026-01-01 12:01:00",
            ),
        ]
    )
    bet = tables["bet_facts"].iloc[0]
    assert bet.execution_status == "unknown"
    assert bet.created_at == ""
    assert pd.isna(bet.created_at_utc)


def test_nonfinite_ticket_amount_not_consistent():
    bet = build([observation(parlay_id="ticket", bet_profit="inf")])["bet_facts"].iloc[0]
    assert pd.isna(bet.reported_profit_usd)
    assert not bet.ticket_economics_consistent


def test_actual_cash_mode_does_not_change_economic_eligibility():
    bets = build(
        [
            observation(0),
            observation(1, _ingest_execution_status="paper"),
            observation(2, _ingest_execution_status="unknown"),
        ]
    )["bet_facts"]
    assert bets.reported_roi_eligible.all()
    summary = summarize_economics(bets)
    assert len(summary) == 3
    assert set(summary.execution_status) == {"real_money", "paper", "unknown"}
