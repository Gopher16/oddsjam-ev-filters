"""
payouts.py
===============================================================================
Purpose
-------
Qualify source-reported economics without fabricating fees or ticket payouts.

What this module provides
-------------------------
Reported-ROI eligibility, constrained fixed-odds checks, and cohort summaries.

Design Notes
------------
Reported profit is never overwritten. Exchange returns remain fee-unverified.
Refunds/cancellations, candidate legs, and ambiguous products are excluded from ROI.
===============================================================================
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.metrics.odds import american_to_multiplier

SUPPORTED_SETTLEMENTS = {"won", "lost", "half_won", "half_lost"}


def qualify_economics(bets: pd.DataFrame, cfg: CanonicalConfig) -> pd.DataFrame:
    """Preserve amounts and qualify explicit reported-return cohorts.

    Parameters
    ----------
    bets : pandas.DataFrame
        Canonical wager candidates, never leg-grain rows.
    cfg : CanonicalConfig
        Explicit USD and eligibility policy.

    Returns
    -------
    pandas.DataFrame
        Facts with reported ROI eligibility, exclusions, and separate model checks.
    """
    out = bets.copy()
    modeled = []
    reasons = []
    supported_model = []
    for row in out.to_dict("records"):
        why = []
        kind, status = row["product"], row["settlement_status"]
        stake, profit = row["reported_stake_usd"], row["reported_profit_usd"]
        if row["fact_grain"] == "ticket_candidate":
            why.append("ticket_payout_and_completeness_unverified")
            if row["unresolved_leg_observations"]:
                why.append("unresolved_ticket_members")
            if not row["ticket_economics_consistent"]:
                why.append("ticket_economics_missing_or_conflicting")
            supported_model.append(False)
            modeled.append(np.nan)
            reasons.append(json.dumps(sorted(set(why))))
            continue
        if kind not in {"fixed_odds", "exchange"}:
            why.append("pickem_candidate_leg" if kind == "pickem" else "unknown_product")
        if row["pickem_label_guard"]:
            why.append("pickem_filter_evidence")
        if row["bet_type"] not in {"positive_ev", "normal"}:
            why.append("unsupported_bet_type")
        if status not in SUPPORTED_SETTLEMENTS:
            why.append("unresolved_or_excluded_settlement")
        if not np.isfinite(stake) or stake <= 0:
            why.append("invalid_stake")
        if not np.isfinite(profit):
            why.append("missing_or_nonfinite_profit")
        # Blank/unknown flags do not establish ordinary, unpromoted pre-match bets.
        for flag in ("is_live_bet", "is_free_bet", "is_odds_boost"):
            if str(row[flag]).strip().lower() != "false":
                why.append("unsupported_or_unknown_" + flag)
        odds = pd.to_numeric(row["odds"], errors="coerce")
        if not np.isfinite(odds) or odds == 0:
            why.append("invalid_odds")
        can_model = kind == "fixed_odds" and not why
        expected = np.nan
        if can_model:
            win_profit = stake * american_to_multiplier(odds)
            expected = {
                "won": win_profit,
                "lost": -stake,
                "half_won": win_profit / 2,
                "half_lost": -stake / 2,
            }[status]
            if abs(profit - expected) > cfg.profit_tolerance_usd + 1e-9:
                why.append("reported_profit_model_mismatch")
        supported_model.append(can_model)
        modeled.append(expected)
        reasons.append(json.dumps(sorted(set(why))))
    out["currency"] = cfg.currency
    out["currency_evidence"] = cfg.currency_evidence
    out["fee_inclusion"] = cfg.fee_inclusion
    out["fee_verified_net_roi_eligible"] = False
    out["modeled_profit_usd"] = modeled
    out["fixed_odds_model_supported"] = supported_model
    out["economic_exclusions_json"] = reasons
    out["reported_roi_eligible"] = out.economic_exclusions_json.eq("[]")
    out["reported_roi"] = out.reported_profit_usd.div(out.reported_stake_usd).where(
        out.reported_roi_eligible
    )
    return out


def summarize_economics(bets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matching eligible profit/stake populations by evidence cohort.

    Parameters
    ----------
    bets : pandas.DataFrame
        Qualified canonical bet facts.

    Returns
    -------
    pandas.DataFrame
        USD reported-return diagnostics, separated by product and execution mode.
        Empty eligible populations have null totals/ROI, never misleading zeros.
    """
    rows = []
    for (product, execution), group in bets.groupby(["product", "execution_status"], sort=True):
        eligible = group[group.reported_roi_eligible]
        stake = eligible.reported_stake_usd.sum(min_count=1)
        profit = eligible.reported_profit_usd.sum(min_count=1)
        rows.append(
            [
                product,
                execution,
                len(group),
                len(eligible),
                stake,
                profit,
                profit / stake if pd.notna(stake) and stake > 0 else np.nan,
                "USD",
                "unverified",
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "product",
            "execution_status",
            "candidate_count",
            "eligible_count",
            "eligible_stake_usd",
            "eligible_reported_profit_usd",
            "reported_roi",
            "currency",
            "fee_inclusion",
        ],
    )
