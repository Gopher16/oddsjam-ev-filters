"""
experiment_outcomes.py
===============================================================================
Purpose
-------
Link explicitly enrolled observations to pinned canonical outcome facts.

What this module provides
-------------------------
Prospective paper metrics, mode conflicts, availability summaries and readiness audits.

Design Notes
------------
Unchecked scanner signals remain useful hypothetical evidence, not executable bets.
Core Exchange results describe selected funded trades, not all scanner opportunities.
===============================================================================
"""

import pandas as pd

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.analysis.experiments import replay_experiments
from oddsjam_ev.io.reconcile import classify_execution


def build_experiment_outcomes(
    events: list[dict], tables: dict, as_of: str
) -> dict[str, pd.DataFrame]:
    """Build reports without inferring membership from names or outcomes.

    Parameters
    ----------
    events : list[dict]
        Verified events pinned by recording cutoff.
    tables : dict
        Verified canonical bundle tables.
    as_of : str
        Explicit system-knowledge cutoff, not presumed settlement time.

    Returns
    -------
    dict[str,pandas.DataFrame]
        Protocols, events, enrollments, links, outcomes, metrics, checks and audit.
    """
    import json

    state = replay_experiments(events)
    now = aware(as_of)
    protocols = []
    enrollments = []
    links = []
    checks = []
    outcomes = []
    audits = []
    metrics = []
    availability_rows = []
    bets = tables["bet_facts"]
    sources = tables["source_features"]
    for ident, item in sorted(state.items()):
        protocol = item["protocol"]
        launch = item["launch"]
        protocols.append(
            {
                "experiment_id": ident,
                "status": item["status"],
                "protocol_sha256": item["registration"]["payload"]["protocol_sha256"],
                "protocol_json": json.dumps(protocol, sort_keys=True),
                "experiment_type": protocol["experiment_type"],
                "registered_at": item["registration"]["recorded_at"],
            }
        )
        linked_bets = {}
        for enrollment_id, enrollment in sorted(item["enrollments"].items()):
            payload = enrollment["payload"]
            link = item["links"].get(enrollment_id)
            enrollments.append(
                {
                    "experiment_id": ident,
                    "enrollment_id": enrollment_id,
                    "recorded_at": enrollment["recorded_at"],
                    "timing_status": payload["timing_status"],
                    "intended_mode": "paper",
                    "feature_values_json": json.dumps(payload["feature_values"], sort_keys=True),
                }
            )
            records = []
            if link:
                records = (
                    sources.loc[
                        sources._ingest_record_id.eq(link["payload"]["source_record_id"]), "bet_id"
                    ]
                    .unique()
                    .tolist()
                )
                links.append(
                    {
                        "experiment_id": ident,
                        "enrollment_id": enrollment_id,
                        "source_record_id": link["payload"]["source_record_id"],
                        "link_event_id": link["event_id"],
                        "resolution": "resolved_current"
                        if len(records) == 1
                        else "unresolved_in_snapshot",
                    }
                )
            reason = []
            bet = None
            mode = "unknown"
            source_mode = "unknown"
            if len(records) == 1:
                selected = bets.loc[bets.bet_id.eq(records[0])]
                if len(selected) == 1:
                    bet = selected.iloc[0]
            if bet is None:
                reason.append("unlinked_or_unresolved_identity")
            else:
                source_mode = bet.execution_status
                # Apply the current declared assumption independently, preserving pinned source mode.
                mode = classify_execution(
                    str(bet.saved_filter_names or ""), str(bet.created_at or "")
                )
                if mode != "paper":
                    reason.append("funded_or_unknown_mode_conflict")
                if bet.bet_id in linked_bets:
                    reason.append("duplicate_canonical_wager")
                linked_bets[bet.bet_id] = enrollment_id
                quote = payload["scanner_quote"]
                try:
                    if str(bet.bet_name) != quote["selection"] or float(bet.odds) != float(
                        quote["odds"]
                    ):
                        reason.append("source_quote_conflicts_with_enrollment")
                except (TypeError, ValueError):
                    reason.append("source_quote_unresolved")
                if not bet.reported_roi_eligible:
                    reason.append("canonical_economics_excluded")
                if (
                    pd.notna(bet.created_at_utc)
                    and launch
                    and not aware(launch["payload"]["launched_at_utc"])
                    <= aware(bet.created_at_utc)
                    < aware(protocol["planned_end_utc"])
                ):
                    reason.append("source_time_outside_window")
            if payload["timing_status"] != "prospective":
                reason.append("enrollment_not_verified_prospective")
            history = item["checks"].get(enrollment_id, [])
            check = history[-1]["payload"] if history else None
            for event in history:
                checks.append(
                    {
                        "experiment_id": ident,
                        "enrollment_id": enrollment_id,
                        "check_event_id": event["event_id"],
                        "status": event["payload"]["status"],
                        "checked_at_utc": event["payload"]["checked_at_utc"],
                        "check_delay_seconds": event["payload"]["check_delay_seconds"],
                        "reason": event["payload"]["reason"],
                        "payload_json": json.dumps(event["payload"], sort_keys=True),
                    }
                )
            availability = check["status"] if check else "not_checked"
            if (
                check
                and check["status"] == "checked_available"
                and not check["original_quote_matched"]
            ):
                availability = "checked_available_changed_quote"
            if check and payload["event_start_utc"] is None:
                availability = "inconclusive_timing_unknown"
            elif check and aware(check["checked_at_utc"]) >= aware(payload["event_start_utc"]):
                availability = "inconclusive_after_event_start"
            outcomes.append(
                {
                    "experiment_id": ident,
                    "enrollment_id": enrollment_id,
                    "bet_id": bet.bet_id if bet is not None else None,
                    "game_id": bet.game_id if bet is not None else None,
                    "namespace": bet.namespace if bet is not None else None,
                    "source_execution_status": source_mode,
                    "assumed_execution_status": mode,
                    "timing_status": payload["timing_status"],
                    "availability_status": availability,
                    "prospective_paper_roi_eligible": not reason,
                    "reported_stake_usd": bet.reported_stake_usd if bet is not None else None,
                    "reported_profit_usd": bet.reported_profit_usd if bet is not None else None,
                    "settlement_status": bet.settlement_status if bet is not None else "unresolved",
                    "source_timezone_unknown": bet.timestamp_timezone == "unknown"
                    if bet is not None
                    else True,
                    "exclusions_json": json.dumps(reason),
                    "fees": "unverified",
                }
            )
        current = [row for row in outcomes if row["experiment_id"] == ident]
        eligible = [row for row in current if row["prospective_paper_roi_eligible"]]
        pending = sum(row["settlement_status"] in {"pending", "open"} for row in current)
        unresolved = sum(
            "unlinked_or_unresolved_identity" in row["exclusions_json"] for row in current
        )
        for availability in sorted({row["availability_status"] for row in current}):
            availability_rows.append(
                {
                    "experiment_id": ident,
                    "availability_status": availability,
                    "enrollment_count": sum(
                        row["availability_status"] == availability for row in current
                    ),
                    "denominator_enrollments": len(current),
                    "sampling_rule": protocol.get("sampling_rule", "not_a_verification_experiment"),
                    "represents_all_scanner_opportunities": False,
                }
            )
        deviation_count = sum(e["kind"] == "DEVIATION" for e in item["events"])
        early_enrollment_end = any(
            e["kind"] == "TRANSITION"
            and e["payload"]["to_status"] == "AWAITING_OUTCOMES"
            and aware(e["recorded_at"]) < aware(protocol["planned_end_utc"])
            for e in item["events"]
        )
        group_count = len(
            {(row["namespace"], row["game_id"]) for row in eligible if row["game_id"]}
        )
        duration_complete = now >= aware(protocol["planned_end_utc"])
        readiness = (
            "eligible_for_human_readiness_review"
            if duration_complete
            and not early_enrollment_end
            and not deviation_count
            and not pending
            and not unresolved
            and group_count >= protocol["min_event_groups"]
            else "incomplete_or_inconclusive"
        )
        audits.append(
            {
                "experiment_id": ident,
                "duration_complete": duration_complete,
                "deviation_count": deviation_count,
                "early_enrollment_end": early_enrollment_end,
                "event_group_target_met": group_count >= protocol["min_event_groups"],
                "pending_count": pending,
                "unresolved_count": unresolved,
                "excluded_count": len(current) - len(eligible),
                "settlement_followup_elapsed": now
                >= aware(protocol["planned_end_utc"])
                + pd.Timedelta(days=protocol["settlement_followup_days"]),
                "readiness": readiness,
                "prospective_validity_verified": False,
                "promotion_allowed": False,
            }
        )
        for observed_mode, availability in sorted(
            {(row["assumed_execution_status"], row["availability_status"]) for row in current}
        ):
            subset = [
                row
                for row in current
                if row["assumed_execution_status"] == observed_mode
                and row["availability_status"] == availability
            ]
            valid = [row for row in subset if row["prospective_paper_roi_eligible"]]
            subtotal = sum(row["reported_stake_usd"] for row in valid) if valid else None
            subprofit = sum(row["reported_profit_usd"] for row in valid) if valid else None
            metrics.append(
                {
                    "experiment_id": ident,
                    "execution_status": observed_mode,
                    "availability_status": availability,
                    "enrolled_count": len(subset),
                    "eligible_count": len(valid),
                    "eligible_stake_usd": subtotal,
                    "eligible_reported_profit_usd": subprofit,
                    "hypothetical_reported_roi": subprofit / subtotal
                    if subtotal and subtotal > 0
                    else None,
                    "checked_available_count": sum(
                        row["availability_status"] == "checked_available" for row in subset
                    ),
                    "not_checked_count": sum(
                        row["availability_status"] == "not_checked" for row in subset
                    ),
                    "funded_execution_authorized": False,
                    "currency": "USD",
                    "fee_inclusion": "unverified",
                }
            )

    def frame(rows, columns):
        return pd.DataFrame(rows, columns=columns)

    return {
        "experiment_availability": frame(
            availability_rows,
            [
                "experiment_id",
                "availability_status",
                "enrollment_count",
                "denominator_enrollments",
                "sampling_rule",
                "represents_all_scanner_opportunities",
            ],
        ),
        "experiment_protocols": frame(
            protocols,
            [
                "experiment_id",
                "status",
                "protocol_sha256",
                "protocol_json",
                "experiment_type",
                "registered_at",
            ],
        ),
        "experiment_events": frame(
            [
                {
                    "experiment_id": e["experiment_id"],
                    "event_id": e["event_id"],
                    "kind": e["kind"],
                    "recorded_at": e["recorded_at"],
                    "payload_json": json.dumps(e["payload"], sort_keys=True),
                }
                for e in events
            ],
            ["experiment_id", "event_id", "kind", "recorded_at", "payload_json"],
        ),
        "experiment_enrollments": frame(
            enrollments,
            [
                "experiment_id",
                "enrollment_id",
                "recorded_at",
                "timing_status",
                "intended_mode",
                "feature_values_json",
            ],
        ),
        "experiment_links": frame(
            links,
            ["experiment_id", "enrollment_id", "source_record_id", "link_event_id", "resolution"],
        ),
        "experiment_checks": frame(
            checks,
            [
                "experiment_id",
                "enrollment_id",
                "check_event_id",
                "status",
                "checked_at_utc",
                "check_delay_seconds",
                "reason",
                "payload_json",
            ],
        ),
        "experiment_outcomes": frame(
            outcomes,
            [
                "experiment_id",
                "enrollment_id",
                "bet_id",
                "game_id",
                "namespace",
                "source_execution_status",
                "assumed_execution_status",
                "timing_status",
                "availability_status",
                "prospective_paper_roi_eligible",
                "reported_stake_usd",
                "reported_profit_usd",
                "settlement_status",
                "source_timezone_unknown",
                "exclusions_json",
                "fees",
            ],
        ),
        "experiment_metrics": frame(
            metrics,
            [
                "experiment_id",
                "execution_status",
                "availability_status",
                "enrolled_count",
                "eligible_count",
                "eligible_stake_usd",
                "eligible_reported_profit_usd",
                "hypothetical_reported_roi",
                "checked_available_count",
                "not_checked_count",
                "funded_execution_authorized",
                "currency",
                "fee_inclusion",
            ],
        ),
        "experiment_audit": frame(
            audits,
            [
                "experiment_id",
                "duration_complete",
                "deviation_count",
                "early_enrollment_end",
                "event_group_target_met",
                "pending_count",
                "unresolved_count",
                "excluded_count",
                "settlement_followup_elapsed",
                "readiness",
                "prospective_validity_verified",
                "promotion_allowed",
            ],
        ),
    }
