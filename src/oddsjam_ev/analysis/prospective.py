"""
prospective.py
===============================================================================
Purpose
-------
Bind prospective observations and paper outcomes to frozen definition versions.

What this module provides
-------------------------
Typed decision-time predicates, strict capture validation and per-version review data.

Design Notes
------------
Manual evidence is an operator attestation, not a verified fill or causal experiment.
Parent, subgroup and complement results overlap and must never be added together.
===============================================================================
"""

import json
import math
from decimal import Decimal, InvalidOperation

import pandas as pd

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.io.source_csv import fingerprint


def matches(terms: list[dict], values: dict) -> bool:
    """Evaluate an AND predicate with explicit missingness and decimal endpoints.

    Parameters
    ----------
    terms : list[dict]
        Published exact or numeric predicate terms.
    values : dict
        Decision-time values, numeric EV in percentage points.

    Returns
    -------
    bool
        Membership. Missing/unsupported evidence raises rather than becoming complement.
    """
    results = []
    for term in terms:
        dimension, operator = term["dimension"], term["operator"]
        if dimension not in values or values[dimension] is None:
            raise ValueError("Missing decision-time feature: " + dimension)
        value = values[dimension]
        if not isinstance(value, (str, int, float, Decimal)) or pd.isna(value):
            raise ValueError("Scalar nonmissing decision-time feature required")
        if operator == "==":
            results.append(value == term["value"])
            continue
        if operator not in {">", ">=", "<", "<="}:
            raise ValueError("Unsupported typed predicate operator")
        try:
            if isinstance(value, bool):
                raise ValueError("Boolean is not numeric evidence")
            left, right = Decimal(str(value)), Decimal(str(term["endpoint"]))
            if not left.is_finite() or not right.is_finite():
                raise ValueError("Finite numeric predicate evidence required")
        except (InvalidOperation, KeyError) as exc:
            raise ValueError("Numeric predicate evidence required") from exc
        results.append(
            {">": left > right, ">=": left >= right, "<": left < right, "<=": left <= right}[
                operator
            ]
        )
    return all(results)


def validate_bound_protocol(protocol: dict) -> None:
    """Validate an operational protocol against its embedded immutable definition.

    Parameters
    ----------
    protocol : dict
        Definition-bound operational contract, not a mutable copied plan.

    Returns
    -------
    None
        Raises for changes to the definition, baseline, predicate or paper funding.
    """
    from oddsjam_ev.analysis.paper_definitions import validate_definition

    expected = {
        "contract_version",
        "definition",
        "definition_publication_id",
        "definition_manifest_sha256",
        "experiment_type",
        "filter_id",
        "filter_config",
        "predicate",
        "planned_start_utc",
        "planned_end_utc",
        "funding",
        "max_funded_stake_usd",
        "sampling_rule",
    }
    if set(protocol) != expected or protocol["contract_version"] != 2:
        raise ValueError("Exact definition-bound protocol fields required")
    definition = dict(protocol["definition"])
    ident = definition.pop("version_id", None)
    if not ident or fingerprint(definition) != ident:
        raise ValueError("Definition version hash mismatch")
    plan = definition["protocol"]
    validate_definition(plan)
    if definition.get("status") != "definition_only_not_launched":
        raise ValueError("Frozen definition required")
    pinned = {
        "filter_config": plan["baseline"]["filter_config"],
        "predicate": definition["source"]["exact_predicate"],
        **{
            k: plan[k]
            for k in ("planned_start_utc", "planned_end_utc", "funding", "max_funded_stake_usd")
        },
    }
    if any(protocol[k] != v for k, v in pinned.items()):
        raise ValueError("Operational protocol differs from definition")
    if not isinstance(protocol["predicate"], list) or not protocol["predicate"]:
        raise ValueError("Typed candidate predicate required")
    for term in protocol["predicate"]:
        if term.get("operator") not in {"==", ">", ">=", "<", "<="}:
            raise ValueError("Unsupported typed predicate operator")
        if term["operator"] != "==" and term.get("units") != "percentage_points":
            raise ValueError("Unsupported numeric predicate units")
    for key in (
        "definition_publication_id",
        "definition_manifest_sha256",
        "filter_id",
        "sampling_rule",
    ):
        if not isinstance(protocol[key], str) or not protocol[key].strip():
            raise ValueError(key + " must be explicit")
    if protocol["experiment_type"] not in {"signal_screening", "execution_verification"}:
        raise ValueError("Unknown experiment type")


def prepare_event(item: dict, states: dict, kind: str, payload: dict, now: str) -> dict:
    """Validate and normalize one event before append and again during replay.

    Parameters
    ----------
    item, states : dict
        Current experiment and full register state, used for cross-version identity checks.
    kind, payload, now : str, dict, str
        Explicit command and system recording time; never caller-backdated recording.

    Returns
    -------
    dict
        Version-bound payload with derived membership and paper economics.
    """
    p = item["protocol"]
    definition = p["definition"]
    plan = definition["protocol"]
    out = dict(payload)
    if out.get("definition_version_id", definition["version_id"]) != definition["version_id"]:
        raise ValueError("Cross-version event rejected")
    out["definition_version_id"] = definition["version_id"]
    if kind == "TRANSITION":
        target = out.get("to_status")
        if not out.get("evidence_reference"):
            raise ValueError("Explicit lifecycle evidence required")
        if target == "RUNNING":
            launched = aware(out.get("launched_at_utc"))
            if (
                out.get("filter_config") != p["filter_config"]
                or out.get("filter_id") != p["filter_id"]
                or not aware(p["planned_start_utc"])
                <= launched
                <= aware(now)
                < aware(p["planned_end_utc"])
            ):
                raise ValueError(
                    "Exact launch/resume configuration and in-window evidence required"
                )
            # Capture cannot precede the recorded confirmation, even after a resume.
        if target == "AWAITING_OUTCOMES":
            if out.get("ended_at_utc") is None:
                raise ValueError("Actual enrollment-end evidence required")
            ended = aware(out.get("ended_at_utc"))
            if not item["launch"] or not aware(item["launch"]["recorded_at"]) <= ended <= aware(
                now
            ):
                raise ValueError("Actual enrollment-end evidence required")
        if target == "AWAITING_OUTCOMES" and any(
            aware(e["recorded_at"]) > aware(out["ended_at_utc"])
            for e in item["enrollments"].values()
        ):
            raise ValueError("Actual end cannot precede captured observations")
        if target == "CLOSED":
            if not out.get("review_decision_reference") or not out.get("review_decision_event_id"):
                raise ValueError("Explicit published reviewed-decision evidence required to close")
    elif kind == "ENROLL":
        if item["status"] != "RUNNING":
            raise ValueError("Enrollment is closed or paused")
        running = [
            e
            for e in item["events"]
            if e["kind"] == "TRANSITION" and e["payload"]["to_status"] == "RUNNING"
        ][-1]
        observed, start, recorded = map(
            aware, (out.get("observed_at_utc"), out.get("event_start_utc"), now)
        )
        if not aware(
            running["recorded_at"]
        ) <= observed <= recorded < start or not recorded < aware(p["planned_end_utc"]):
            raise ValueError("Retrospective, future or out-of-window enrollment rejected")
        for key in (
            "enrollment_id",
            "observation_key",
            "event_namespace",
            "event_id",
            "evidence_reference",
        ):
            if not isinstance(out.get(key), str) or not out[key].strip():
                raise ValueError("Stable observation/event identity and evidence required")
        for other in states.values():
            other_definition = other["protocol"].get("definition", {})
            if other_definition.get("test_key") == definition["test_key"]:
                if any(
                    e["payload"]["observation_key"] == out["observation_key"]
                    for e in other["enrollments"].values()
                ):
                    raise ValueError("Duplicate observation across this test's versions")
        identity = fingerprint(
            {
                k: out.get(k)
                for k in ("event_namespace", "event_id", "observed_at_utc", "scanner_quote")
            }
        )
        if any(
            e["payload"].get("capture_identity_sha256") == identity
            for other in states.values()
            if other["protocol"].get("definition", {}).get("test_key") == definition["test_key"]
            for e in other["enrollments"].values()
        ):
            raise ValueError("Duplicate decision-time quote identity across test versions")
        out["capture_identity_sha256"] = identity
        if out["enrollment_id"] in item["enrollments"]:
            raise ValueError("Duplicate enrollment")
        if out.get("baseline_config_sha256") != fingerprint(p["filter_config"]) or not out.get(
            "parent_membership_reference"
        ):
            raise ValueError("Pinned parent membership evidence required")
        features = out.get("feature_values")
        if not isinstance(features, dict):
            raise ValueError("Decision-time feature values required")
        for term in p["predicate"]:
            name = term["dimension"]
            knowledge = out.get("feature_available_at_utc", {}).get(name)
            if knowledge is None or aware(knowledge) > observed:
                raise ValueError("Feature must be known at decision time")
        out["arm"] = "subgroup" if matches(p["predicate"], features) else "complement"
        quote = out.get("scanner_quote", {})
        try:
            odds = float(quote.get("odds"))
            if (
                isinstance(quote.get("odds"), bool)
                or not math.isfinite(odds)
                or abs(odds) < 100
                or not quote.get("selection")
                or quote.get("line") is None
            ):
                raise ValueError("Valid scanner selection, line and American odds required")
        except (TypeError, ValueError) as exc:
            raise ValueError("Valid scanner quote required") from exc
        out["timing_status"] = "prospective"
        out["intended_mode"] = "paper"
        out["availability_status"] = "not_checked"
    elif kind == "SETTLE":
        enrollment = item["enrollments"].get(out.get("enrollment_id"))
        if enrollment is None or item["status"] in {"CANCELLED", "CLOSED"}:
            raise ValueError("Existing open experiment enrollment required")
        if out.get("result") not in {"won", "lost", "push", "void", "pending"} or not out.get(
            "evidence_reference"
        ):
            raise ValueError("Explicit result and settlement evidence required")
        available = aware(out.get("outcome_available_at_utc"))
        if not aware(enrollment["payload"]["event_start_utc"]) <= available <= aware(now):
            raise ValueError("Outcome must follow event start and be known by recording time")
        prior = item["settlements"].get(out["enrollment_id"])
        if prior and (
            out.get("supersedes_event_id") != prior["event_id"] or not out.get("correction_reason")
        ):
            raise ValueError("Settlement revisions require explicit prior identity and reason")
        if not prior and out.get("supersedes_event_id") is not None:
            raise ValueError("No prior settlement to supersede")
        odds = float(enrollment["payload"]["scanner_quote"]["odds"])
        unit = {
            "won": odds / 100 if odds > 0 else 100 / abs(odds),
            "lost": -1.0,
            "push": 0.0,
            "void": 0.0,
            "pending": None,
        }[out["result"]]
        out["unit_return"] = unit
        out["paper_profit_usd"] = None if unit is None else unit * plan["sizing"]["stake_usd"]
        out["settlement_basis"] = "operator_attested_paper_outcome_at_original_quote"
    elif kind in {"CHECK", "LINK"}:
        if out.get("enrollment_id") not in item["enrollments"]:
            raise ValueError("Unknown enrollment")
        # Existing check/link validators additionally enforce quote and identity evidence.
    elif kind == "REVIEW":
        scheduled = out.get("scheduled_review_utc")
        if scheduled not in plan["review_dates_utc"] or aware(scheduled) > aware(now):
            raise ValueError("A due predeclared review date is required")
        if not out.get("evidence_reference") or not out.get("notes"):
            raise ValueError("Explicit review evidence and notes required")
        prior = [e for e in item["reviews"] if e["payload"]["scheduled_review_utc"] == scheduled]
        if prior:
            if out.get("supersedes_review_event_id") != prior[-1]["event_id"] or not out.get(
                "correction_reason"
            ):
                raise ValueError(
                    "Scheduled review already recorded; explicit latest review amendment and reason required"
                )
        elif out.get("supersedes_review_event_id") is not None:
            raise ValueError("No prior scheduled review to supersede")
    elif kind != "DEVIATION":
        raise ValueError("Unknown prospective event kind")
    return out


def build_prospective_tables(events: list[dict], as_of: str) -> dict[str, pd.DataFrame]:
    """Publish per-version cohorts, availability, economics and scheduled review status.

    Parameters
    ----------
    events : list[dict]
        Integrity-checked system-known events, including legacy experiments.
    as_of : str
        Explicit report knowledge cutoff.

    Returns
    -------
    dict[str,pandas.DataFrame]
        Observation, metric and review tables; never pooled across definition versions.
    """
    from oddsjam_ev.analysis.experiments import replay_experiments

    states = replay_experiments([e for e in events if aware(e["recorded_at"]) <= aware(as_of)])
    observations, metrics, reviews = [], [], []
    all_groups = {}
    for ident, item in states.items():
        if item["protocol"].get("contract_version") != 2:
            continue
        for event in item["enrollments"].values():
            e = event["payload"]
            all_groups.setdefault((e["event_namespace"], e["event_id"]), set()).add(ident)
    for ident, item in states.items():
        p = item["protocol"]
        if p.get("contract_version") != 2:
            continue
        d, plan = p["definition"], p["definition"]["protocol"]
        common = {
            "experiment_id": ident,
            "definition_version_id": d["version_id"],
            "test_key": d["test_key"],
            "version": d["version"],
            "candidate_id": d["source"]["candidate_id"],
        }
        rows = []
        for key, event in item["enrollments"].items():
            e = event["payload"]
            outcome = item["settlements"].get(key)
            settlement = outcome["payload"] if outcome else {}
            checks = item["checks"].get(key, [])
            check = checks[-1]["payload"] if checks else {}
            available = bool(
                check.get("original_quote_matched")
                and aware(checks[-1]["recorded_at"]) < aware(e["event_start_utc"])
                and aware(check["checked_at_utc"]) < aware(e["event_start_utc"])
                and float(check.get("available_stake_usd", 0)) >= plan["sizing"]["stake_usd"]
            )
            row = {
                **common,
                "enrollment_id": key,
                "observation_key": e["observation_key"],
                "event_namespace": e["event_namespace"],
                "event_id": e["event_id"],
                "arm": e["arm"],
                "observed_at_utc": e["observed_at_utc"],
                "result": settlement.get("result", "pending"),
                "unit_return": settlement.get("unit_return"),
                "paper_profit_usd": settlement.get("paper_profit_usd"),
                "paper_stake_usd": plan["sizing"]["stake_usd"],
                "availability_status": check.get("status", "not_checked"),
                "original_quote_available": available,
                "source_link_status": "operator_linked_not_independently_verified"
                if key in item["links"]
                else "unlinked",
                "settlement_event_id": outcome["event_id"] if outcome else None,
                "shared_event_experiments": len(all_groups[(e["event_namespace"], e["event_id"])]),
            }
            rows.append(row)
        observations.extend(rows)
        arm_support = []
        for arm in ("parent", "subgroup", "complement"):
            selected = [r for r in rows if arm == "parent" or r["arm"] == arm]
            for cohort in ("all_captured_hypothetical", "original_quote_checked_available"):
                cohort_rows = (
                    selected
                    if cohort == "all_captured_hypothetical"
                    else [r for r in selected if r["original_quote_available"]]
                )
                settled = [
                    r for r in cohort_rows if r["unit_return"] is not None and r["result"] != "void"
                ]
                n = len(settled)
                groups = len({(r["event_namespace"], r["event_id"]) for r in settled})
                stake = sum(r["paper_stake_usd"] for r in settled)
                profit = sum(r["paper_profit_usd"] for r in settled)
                supported = (
                    n >= plan["min_wagers_per_arm"] and groups >= plan["min_event_groups_per_arm"]
                )
                metrics.append(
                    {
                        **common,
                        "arm": arm,
                        "cohort": cohort,
                        "captured": len(cohort_rows),
                        "settled_wagers": n,
                        "event_groups": groups,
                        "pending": sum(r["result"] == "pending" for r in cohort_rows),
                        "void": sum(r["result"] == "void" for r in cohort_rows),
                        "paper_stake_usd": stake,
                        "paper_profit_usd": profit,
                        "stake_weighted_roi": profit / stake if stake else None,
                        "equal_stake_mean_return": sum(r["unit_return"] for r in settled) / n
                        if n
                        else None,
                        "support_met": supported,
                        "promotion_eligible": False,
                    }
                )
                if arm != "parent" and cohort == "all_captured_hypothetical":
                    arm_support.append(supported)
        ended = [
            e
            for e in item["events"]
            if e["kind"] == "TRANSITION" and e["payload"]["to_status"] == "AWAITING_OUTCOMES"
        ]
        blockers = []
        if not item["launch"]:
            blockers.append("no_confirmed_launch")
        if not ended:
            blockers.append("no_confirmed_enrollment_end")
        elif aware(ended[-1]["payload"]["ended_at_utc"]) < aware(plan["planned_end_utc"]):
            blockers.append("enrollment_ended_early")
        if not all(arm_support):
            blockers.append("insufficient_arm_support")
        if any(r["result"] == "pending" for r in rows):
            blockers.append("pending_outcomes")
        if any(not r["original_quote_available"] for r in rows):
            blockers.append("availability_incomplete_or_failed")
        if any(e["kind"] == "DEVIATION" for e in item["events"]):
            blockers.append("protocol_deviation")
        for scheduled in plan["review_dates_utc"]:
            recorded = [
                e for e in item["reviews"] if e["payload"]["scheduled_review_utc"] == scheduled
            ]
            reviews.append(
                {
                    **common,
                    "status": item["status"],
                    "scheduled_review_utc": scheduled,
                    "review_due": aware(scheduled) <= aware(as_of),
                    "review_recorded": bool(recorded),
                    "review_event_id": recorded[-1]["event_id"] if recorded else None,
                    "blockers_json": json.dumps(blockers),
                    "readiness": "eligible_for_human_readiness_review"
                    if not blockers
                    else "inconclusive_or_incomplete",
                    "settlement_followup_elapsed": aware(as_of)
                    >= aware(plan["planned_end_utc"])
                    + pd.Timedelta(days=plan["settlement_followup_days"]),
                    "promotion_eligible": False,
                }
            )
    columns = {
        "prospective_observations": [
            "experiment_id",
            "definition_version_id",
            "test_key",
            "version",
            "candidate_id",
            "enrollment_id",
            "observation_key",
            "event_namespace",
            "event_id",
            "arm",
            "observed_at_utc",
            "result",
            "unit_return",
            "paper_profit_usd",
            "paper_stake_usd",
            "availability_status",
            "original_quote_available",
            "source_link_status",
            "settlement_event_id",
            "shared_event_experiments",
        ],
        "prospective_metrics": [
            "experiment_id",
            "definition_version_id",
            "test_key",
            "version",
            "candidate_id",
            "arm",
            "cohort",
            "captured",
            "settled_wagers",
            "event_groups",
            "pending",
            "void",
            "paper_stake_usd",
            "paper_profit_usd",
            "stake_weighted_roi",
            "equal_stake_mean_return",
            "support_met",
            "promotion_eligible",
        ],
        "prospective_reviews": [
            "experiment_id",
            "definition_version_id",
            "test_key",
            "version",
            "candidate_id",
            "status",
            "scheduled_review_utc",
            "review_due",
            "review_recorded",
            "review_event_id",
            "blockers_json",
            "readiness",
            "settlement_followup_elapsed",
            "promotion_eligible",
        ],
    }
    return {
        name: pd.DataFrame(rows, columns=columns[name])
        for name, rows in zip(columns, (observations, metrics, reviews), strict=True)
    }
