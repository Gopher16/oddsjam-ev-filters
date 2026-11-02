"""
filter_lifecycle.py
===============================================================================
Purpose
-------
Keep human research decisions separate from approved and applied scanner changes.

What this module provides
-------------------------
Reviewed policy validation, complete configuration diffs and version-scoped monitoring.

Design Notes
------------
A descriptive paper policy cannot approve PROD changes. The separately reviewed
fixed-look policy permits bounded human pilot review only. Operator evidence is an
attestation, never an independently verified fill.
===============================================================================
"""

import json
import math

import pandas as pd

from oddsjam_ev.analysis.decision_config import CONTROLS, normalize_settings
from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.io.source_csv import fingerprint

ACTIONS = {"continue", "modify", "stop", "inconclusive"}
FEATURES = {"advertised_ev_percent", "american_odds", "time_to_event_minutes"}


def nonempty(value):
    """Require explicit nonblank human evidence rather than a truthy placeholder."""
    return isinstance(value, str) and bool(value.strip())


def finite(value):
    """Accept finite numeric scalars while rejecting booleans and missing values."""
    return type(value) in (int, float) and math.isfinite(value)


def validate_policy(policy: dict) -> None:
    """Require a declared descriptive review policy with explicit monitoring rules.

    Parameters
    ----------
    policy : dict
        Operator-reviewed policy. No production or error-control thresholds are inferred.

    Returns
    -------
    None
        Raises if a paper policy claims production authorization.
    """
    extra = {"promotion"} if policy.get("scope") == "bounded_prod_pilot_review" else set()
    if set(policy) - extra != {
        "policy_key",
        "version",
        "scope",
        "error_control",
        "support_interpretation",
        "review_rules",
        "monitoring",
        "evidence_reference",
    }:
        raise ValueError("Exact declared policy fields required")
    if policy["scope"] == "bounded_prod_pilot_review":
        from oddsjam_ev.analysis.promotion import validate_promotion_policy

        validate_promotion_policy(policy["promotion"])
    elif policy["scope"] != "descriptive_paper_review_only":
        raise ValueError("Production policy requires separate prospective policy approval")
    if type(policy["version"]) is not int or policy["version"] < 1:
        raise ValueError("Positive policy version required")
    for key in ("policy_key", "error_control", "support_interpretation", "evidence_reference"):
        if not nonempty(policy[key]):
            raise ValueError("Explicit review/error-control policy evidence required")
    if set(policy["review_rules"]) != ACTIONS or not all(
        nonempty(v) for v in policy["review_rules"].values()
    ):
        raise ValueError("Continue/modify/stop/inconclusive criteria required")
    m = policy["monitoring"]
    if set(m) != {
        "paper_stake_usd",
        "min_wagers",
        "min_event_groups",
        "reference_means",
        "absolute_alert_deltas",
    }:
        raise ValueError("Exact prospective monitoring rules required")
    if not finite(m["paper_stake_usd"]) or m["paper_stake_usd"] <= 0:
        raise ValueError("Positive predeclared paper monitoring stake required")
    if any(type(m[k]) is not int or m[k] < 1 for k in ("min_wagers", "min_event_groups")):
        raise ValueError("Positive monitoring support counts required")
    refs, deltas = m["reference_means"], m["absolute_alert_deltas"]
    if (
        not isinstance(refs, dict)
        or not isinstance(deltas, dict)
        or set(refs) != set(deltas)
        or set(refs) - FEATURES
    ):
        raise ValueError("Matched supported drift references and thresholds required")
    if not all(finite(v) for v in refs.values()) or not all(
        finite(v) and v > 0 for v in deltas.values()
    ):
        raise ValueError("Finite reference means and positive drift thresholds required")


def settings_snapshot(settings: dict, reference: str, recorded: str) -> dict:
    """Normalize all scanner controls, retaining unknown/unset/default distinctions.

    Parameters
    ----------
    settings : dict
        Explicit state/value/evidence records. Omissions remain unknown.
    reference, recorded : str
        Operator source and system recording clock.

    Returns
    -------
    dict
        Complete immutable setting catalog.
    """
    return normalize_settings(settings, reference, recorded)


def setting_diff(before: dict, after: dict) -> list[dict]:
    """Return semantic before/after changes, excluding evidence-only recaptures.

    Parameters
    ----------
    before, after : dict
        Complete normalized settings; an empty before denotes a new filter.

    Returns
    -------
    list[dict]
        Explicit control/state/value changes, including unknown and unset states.
    """
    keys = ("state", "value", "platform_version")
    return [
        {"control": c, "before": before.get(c), "after": after[c]}
        for c in CONTROLS
        if not before or any(before[c].get(k) != after[c].get(k) for k in keys)
    ]


def replay_lifecycle(events: list[dict]) -> dict:
    """Reconstruct approved proposals and confirmed application versions separately.

    Parameters
    ----------
    events : list[dict]
        Verified journal events, already cut off at the desired knowledge time.

    Returns
    -------
    dict
        Policies, decisions, changes, applications and observation/settlement histories.
    """
    state = {
        "policies": {},
        "decisions": {},
        "changes": {},
        "applications": {},
        "current": {},
        "observations": {},
        "settlements": {},
        "reviews": [],
        "evaluations": {},
    }
    for e in events:
        p, kind, ident = e["payload"], e["kind"], e["event_id"]
        if kind == "POLICY":
            validate_policy(p)
            state["policies"][ident] = e
        elif kind == "EVALUATE_PROMOTION":
            if p["policy_event_id"] not in state["policies"] or any(
                v["payload"]["policy_event_id"] == p["policy_event_id"]
                for v in state["evaluations"].values()
            ):
                raise ValueError("One fixed-look evaluation per declared policy required")
            state["evaluations"][ident] = e
        elif kind == "DECISION":
            if (
                p["action"] not in ACTIONS
                or p["policy_event_id"] not in state["policies"]
                or p["production_authorized"] is not False
            ):
                raise ValueError("Invalid reviewed decision lineage")
            state["decisions"][ident] = e
        elif kind in {"BASELINE", "CONFIRM_APPLICATION"}:
            if kind == "CONFIRM_APPLICATION":
                change = state["changes"].get(p["change_event_id"])
                if (
                    not change
                    or change["status"] != "approved"
                    or p["approval_event_id"] != change["approval"]["event_id"]
                ):
                    raise ValueError("Application requires exact approved change")
                if (
                    p["settings"] != change["event"]["payload"]["after_settings"]
                    or p["status"] != change["event"]["payload"]["target_status"]
                ):
                    raise ValueError("Applied configuration differs from approval")
                change["status"] = "applied"
            if p["previous_application_id"] != state["current"].get(p["filter_id"]):
                raise ValueError("Stale applied configuration history")
            state["applications"][ident] = e
            state["current"][p["filter_id"]] = ident
        elif kind == "CHANGE":
            if (
                p["decision_event_id"] not in state["decisions"]
                or p["policy_event_id"] not in state["policies"]
            ):
                raise ValueError("Missing reviewed decision or policy")
            state["changes"][ident] = {"event": e, "status": "proposed", "approval": None}
        elif kind in {"APPROVE_CHANGE", "REJECT_CHANGE"}:
            change = state["changes"].get(p["change_event_id"])
            if not change or change["status"] not in {"proposed", "approved"}:
                raise ValueError("Change is no longer reviewable")
            if kind == "APPROVE_CHANGE":
                if (
                    change["status"] != "proposed"
                    or change["event"]["payload"]["approval_blockers"]
                ):
                    raise ValueError("Blocked settings change cannot be approved")
                state["changes"][p["change_event_id"]].update(status="approved", approval=e)
            else:
                change["status"] = "rejected"
        elif kind == "OBSERVE":
            if p["application_id"] not in state["applications"] or ident in state["observations"]:
                raise ValueError("Unknown applied version")
            state["observations"][ident] = e
        elif kind == "SETTLE":
            if p["observation_event_id"] not in state["observations"]:
                raise ValueError("Unknown monitoring observation")
            prior = state["settlements"].get(p["observation_event_id"])
            if p.get("supersedes_event_id") != (prior["event_id"] if prior else None):
                raise ValueError("Monitoring correction chain mismatch")
            state["settlements"][p["observation_event_id"]] = e
        elif kind == "MONITOR_REVIEW":
            if p["application_id"] not in state["applications"]:
                raise ValueError("Unknown monitoring application")
            state["reviews"].append(e)
        else:
            raise ValueError("Unknown filter lifecycle event")
    return state


def monitoring_tables(events: list[dict], as_of: str) -> dict[str, pd.DataFrame]:
    """Summarize current status, nonpooled economics, support and declared drift.

    Parameters
    ----------
    events : list[dict]
        System-known register events, not arbitrary historical exports.
    as_of : str
        Explicit knowledge cutoff; application periods use half-open UTC intervals.

    Returns
    -------
    dict[str,pandas.DataFrame]
        Application periods, monitoring outcomes, metric cohorts and drift checks.
    """
    events = [e for e in events if aware(e["recorded_at"]) <= aware(as_of)]
    state = replay_lifecycle(events)
    periods, rows, metrics, drifts = [], [], [], []
    for ident, application in state["applications"].items():
        p = application["payload"]
        successors = [
            a
            for a in state["applications"].values()
            if a["payload"]["previous_application_id"] == ident
        ]
        end = successors[0]["payload"]["effective_at_utc"] if successors else None
        common = {
            "application_id": ident,
            "filter_id": p["filter_id"],
            "configuration_sha256": fingerprint(p["settings"]),
            "policy_event_id": p["policy_event_id"],
        }
        periods.append(
            {
                **common,
                "status": p["status"],
                "effective_from": p["effective_at_utc"],
                "effective_until": end,
                "confirmed_at": application["recorded_at"],
                "current": state["current"][p["filter_id"]] == ident,
                "origin": application["kind"],
                "change_event_id": p.get("change_event_id"),
                "production_approval": p.get("production_approval", "not_authorized"),
                "pilot_caps_json": json.dumps(p.get("pilot_caps")),
                "pilot_expires_at": (
                    aware(p["effective_at_utc"])
                    + pd.Timedelta(days=p["pilot_caps"]["max_duration_days"])
                ).isoformat()
                if p.get("pilot_caps")
                else None,
                "pilot_expiry_review_due": bool(
                    p.get("pilot_caps")
                    and state["current"][p["filter_id"]] == ident
                    and p["status"] == "PROD"
                    and aware(as_of)
                    >= aware(p["effective_at_utc"])
                    + pd.Timedelta(days=p["pilot_caps"]["max_duration_days"])
                ),
            }
        )
        selected = []
        for obs_id, observation in state["observations"].items():
            obs = observation["payload"]
            if obs["application_id"] != ident:
                continue
            settlement = state["settlements"].get(obs_id)
            outcome = settlement["payload"] if settlement else {}
            included = end is None or aware(obs["observed_at_utc"]) < aware(end)
            row = {
                **common,
                **obs,
                "observation_event_id": obs_id,
                "result": outcome.get("result", "pending"),
                "unit_return": outcome.get("unit_return"),
                "profit_usd": outcome.get("profit_usd"),
                "eligible_period": included,
                "settlement_event_id": settlement["event_id"] if settlement else None,
                "pilot_breaches_json": json.dumps(obs.get("pilot_breaches", [])),
            }
            rows.append(row)
            if included:
                selected.append(row)
        monitor = state["policies"][p["policy_event_id"]]["payload"]["monitoring"]
        for mode in ("paper", "operator_reported_real_money"):
            cohort = [r for r in selected if r["execution_mode"] == mode]
            for availability in ("all_captured", "original_quote_checked_available"):
                chosen = (
                    cohort
                    if availability == "all_captured"
                    else [r for r in cohort if r["availability_status"] == "checked_available"]
                )
                settled = [
                    r for r in chosen if r["unit_return"] is not None and r["result"] != "void"
                ]
                n = len(settled)
                groups = len({(r["event_namespace"], r["event_id"]) for r in settled})
                stake = sum(r["stake_usd"] for r in settled)
                profit = sum(r["profit_usd"] for r in settled)
                metrics.append(
                    {
                        **common,
                        "execution_mode": mode,
                        "availability_cohort": availability,
                        "captured": len(chosen),
                        "settled": n,
                        "pending": sum(r["result"] == "pending" for r in chosen),
                        "void": sum(r["result"] == "void" for r in chosen),
                        "event_groups": groups,
                        "stake_usd": stake,
                        "profit_usd": profit,
                        "stake_weighted_roi": profit / stake if stake else None,
                        "equal_stake_mean_return": sum(r["unit_return"] for r in settled) / n
                        if n
                        else None,
                        "support_met": n >= monitor["min_wagers"]
                        and groups >= monitor["min_event_groups"],
                        "pilot_breach_count": sum(bool(r.get("pilot_breaches")) for r in chosen),
                        "interpretation": "descriptive_not_promotion_or_causal_evidence",
                    }
                )
            for feature in sorted(FEATURES):
                values = [r["features"][feature] for r in cohort if feature in r["features"]]
                reference = monitor["reference_means"].get(feature)
                mean = sum(values) / len(values) if values else None
                delta = mean - reference if mean is not None and reference is not None else None
                limit = monitor["absolute_alert_deltas"].get(feature)
                drifts.append(
                    {
                        **common,
                        "execution_mode": mode,
                        "feature": feature,
                        "observed_count": len(values),
                        "denominator": len(cohort),
                        "mean": mean,
                        "reference_mean": reference,
                        "delta": delta,
                        "alert_threshold": limit,
                        "status": "unknown_reference_or_no_observations"
                        if delta is None
                        else "review_alert"
                        if abs(delta) >= limit
                        else "within_declared_threshold",
                        "automatic_action": False,
                    }
                )
    columns = {
        "applications": [
            "application_id",
            "filter_id",
            "configuration_sha256",
            "policy_event_id",
            "status",
            "effective_from",
            "effective_until",
            "confirmed_at",
            "current",
            "origin",
            "change_event_id",
            "production_approval",
            "pilot_caps_json",
            "pilot_expires_at",
            "pilot_expiry_review_due",
        ],
        "observations": [
            "application_id",
            "filter_id",
            "configuration_sha256",
            "policy_event_id",
            "observation_key",
            "observed_at_utc",
            "event_start_utc",
            "event_namespace",
            "event_id",
            "execution_mode",
            "stake_usd",
            "american_odds",
            "availability_status",
            "features",
            "evidence_reference",
            "fill_reference",
            "availability_reference",
            "observation_event_id",
            "result",
            "unit_return",
            "profit_usd",
            "eligible_period",
            "settlement_event_id",
            "pilot_breaches_json",
        ],
        "metrics": [
            "application_id",
            "filter_id",
            "configuration_sha256",
            "policy_event_id",
            "execution_mode",
            "availability_cohort",
            "captured",
            "settled",
            "pending",
            "void",
            "event_groups",
            "stake_usd",
            "profit_usd",
            "stake_weighted_roi",
            "equal_stake_mean_return",
            "support_met",
            "pilot_breach_count",
            "interpretation",
        ],
        "drift": [
            "application_id",
            "filter_id",
            "configuration_sha256",
            "policy_event_id",
            "execution_mode",
            "feature",
            "observed_count",
            "denominator",
            "mean",
            "reference_mean",
            "delta",
            "alert_threshold",
            "status",
            "automatic_action",
        ],
    }
    return {
        name: pd.DataFrame(data, columns=columns[name])
        for name, data in zip(columns, (periods, rows, metrics, drifts), strict=True)
    }
