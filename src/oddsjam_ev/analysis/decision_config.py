"""
decision_config.py
===============================================================================
Purpose
-------
Describe every scanner/account control without guessing missing settings or defaults.

What this module provides
-------------------------
Typed setting states, prospective baseline validation and bitemporal lifecycle evidence.

Design Notes
------------
Configuration evidence and historical economics are separate. No name implies PROD.
===============================================================================
"""

import math
from zoneinfo import ZoneInfo

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.io.source_csv import fingerprint

# All controls are displayed, including unsupported or unknown ones.
CONTROLS = {
    "recommended_filters": "bool",
    "ev_formula_builder": "object",
    "devig_books": "list",
    "devig_weights": "weights",
    "devig_method": "text",
    "model_version": "text",
    "sportsbooks": "list",
    "sports": "list",
    "leagues": "list",
    "market_types": "list",
    "markets": "list",
    "games": "list",
    "min_odds": "odds_min",
    "max_odds": "odds_max",
    "min_ev": "ev_min",
    "max_ev": "ev_max",
    "minimum_liquidity": "usd",
    "minimum_recommended_bet": "usd",
    "market_width": "nonnegative",
    "date_range": "text",
    "pregame_live": "text",
    "time_to_event": "object",
    "timezone": "zone",
    "bet_warnings": "flags",
    "paper_sizing": "object",
    "bankroll": "usd",
    "kelly_multiplier": "fraction",
}
INPUT_STATES = {
    "observed",
    "unrestricted",
    "unset",
    "platform_default",
    "not_applicable",
    "unknown",
}
BASES = {
    "historical_observation",
    "prospective_design",
    "operational_constraint",
    "platform_requirement",
}


def unknown(reason="No supported setting evidence supplied"):
    """Return a missing setting, not an inactive control or a numeric zero."""
    return {
        "state": "unknown",
        "value": None,
        "reason": reason,
        "basis": None,
        "evidence_reference": None,
        "observed_at": None,
        "platform_version": None,
    }


def validate_value(control, value):
    """Validate units, bound endpoints and control-specific scalar/list types."""
    kind = CONTROLS[control]
    valid = False
    if kind == "bool":
        valid = type(value) is bool
    elif kind == "list":
        valid = (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(v, str) and v.strip() for v in value)
            and len(value) == len(set(value))
        )
    elif kind in {"text", "zone"}:
        valid = isinstance(value, str) and bool(value.strip())
        if valid and kind == "zone":
            ZoneInfo(value)
    elif kind == "object":
        valid = isinstance(value, dict) and bool(value)
    elif kind in {"flags", "weights"}:
        valid = (
            isinstance(value, dict)
            and bool(value)
            and all(isinstance(k, str) and k.strip() for k in value)
        )
        if valid:
            valid = (
                all(type(v) is bool for v in value.values())
                if kind == "flags"
                else all(
                    type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in value.values()
                )
                and sum(value.values()) > 0
            )
    elif kind in {"usd", "nonnegative", "fraction"}:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        if valid and kind == "fraction":
            valid = value <= 1
    elif kind.startswith(("odds_", "ev_")):
        valid = isinstance(value, dict) and set(value) == {"value", "operator", "units"}
        if valid:
            n = value["value"]
            valid = type(n) in (int, float) and math.isfinite(n)
            valid = valid and value["operator"] in (
                {">", ">="} if kind.endswith("min") else {"<", "<="}
            )
            valid = valid and value["units"] == (
                "american_odds" if kind.startswith("odds") else "percentage_points"
            )
            if valid and kind.startswith("odds"):
                valid = abs(n) >= 100
    if not valid:
        raise ValueError("Invalid typed control value: " + control)


def normalize_settings(settings, reference, recorded_at):
    """Fill omitted controls as unknown while validating every supplied setting.

    Parameters
    ----------
    settings : dict
        Control names mapped to explicit value/state/evidence objects.
    reference, recorded_at : str
        Baseline evidence reference and aware recording time; setting-specific
        evidence may override the reference but never postdate recording.

    Returns
    -------
    dict
        Complete catalog with original semantics retained.
    """
    if not isinstance(settings, dict) or set(settings) - set(CONTROLS):
        raise ValueError("Unknown configuration controls")
    result = {key: unknown() for key in CONTROLS}
    for key, item in settings.items():
        if not isinstance(item, dict) or set(item) - {
            "state",
            "value",
            "reason",
            "basis",
            "evidence_reference",
            "observed_at",
            "platform_version",
        }:
            raise ValueError("Invalid setting record")
        state = item.get("state")
        if (
            state not in INPUT_STATES
            or not isinstance(item.get("reason"), str)
            or not item["reason"].strip()
        ):
            raise ValueError("Setting state and reason required")
        value = item.get("value")
        if state in {"unknown", "unset", "unrestricted", "not_applicable"}:
            if value is not None:
                raise ValueError("Non-value setting states require null, not zero/default")
        else:
            validate_value(key, value)
        evidence = item.get("evidence_reference", reference)
        observed = item.get("observed_at", recorded_at)
        if state != "unknown":
            if (
                item.get("basis") not in BASES
                or not isinstance(evidence, str)
                or not evidence.strip()
            ):
                raise ValueError("Known setting needs explicit basis and evidence")
            if aware(observed) > aware(recorded_at):
                raise ValueError("Setting observation postdates its recording")
            if state == "platform_default" and not item.get("platform_version"):
                raise ValueError("Verified platform defaults require a platform version")
        result[key] = {
            **unknown(),
            **item,
            "value": value,
            "evidence_reference": evidence,
            "observed_at": observed,
        }
    return result


def validate_evidence(config):
    """Validate baseline, lifecycle and exact UI-translation records.

    Parameters
    ----------
    config : dict
        Explicit local evidence; no current source YAML is consulted implicitly.

    Returns
    -------
    dict
        Normalized configuration with stable baseline fingerprints.
    """
    if (
        not isinstance(config, dict)
        or set(config) != {"contract_version", "baselines", "lifecycle", "translations"}
        or type(config["contract_version"]) is not int
        or config["contract_version"] != 1
    ):
        raise ValueError("Unsupported decision evidence contract")
    if not all(isinstance(config[k], list) for k in ("baselines", "lifecycle", "translations")):
        raise ValueError("Evidence collections must be lists")
    normalized = {"contract_version": 1, "baselines": [], "lifecycle": [], "translations": []}
    ids = set()
    for baseline in config["baselines"]:
        if set(baseline) != {
            "baseline_id",
            "filter_id",
            "kind",
            "recorded_at",
            "effective_from",
            "effective_until",
            "evidence_reference",
            "settings",
        }:
            raise ValueError("Exact baseline evidence fields required")
        if baseline["kind"] not in {"prospective_parent", "new_prospective"}:
            raise ValueError("Historical registry snapshots are not prospective baselines")
        for key in ("baseline_id", "filter_id", "evidence_reference"):
            if not isinstance(baseline[key], str) or not baseline[key].strip():
                raise ValueError("Explicit baseline identity and evidence required")
        if baseline["baseline_id"] in ids:
            raise ValueError("Duplicate baseline ID")
        ids.add(baseline["baseline_id"])
        _dates(baseline)
        item = {
            **baseline,
            "settings": normalize_settings(
                baseline["settings"], baseline["evidence_reference"], baseline["recorded_at"]
            ),
        }
        item["baseline_sha256"] = fingerprint(item)
        normalized["baselines"].append(item)
    for lifecycle in config["lifecycle"]:
        if (
            set(lifecycle)
            != {
                "filter_id",
                "status",
                "recorded_at",
                "effective_from",
                "effective_until",
                "evidence_reference",
                "actor",
                "confirmation",
            }
            or lifecycle["status"] not in {"PROD", "TEST", "retired", "unknown"}
            or lifecycle["confirmation"] != "operator_confirmed"
        ):
            raise ValueError(
                "Explicit operator-confirmed lifecycle required; names do not confirm PROD"
            )
        if not all(
            isinstance(lifecycle[k], str) and lifecycle[k].strip()
            for k in ("filter_id", "evidence_reference", "actor")
        ):
            raise ValueError("Lifecycle identity actor and evidence required")
        _dates(lifecycle)
        normalized["lifecycle"].append(dict(lifecycle))
    for translation in config["translations"]:
        if (
            set(translation)
            != {
                "term",
                "control",
                "value",
                "recorded_at",
                "evidence_reference",
                "platform_version",
                "semantics_verified",
            }
            or translation["control"] not in CONTROLS
            or translation["semantics_verified"] is not True
        ):
            raise ValueError("Explicit verified scanner translation required")
        if (
            not isinstance(translation["term"], dict)
            or not translation["term"]
            or not all(
                isinstance(translation[k], str) and translation[k].strip()
                for k in ("evidence_reference", "platform_version")
            )
        ):
            raise ValueError("Translation needs exact source term and platform evidence")
        aware(translation["recorded_at"])
        validate_value(translation["control"], translation["value"])
        normalized["translations"].append(dict(translation))
    return normalized


def _dates(record):
    aware(record["recorded_at"])
    start = aware(record["effective_from"])
    if record["effective_until"] is not None and aware(record["effective_until"]) <= start:
        raise ValueError("Effective evidence interval must be positive and half-open")


def active(record, effective_at, as_of):
    """Use evidence known by the cutoff and effective in the half-open interval."""
    return (
        aware(record["recorded_at"]) <= aware(as_of)
        and aware(record["effective_from"]) <= aware(effective_at)
        and (
            record["effective_until"] is None
            or aware(effective_at) < aware(record["effective_until"])
        )
    )
