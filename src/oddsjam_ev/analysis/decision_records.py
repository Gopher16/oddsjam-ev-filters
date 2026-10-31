"""
decision_records.py
===============================================================================
Purpose
-------
Resolve candidate changes against explicit prospective configuration and lifecycle.

What this module provides
-------------------------
Complete parameter sheets, narrow-only intersections and auditable setup blockers.

Design Notes
------------
A ready setup is not approval or deployment. Compound exclusions are not inverted
into unrelated independent scanner controls. Historical unknowns remain unknown.
===============================================================================
"""

import json
import re

from oddsjam_ev.analysis.decision_config import CONTROLS, active, unknown
from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.io.source_csv import fingerprint

CATEGORICAL = {
    "sportsbook": "sportsbooks",
    "sport": "sports",
    "league": "leagues",
    "market_family": "markets",
}


def _constraints(term):
    dimension = term.get("dimension")
    if dimension in CATEGORICAL and term.get("operator") == "==":
        return [(CATEGORICAL[dimension], [term["value"]])]
    if (
        dimension == "advertised_ev_percent"
        and term.get("operator") in {">", ">=", "<", "<="}
        and term.get("units") == "percentage_points"
    ):
        key = "min_ev" if term["operator"].startswith(">") else "max_ev"
        return [
            (
                key,
                {
                    "value": float(term["endpoint"]),
                    "operator": term["operator"],
                    "units": "percentage_points",
                },
            )
        ]
    if dimension in {"odds_band", "advertised_ev_band"} and term.get("operator") == "==":
        match = re.fullmatch(
            r"([\[(])\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*([\])])", term["value"]
        )
        if match:
            prefix = "odds" if dimension == "odds_band" else "ev"
            units = "american_odds" if prefix == "odds" else "percentage_points"
            return [
                (
                    "min_" + prefix,
                    {
                        "value": float(match[2]),
                        "operator": ">=" if match[1] == "[" else ">",
                        "units": units,
                    },
                ),
                (
                    "max_" + prefix,
                    {
                        "value": float(match[3]),
                        "operator": "<=" if match[4] == "]" else "<",
                        "units": units,
                    },
                ),
            ]
    return []


def _intersection(control, baseline, proposed):
    if baseline["state"] == "not_applicable":
        raise ValueError("Candidate constrains a not-applicable control")
    if baseline["state"] in {"unknown", "unset", "unrestricted"}:
        return proposed
    old = baseline["value"]
    if isinstance(old, list):
        # An explicit alias mapping cannot silently widen a known scanner universe.
        if not set(proposed) <= set(old):
            raise ValueError("Proposed scanner values fall outside the prospective baseline")
        return proposed
    if control.startswith(("min_", "max_")) and isinstance(old, dict):
        is_min = control.startswith("min_")
        if old["value"] == proposed["value"]:
            op = ">" if is_min else "<"
            return {
                **old,
                "operator": op
                if op in {old["operator"], proposed["operator"]}
                else old["operator"],
            }
        return old if (old["value"] > proposed["value"]) == is_min else proposed
    return proposed


def resolve_decision(candidate, evidence, design, as_of, effective_at):
    """Build a complete setup sheet from explicit known/effective evidence.

    Parameters
    ----------
    candidate : dict
        Frozen identity, scope and exact predicate; no performance is recomputed.
    evidence : dict
        Normalized baseline/lifecycle/translation evidence.
    design : str
        subgroup_tracking or exclusion_comparison.
    as_of, effective_at : str
        Aware knowledge cutoff and effective configuration time.

    Returns
    -------
    dict
        Full baseline/proposed settings, logical change, provenance and readiness.
    """
    if design not in {"subgroup_tracking", "exclusion_comparison"}:
        raise ValueError("Unsupported decision design")
    if aware(effective_at) > aware(as_of):
        raise ValueError("Future lifecycle is not current confirmed production")
    parent = candidate.get("parent_id")
    baselines = [
        b
        for b in evidence["baselines"]
        if b["filter_id"] == candidate.get("filter_id") and active(b, effective_at, as_of)
    ]
    blockers = []
    baseline = baselines[0] if len(baselines) == 1 else None
    if baseline is None:
        blockers.append(
            "missing_prospective_baseline" if not baselines else "ambiguous_prospective_baselines"
        )
    elif baseline["kind"] != ("prospective_parent" if parent else "new_prospective"):
        blockers.append("baseline_scope_mismatch")
        baseline = None
    settings = baseline["settings"] if baseline else {k: unknown() for k in CONTROLS}
    lifecycle = [
        item
        for item in evidence["lifecycle"]
        if item["filter_id"] == candidate.get("filter_id") and active(item, effective_at, as_of)
    ]
    statuses = {item["status"] for item in lifecycle}
    status = next(iter(statuses)) if len(statuses) == 1 else "unknown"
    lifecycle_status = (
        "confirmed"
        if len(statuses) == 1 and status != "unknown"
        else "conflicting"
        if len(statuses) > 1
        else "unknown"
    )
    if design == "exclusion_comparison":
        if lifecycle_status != "confirmed" or status != "PROD":
            blockers.append("production_lifecycle_not_confirmed")
        blockers.append("compound_exclusion_requires_verified_implementation")
    terms = json.loads(candidate["predicate_json"])
    if not isinstance(terms, list) or not terms:
        raise ValueError("Exact nonempty candidate predicate required")
    rows = {
        control: {
            "control": control,
            "baseline_state": item["state"],
            "baseline_value": item["value"],
            "state": "inherited" if baseline and item["state"] != "unknown" else "unknown",
            "final_value": item["value"],
            "candidate_override": None,
            "basis": item.get("basis"),
            "evidence_reference": item.get("evidence_reference"),
            "observed_at": item.get("observed_at"),
            "reason": item["reason"],
            "baseline_id": baseline["baseline_id"] if baseline else None,
            "baseline_sha256": baseline["baseline_sha256"] if baseline else None,
            "translation_status": "not_changed",
            "setup_blockers": [],
        }
        for control, item in settings.items()
    }
    # Study predicates narrow the baseline. Exclusions retain its controls and expose
    # the exact NOT(conjunction) separately; they are never guessed as UI edits.
    changed_controls = set()
    for term in terms if design == "subgroup_tracking" else []:
        constraints = _constraints(term)
        if not constraints:
            blockers.append("unsupported_candidate_dimension:" + str(term.get("dimension")))
        for control, suggested in constraints:
            row = rows[control]
            if control in changed_controls:
                row["setup_blockers"].append("multiple_constraints_require_composed_translation")
                continue
            changed_controls.add(control)
            translations = [
                t
                for t in evidence["translations"]
                if t["term"] == term
                and t["control"] == control
                and aware(t["recorded_at"]) <= aware(as_of)
            ]
            row["candidate_override"] = {"analysis_term": term, "suggested_value": suggested}
            verified = len(translations) == 1
            if verified and isinstance(suggested, dict):
                verified = translations[0]["value"] == suggested
            if not verified:
                row["translation_status"] = "unverified_or_ambiguous"
                row["setup_blockers"].append("exact_scanner_translation_required")
                continue
            translation = translations[0]
            try:
                final = _intersection(control, settings[control], translation["value"])
            except ValueError:
                row["setup_blockers"].append("candidate_baseline_conflict")
                continue
            if settings[control]["state"] == "unknown":
                row["setup_blockers"].append("baseline_restriction_unknown")
            row.update(
                state="override",
                final_value=final,
                basis="prospective_design",
                evidence_reference=translation["evidence_reference"],
                observed_at=translation["recorded_at"],
                translation_status="verified",
                reason="Explicit scanner translation intersected with the prospective baseline",
            )
    for control, row in rows.items():
        if row["state"] == "unknown":
            row["setup_blockers"].append("setting_unknown")
        if row["setup_blockers"]:
            blockers.extend(control + ":" + reason for reason in row["setup_blockers"])
    for prefix in ("odds", "ev"):
        low, high = rows["min_" + prefix]["final_value"], rows["max_" + prefix]["final_value"]
        if isinstance(low, dict) and isinstance(high, dict):
            if low["value"] > high["value"] or (
                low["value"] == high["value"]
                and (low["operator"] == ">" or high["operator"] == "<")
            ):
                blockers.append("empty_" + prefix + "_intersection")
    blocked = sorted(set(blockers))
    record = {
        "candidate_id": candidate["candidate_id"],
        "parent_id": parent,
        "filter_id": candidate.get("filter_id"),
        "historical_funding": candidate.get("execution_status"),
        "product": candidate.get("product"),
        "design": design,
        "historical_settings_reconstructed": False,
        "logical_change": {
            "operator": "AND" if design == "subgroup_tracking" else "NOT",
            "terms": terms,
            "scope": "within_parent" if parent else "new_prospective_universe",
        },
        "baseline": baseline,
        "settings": list(rows.values()),
        "lifecycle": {
            "status": status,
            "confirmation_status": lifecycle_status,
            "evidence": lifecycle,
            "effective_at": effective_at,
            "knowledge_as_of": as_of,
        },
        "setup_status": "blocked" if blocked else "ready_for_operator_configuration_review",
        "setup_blockers": blocked,
        "production_change_authorized": False,
        "external_execution_performed": False,
        "next_action": "resolve_setup_evidence"
        if blocked
        else "review_and_freeze_paper_definition",
    }
    record["record_sha256"] = fingerprint(record)
    return record


def parameter_sheet(record):
    """Return display rows without hiding null values or their distinct states."""
    return [
        {
            "Control": r["control"].replace("_", " ").title(),
            "Baseline state": r["baseline_state"],
            "Baseline value": json.dumps(r["baseline_value"]),
            "Candidate change": json.dumps(r["candidate_override"]),
            "Final state": r["state"],
            "Final value": json.dumps(r["final_value"]),
            "Units": (
                r["final_value"].get("units")
                if isinstance(r["final_value"], dict) and "units" in r["final_value"]
                else "USD"
                if CONTROLS[r["control"]] == "usd"
                else "fraction"
                if CONTROLS[r["control"]] == "fraction"
                else "control-specific"
            ),
            "Basis": r["basis"],
            "Evidence": r["evidence_reference"],
            "Setup blocker": "; ".join(r["setup_blockers"]),
            "Reason": r["reason"],
        }
        for r in record["settings"]
    ]
