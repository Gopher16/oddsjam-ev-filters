"""
operating_regimes.py
===============================================================================
Purpose
-------
Preserve optional operating-process evidence without rewriting historical facts.

What this module provides
-------------------------
Append-only validation, explicit legacy adapters and non-economic wager linkage.

Design Notes
------------
Date-only boundaries are documentary evidence, not invented midnight intervals.
Settings, selection, funding and accepted fills are distinct evidence dimensions.
===============================================================================
"""

from __future__ import annotations

import copy
import json
from datetime import date

import pandas as pd

from oddsjam_ev.analysis.decision_config import normalize_settings, validate_evidence
from oddsjam_ev.analysis.filter_attribution import resolve_labels, validate_registry
from oddsjam_ev.analysis.sizing_config import aware, validate_history
from oddsjam_ev.io.source_csv import fingerprint

RECORD_KEYS = {
    "evidence_id",
    "namespace",
    "filter_id",
    "product",
    "effective_from",
    "effective_to",
    "open_ended",
    "precision",
    "effective_interval_verified",
    "recorded_at",
    "evidence_type",
    "confidence",
    "source_reference",
    "selection_mode",
    "funding",
    "paper_sizing",
    "settings",
    "venue_access",
}
PLACEMENT_KEYS = {
    "evidence_id",
    "bet_id",
    "placement_at",
    "placement_verified",
    "recorded_at",
    "source_reference",
}
COLUMNS = [
    "bet_id",
    "regime_ids_json",
    "linkage_status",
    "selection_mode",
    "funding_evidence",
    "paper_sizing",
    "settings_json",
    "venue_access_json",
]


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty operating evidence identity/reference required")


def _keys(value, keys):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Exact operating evidence fields required")


def _bounds(record):
    precision = record["precision"]
    start, end = record["effective_from"], record["effective_to"]
    if precision == "unknown":
        if start is not None or end is not None or record["open_ended"]:
            raise ValueError("Unknown interval requires null boundaries")
    elif precision in {"timestamp", "date"}:
        parse = aware if precision == "timestamp" else date.fromisoformat
        if not isinstance(start, str):
            raise ValueError("Explicit effective start required")
        begin = parse(start)
        if precision == "date" and any(
            value is not None and parse(value).isoformat() != value for value in (start, end)
        ):
            raise ValueError("Date boundaries require YYYY-MM-DD")
        if end is not None and (not isinstance(end, str) or parse(end) <= begin):
            raise ValueError("Invalid effective interval")
        if record["open_ended"] and end is not None:
            raise ValueError("Open interval cannot have an end")
        if record["effective_interval_verified"] and end is None and not record["open_ended"]:
            raise ValueError("Verified interval needs a known or explicitly open end")
    else:
        raise ValueError("Unsupported interval precision")
    if record["effective_interval_verified"] and precision != "timestamp":
        raise ValueError("Only exact aware intervals support verified linkage")


def validate_regimes(history: dict, registry: dict, previous: dict | None = None) -> str:
    """Validate a versioned ledger and optional exact-prefix revision.

    Parameters
    ----------
    history : dict
        Operating records and independent placement evidence at an aware cutoff.
    registry : dict
        Existing stable filter registry; aliases never establish regime membership.
    previous : dict or None
        Prior ledger whose arrays must remain exact prefixes.

    Returns
    -------
    str
        Logical content fingerprint; inputs are never mutated.
    """
    validate_registry(registry)
    ids = {r["id"] for r in registry["filters"]}
    _keys(
        history,
        {"contract_version", "knowledge_as_of", "previous_history_id", "records", "wager_evidence"},
    )
    if type(history["contract_version"]) is not int or history["contract_version"] != 1:
        raise ValueError("Unsupported operating regime contract")
    cutoff = aware(history["knowledge_as_of"])
    if not all(isinstance(history[k], list) for k in ("records", "wager_evidence")):
        raise ValueError("Evidence arrays required")
    if previous is None:
        if history["previous_history_id"] is not None:
            raise ValueError("Predecessor required")
    else:
        if history["previous_history_id"] != fingerprint(previous) or cutoff < aware(
            previous["knowledge_as_of"]
        ):
            raise ValueError("Predecessor/cutoff mismatch")
        if any(
            history[k][: len(previous[k])] != previous[k] for k in ("records", "wager_evidence")
        ):
            raise ValueError("Operating history must be append-only")
    seen = set()
    for row, placement in [(r, False) for r in history["records"]] + [
        (r, True) for r in history["wager_evidence"]
    ]:
        if not isinstance(row, dict):
            raise ValueError("Evidence record must be an object")
        _keys(row, PLACEMENT_KEYS if placement else RECORD_KEYS)
        for key in ("evidence_id", "source_reference"):
            _text(row[key])
        if row["evidence_id"] in seen:
            raise ValueError("Duplicate operating evidence ID")
        seen.add(row["evidence_id"])
        if aware(row["recorded_at"]) > cutoff:
            raise ValueError("Evidence postdates knowledge cutoff")
        if placement:
            _text(row["bet_id"])
            if type(row["placement_verified"]) is not bool:
                raise ValueError("Explicit placement verification required")
            if row["placement_at"] is not None:
                if aware(row["placement_at"]) > aware(row["recorded_at"]):
                    raise ValueError("Placement postdates recording")
            elif row["placement_verified"]:
                raise ValueError("Verified placement requires a timestamp")
            continue
        _text(row["namespace"])
        if row["filter_id"] is not None and row["filter_id"] not in ids:
            raise ValueError("Unknown stable filter ID")
        if row["product"] not in {None, "exchange", "fixed_odds", "pickem", "unknown"}:
            raise ValueError("Unsupported product scope")
        if any(type(row[k]) is not bool for k in ("open_ended", "effective_interval_verified")):
            raise ValueError("Explicit interval flags required")
        _bounds(row)
        choices = {
            "selection_mode": {"unknown", "broad_capture", "discretionary"},
            "funding": {"unknown", "paper", "real_money"},
            "paper_sizing": {"unknown", "kelly_recommendation"},
            "evidence_type": {
                "user_confirmed",
                "contemporaneous_record",
                "reconstructed_assumption",
                "unknown",
                "planned",
            },
            "confidence": {"confirmed", "assumed", "unknown"},
        }
        if any(row[k] not in allowed for k, allowed in choices.items()):
            raise ValueError("Unsupported operating evidence state")
        if row["paper_sizing"] != "unknown" and row["funding"] != "paper":
            raise ValueError("Paper sizing convention requires paper evidence")
        normalize_settings(row["settings"], row["source_reference"], row["recorded_at"])
        if not isinstance(row["venue_access"], list) or any(
            not isinstance(v, str) or not v.strip() for v in row["venue_access"]
        ):
            raise ValueError("Venue access must be explicit names, not inferred permissions")
        if len(row["venue_access"]) != len(set(row["venue_access"])):
            raise ValueError("Duplicate venue access evidence")
    return fingerprint(history)


def regime_record(evidence_id: str, namespace: str, recorded_at: str, reference: str) -> dict:
    """Create an unknown record for explicit private completion.

    Parameters
    ----------
    evidence_id, namespace, recorded_at, reference : str
        Identity, source scope, aware receipt time and evidence locator.

    Returns
    -------
    dict
        Unlinked unknown evidence with no assumed settings or access.
    """
    return dict(
        evidence_id=evidence_id,
        namespace=namespace,
        filter_id=None,
        product=None,
        effective_from=None,
        effective_to=None,
        open_ended=False,
        precision="unknown",
        effective_interval_verified=False,
        recorded_at=recorded_at,
        evidence_type="unknown",
        confidence="unknown",
        source_reference=reference,
        selection_mode="unknown",
        funding="unknown",
        paper_sizing="unknown",
        settings={},
        venue_access=[],
    )


def migrate_evidence(
    registry: dict,
    knowledge_as_of: str,
    *,
    sizing: dict | None = None,
    decision: dict | None = None,
) -> dict:
    """Adapt existing evidence without inventing selection or funded application.

    Parameters
    ----------
    registry : dict
        Existing exact alias/ID registry.
    knowledge_as_of : str
        Explicit new aware knowledge cutoff.
    sizing, decision : dict or None
        Initial legacy sizing ledger or prospective decision-evidence payload.
        Original artifacts must be retained separately; source hashes are referenced.

    Returns
    -------
    dict
        New independent version-1 operating ledger; legacy inputs unchanged.
    """
    mapping = validate_registry(registry)
    stable = {r["id"] for r in registry["filters"]}
    result = dict(
        contract_version=1,
        knowledge_as_of=knowledge_as_of,
        previous_history_id=None,
        records=[],
        wager_evidence=[],
    )
    if sizing is not None:
        # A revision cannot be validated without its predecessor; migrate initial
        # ledgers only, rather than silently trusting a detached revision.
        validate_history(sizing)
        for p in sizing["policies"]:
            row = regime_record(
                "sizing:" + p["policy_id"],
                p["namespace"],
                p["recorded_at"],
                f"legacy-sizing:{fingerprint(sizing)}:{p['policy_id']}",
            )
            row.update(
                product=p["scope"]["product"],
                effective_from=p["effective_from"],
                effective_to=p["effective_to"],
                open_ended=p["open_ended"],
                precision="timestamp" if p["effective_from"] else "unknown",
                effective_interval_verified=p["effective_interval_verified"],
                evidence_type=p["evidence_type"],
                confidence=p["confidence"],
            )
            for control, value in (
                ("bankroll", p["bankroll_amount"]),
                ("kelly_multiplier", p["kelly_multiplier"]),
            ):
                if value is not None:
                    row["settings"][control] = dict(
                        state="observed",
                        value=float(value),
                        reason="Migrated setting; not verified wager application",
                        basis="historical_observation",
                    )
            # Preserve legacy funding scope as a restriction, never as a funding claim.
            if p["scope"]["execution_status"] is not None:
                row["effective_interval_verified"] = False
            result["records"].append(row)
        for w in sizing["wager_evidence"]:
            result["wager_evidence"].append(
                dict(
                    evidence_id="sizing:" + w["evidence_id"],
                    bet_id=w["bet_id"],
                    placement_at=w["placement_at"],
                    placement_verified=w["placement_verified"],
                    recorded_at=w["recorded_at"],
                    source_reference=f"legacy-sizing:{fingerprint(sizing)}:{w['evidence_id']}",
                )
            )
    if decision is not None:
        validate_evidence(decision)
        for b in decision["baselines"]:
            ident = b["filter_id"]
            if ident not in stable:
                resolved, _ = resolve_labels(ident, mapping)
                if len(resolved) != 1:
                    raise ValueError("Ambiguous or unknown legacy filter alias")
                ident = resolved[0]
            # Decision baselines have no account namespace; require owner completion.
            row = regime_record(
                "baseline:" + b["baseline_id"],
                "unassigned_legacy_namespace",
                b["recorded_at"],
                f"legacy-decision:{fingerprint(decision)}:{b['baseline_id']}",
            )
            row.update(
                filter_id=ident,
                evidence_type="planned",
                effective_from=b["effective_from"],
                effective_to=b["effective_until"],
                open_ended=b["effective_until"] is None,
                precision="timestamp",
                settings=copy.deepcopy(b["settings"]),
            )
            result["records"].append(row)
    validate_regimes(result, registry)
    return result


def build_regime_linkage(
    bets: pd.DataFrame, attribution: pd.DataFrame, history: dict, registry: dict
) -> tuple[pd.DataFrame, dict]:
    """Link supported evidence without mutating canonical candidates or money.

    Parameters
    ----------
    bets, attribution : pandas.DataFrame
        Canonical candidates and existing non-economic stable membership bridge.
    history, registry : dict
        Validated operating ledger and pinned registry.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        One non-economic row per candidate and an unresolved-coverage audit.
    """
    # Revision integrity is verified by the publisher; validate the detached
    # payload's fields without altering its predecessor identity.
    initial = {**history, "previous_history_id": None}
    validate_regimes(initial, registry)
    if bets.bet_id.isna().any() or bets.bet_id.duplicated().any():
        raise ValueError("Unique canonical candidate IDs required")
    if set(attribution.bet_id) - set(bets.bet_id) or {
        w["bet_id"] for w in history["wager_evidence"]
    } - set(bets.bet_id):
        raise ValueError("Evidence refers to unknown canonical candidate")
    trusted = [
        r
        for r in history["records"]
        if r["effective_interval_verified"]
        and r["evidence_type"] in {"user_confirmed", "contemporaneous_record"}
        and r["confidence"] == "confirmed"
    ]
    placements_by_bet = {}
    for w in history["wager_evidence"]:
        if w["placement_verified"]:
            placements_by_bet.setdefault(w["bet_id"], set()).add(aware(w["placement_at"]))
    memberships_by_bet = {}
    for membership in attribution.itertuples():
        if membership.attribution_status == "resolved" and pd.notna(membership.filter_id):
            memberships_by_bet.setdefault(membership.bet_id, set()).add(membership.filter_id)
    rows = []
    for b in bets.itertuples():
        places = placements_by_bet.get(b.bet_id, set())
        memberships = memberships_by_bet.get(b.bet_id, set())
        matches = []
        status = (
            "unknown_placement"
            if not places
            else "conflicting_placement"
            if len(places) > 1
            else "unknown_regime"
        )
        if len(places) == 1:
            at = next(iter(places))
            matches = [
                r
                for r in trusted
                if r["namespace"] == b.namespace
                and (r["filter_id"] is None or r["filter_id"] in memberships)
                and (r["product"] is None or r["product"] == b.product)
                and aware(r["effective_from"]) <= at
                and (r["open_ended"] or at < aware(r["effective_to"]))
            ]
            if matches:
                status = "matched" if len(matches) == 1 else "ambiguous_regime"
        row = dict(
            zip(
                COLUMNS,
                [
                    b.bet_id,
                    json.dumps(sorted(r["evidence_id"] for r in matches)),
                    status,
                    "unknown",
                    "unknown",
                    "unknown",
                    "{}",
                    "[]",
                ],
                strict=True,
            )
        )
        if status == "matched":
            r = matches[0]
            if b.execution_status in {"paper", "real_money"} and r["funding"] not in {
                "unknown",
                b.execution_status,
            }:
                row["linkage_status"] = "funding_conflict"
            else:
                row.update(
                    selection_mode=r["selection_mode"],
                    funding_evidence=r["funding"],
                    paper_sizing=r["paper_sizing"],
                    settings_json=json.dumps(
                        normalize_settings(r["settings"], r["source_reference"], r["recorded_at"]),
                        sort_keys=True,
                    ),
                    venue_access_json=json.dumps(r["venue_access"]),
                )
        rows.append(row)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    audit = dict(
        candidate_count=len(bets),
        linkage_status_counts=frame.linkage_status.value_counts().sort_index().to_dict(),
        unlinked_record_ids=sorted(
            r["evidence_id"] for r in history["records"] if r not in trusted
        ),
        canonical_classifications_unchanged=True,
        accepted_fills_not_inferred=True,
    )
    return frame, audit
