"""
decision_register.py
===============================================================================
Purpose
-------
Record explicit human research decisions as an append-only local audit trail.

What this module provides
-------------------------
Version-bound transitions, hash-chained events and atomic single-writer publication.

Design Notes
------------
Reviewer names are attestations, not authenticated identities. This is not a
secure authorization system or tamper-proof store. Approval never executes trades.
===============================================================================
"""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from oddsjam_ev.io.manifests import write_json, writer_lock
from oddsjam_ev.io.recommendations import load_recommendations
from oddsjam_ev.io.source_csv import fingerprint

TRANSITIONS = {
    "PROPOSED": {"UNDER_REVIEW", "DEFERRED", "REJECTED", "SUPERSEDED"},
    "UNDER_REVIEW": {"APPROVED", "REJECTED", "DEFERRED", "SUPERSEDED"},
    "DEFERRED": {"UNDER_REVIEW", "REJECTED", "SUPERSEDED"},
    "APPROVED": {"SUPERSEDED"},
    "REJECTED": {"SUPERSEDED"},
    "SUPERSEDED": set(),
}


def read_decisions(root: Path) -> list[dict]:
    """Verify chained local decisions and legal state transitions.

    Parameters
    ----------
    root : pathlib.Path
        Private register root; absence means no recorded decisions.

    Returns
    -------
    list[dict]
        Verified chronological events. Tail deletion cannot be proven locally.
    """
    events = []
    previous = None
    states = {}
    for sequence, path in enumerate(sorted(root.glob("*.json")), start=1):
        event = json.loads(path.read_text())
        digest = event.pop("event_id", None)
        if (
            path.name != f"{sequence:08d}.json"
            or event.get("sequence") != sequence
            or event.get("previous_event_id") != previous
            or fingerprint(event) != digest
        ):
            raise ValueError("Decision register integrity failure")
        key = (event["recommendation_bundle_id"], event["proposal_id"])
        prior = states.get(key, "PROPOSED")
        if event.get("from_status") != prior or event.get("to_status") not in TRANSITIONS.get(
            prior, set()
        ):
            raise ValueError("Invalid decision history transition")
        if (
            not isinstance(event.get("reviewer"), str)
            or not event["reviewer"].strip()
            or not isinstance(event.get("rationale"), str)
            or not event["rationale"].strip()
        ):
            raise ValueError("Decision actor and rationale required")
        stamp = datetime.fromisoformat(event["recorded_at"])
        if stamp.tzinfo is None:
            raise ValueError("Decision timestamps require timezone")
        event["event_id"] = digest
        events.append(event)
        previous = digest
        states[key] = event["to_status"]
    return events


def record_decision(
    root: Path,
    bundle: Path,
    proposal_id: str,
    to_status: str,
    reviewer: str,
    rationale: str,
    expected_status: str,
    expected_event_id: str | None = None,
) -> dict:
    """Append an explicit human transition bound to one exact proposal version.

    Parameters
    ----------
    root, bundle : pathlib.Path
        Private decision register and immutable recommendation bundle.
    proposal_id : str
        Exact proposal identity from that bundle.
    to_status : str
        Requested research-review state.
    reviewer, rationale : str
        Explicit human attribution and explanation, never inferred from environment.
    expected_status : str
        Optimistic concurrency check on the proposal state.
    expected_event_id : str, optional
        Required latest proposal event ID after its initial transition.

    Returns
    -------
    dict
        Published hash-chained event; no external changes occur.
    """
    if not reviewer.strip() or not rationale.strip():
        raise ValueError("Reviewer and rationale are required")
    tables = load_recommendations(bundle)
    selected = tables["proposals"].loc[tables["proposals"].proposal_id.eq(proposal_id)]
    if len(selected) != 1:
        raise ValueError("Proposal not found in the selected bundle")
    row = selected.iloc[0]
    bundle_id = json.loads((bundle / "manifest.json").read_text())["bundle_id"]
    with writer_lock(root):
        events = read_decisions(root)
        matching = [
            e
            for e in events
            if e["recommendation_bundle_id"] == bundle_id and e["proposal_id"] == proposal_id
        ]
        state = matching[-1]["to_status"] if matching else "PROPOSED"
        latest = matching[-1]["event_id"] if matching else None
        if state != expected_status or expected_event_id != latest:
            raise ValueError("Stale decision state or event version")
        if to_status not in TRANSITIONS.get(state, set()):
            raise ValueError("Invalid decision transition")
        if to_status == "APPROVED" and (
            not bool(row.approval_allowed) or json.loads(row.approval_blockers_json)
        ):
            raise ValueError("Proposal approval requirements are incomplete")
        event = {
            "sequence": len(events) + 1,
            "previous_event_id": events[-1]["event_id"] if events else None,
            "recommendation_bundle_id": bundle_id,
            "source_report_id": row.source_report_id,
            "proposal_id": proposal_id,
            "from_status": state,
            "to_status": to_status,
            "reviewer": reviewer.strip(),
            "rationale": rationale.strip(),
            "recorded_at": datetime.now(UTC).isoformat(),
            "approval_scope": row.approval_scope,
            "external_execution_performed": False,
        }
        event["event_id"] = fingerprint(event)
        destination = root / f"{len(events) + 1:08d}.json"
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as staging:
            temp = Path(staging) / "event.json"
            write_json(temp, event)
            # Atomic exclusive publication; never replace existing audit history.
            os.link(temp, destination)
        return event
