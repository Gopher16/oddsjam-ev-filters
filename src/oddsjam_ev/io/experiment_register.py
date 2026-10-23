"""
experiment_register.py
===============================================================================
Purpose
-------
Record approved paper experiments, enrollment and verification evidence.

What this module provides
-------------------------
Version-bound registration and append-only lifecycle, check and link commands.

Design Notes
------------
No external filter is activated and no trading action is performed.
===============================================================================
"""

import json
import math
import os
import tempfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from oddsjam_ev.analysis.experiment_config import ExperimentPolicy, aware, validate_protocol
from oddsjam_ev.analysis.experiments import STATES, enrollment_timing, replay_experiments
from oddsjam_ev.io.decision_register import read_decisions
from oddsjam_ev.io.manifests import write_json, writer_lock
from oddsjam_ev.io.recommendations import load_recommendations
from oddsjam_ev.io.source_csv import fingerprint


def utc_now() -> str:
    """Return system recording time; the public CLI cannot backdate it."""
    return datetime.now(UTC).isoformat()


def read_experiment_events(root: Path, as_of: str | None = None) -> list[dict]:
    """Verify the whole register, then select events known by a recording cutoff.

    Parameters
    ----------
    root : pathlib.Path
        Private register directory.
    as_of : str, optional
        Aware system-knowledge cutoff; not a source settlement date.

    Returns
    -------
    list[dict]
        Verified pinned events. Tail deletion is not detectable without an anchor.
    """
    events = []
    previous = None
    last_time = None
    for number, path in enumerate(sorted(root.glob("*.json")), 1):
        event = json.loads(path.read_text())
        digest = event.pop("event_id", None)
        stamp = aware(event.get("recorded_at"))
        if (
            event.get("contract_version") != 1
            or path.name != f"{number:08d}.json"
            or event.get("sequence") != number
            or event.get("previous_event_id") != previous
            or fingerprint(event) != digest
            or (last_time is not None and stamp < last_time)
        ):
            raise ValueError("Experiment register integrity failure")
        if not event.get("actor") or not event.get("rationale"):
            raise ValueError("Actor and rationale required")
        event["event_id"] = digest
        events.append(event)
        previous = digest
        last_time = stamp
    replay_experiments(events)
    return [e for e in events if aware(e["recorded_at"]) <= aware(as_of)] if as_of else events


def _append(root, events, ident, kind, payload, actor, rationale, recorded):
    event = {
        "contract_version": 1,
        "sequence": len(events) + 1,
        "previous_event_id": events[-1]["event_id"] if events else None,
        "experiment_id": ident,
        "kind": kind,
        "payload": payload,
        "actor": actor.strip(),
        "rationale": rationale.strip(),
        "recorded_at": recorded,
        "external_execution_performed": False,
    }
    if not event["actor"] or not event["rationale"]:
        raise ValueError("Actor and rationale required")
    if events and aware(recorded) < aware(events[-1]["recorded_at"]):
        raise ValueError("Recording clock moved backwards")
    event["event_id"] = fingerprint(event)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as staging:
        path = Path(staging) / "event.json"
        write_json(path, event)
        os.link(path, root / f"{len(events) + 1:08d}.json")
    return event


def register_experiment(
    root: Path,
    bundle: Path,
    decisions: Path,
    proposal_id: str,
    approval_event_id: str,
    protocol: dict,
    actor: str,
    rationale: str,
    policy: ExperimentPolicy | None = None,
) -> dict:
    """Register a protocol against the latest exact approved paper proposal.

    Parameters
    ----------
    root, bundle, decisions : pathlib.Path
        Experiment register, proposal bundle and decision register.
    proposal_id, approval_event_id : str
        Exact approved proposal and current approval event identities.
    protocol : dict
        Complete operational protocol, preserving every approved draft field.
    actor, rationale : str
        Explicit human attribution.
    policy : ExperimentPolicy, optional
        Paper-only policy.

    Returns
    -------
    dict
        Registration event, not launch confirmation.
    """
    policy = policy or ExperimentPolicy()
    validate_protocol(protocol, policy)
    tables = load_recommendations(bundle)
    selected = tables["proposals"].loc[tables["proposals"].proposal_id.eq(proposal_id)]
    if len(selected) != 1:
        raise ValueError("Unknown proposal")
    proposal = selected.iloc[0]
    bundle_id = json.loads((bundle / "manifest.json").read_text())["bundle_id"]
    if (
        proposal.action != "CREATE_TEST"
        or proposal.approval_scope != "draft_paper_protocol_only"
        or not proposal.approval_allowed
        or proposal.funding != "paper_only"
    ):
        raise ValueError("Only approvable paper-test proposals can register")
    for control in json.loads(proposal.ui_controls_json):
        if protocol["filter_config"].get(control["ui_control"]) != control["ui_value"]:
            raise ValueError("Registered configuration differs from approved UI translation")
    if (
        pd.notna(proposal.target_filter_id)
        and protocol.get("target_registry_id") != proposal.target_filter_id
    ):
        raise ValueError("Registered target differs from approved registry target")
    if json.loads(proposal.predicate_json) != protocol["predicate"]:
        raise ValueError("Protocol predicate differs from approval")
    if any(protocol.get(k) != v for k, v in json.loads(proposal.prospective_protocol_json).items()):
        raise ValueError("Material protocol changes require a new approval")
    # Serialize against review updates while reading current approval.
    with writer_lock(decisions):
        review = read_decisions(decisions)
        matching = [
            e
            for e in review
            if e["recommendation_bundle_id"] == bundle_id and e["proposal_id"] == proposal_id
        ]
        if (
            not matching
            or matching[-1]["event_id"] != approval_event_id
            or matching[-1]["to_status"] != "APPROVED"
        ):
            raise ValueError("Current exact approval required")
        now = utc_now()
        if aware(matching[-1]["recorded_at"]) > aware(now) or aware(
            protocol["planned_start_utc"]
        ) < aware(now):
            raise ValueError("Registration must precede enrollment and follow approval")
        ident = "experiment:" + fingerprint(
            {
                "bundle": bundle_id,
                "proposal": proposal_id,
                "approval": approval_event_id,
                "protocol": protocol,
                "policy": policy.__dict__,
            }
        )
        with writer_lock(root):
            events = read_experiment_events(root)
            if ident in replay_experiments(events):
                raise ValueError("Experiment already registered")
            return _append(
                root,
                events,
                ident,
                "REGISTER",
                {
                    "protocol": protocol,
                    "protocol_sha256": fingerprint(protocol),
                    "recommendation_bundle_id": bundle_id,
                    "proposal_id": proposal_id,
                    "approval_event": matching[-1],
                    "decision_register_path": str(decisions.resolve()),
                    "policy": policy.__dict__,
                },
                actor,
                rationale,
                now,
            )


def record_experiment_event(
    root: Path,
    experiment_id: str,
    kind: str,
    payload: dict,
    actor: str,
    rationale: str,
    expected_event_id: str,
) -> dict:
    """Append a lifecycle, enrollment, verification check or identity-link event.

    Parameters
    ----------
    root : pathlib.Path
        Private register.
    experiment_id, kind : str
        Existing experiment and TRANSITION, ENROLL, CHECK or LINK command.
    payload : dict
        Explicit command evidence; no browser credentials or raw requests.
    actor, rationale, expected_event_id : str
        Human attribution and latest per-experiment event for concurrency control.

    Returns
    -------
    dict
        Published event, with no external action performed.
    """
    with ExitStack() as stack:
        preview = replay_experiments(read_experiment_events(root))
        if experiment_id not in preview:
            raise ValueError("Unknown experiment")
        needs_approval = kind == "ENROLL" or (
            kind == "TRANSITION" and payload.get("to_status") == "RUNNING"
        )
        if needs_approval:
            review_root = Path(
                preview[experiment_id]["registration"]["payload"]["decision_register_path"]
            )
            stack.enter_context(writer_lock(review_root))
        stack.enter_context(writer_lock(root))
        events = read_experiment_events(root)
        state = replay_experiments(events)
        if experiment_id not in state:
            raise ValueError("Unknown experiment")
        item = state[experiment_id]
        if item["events"][-1]["event_id"] != expected_event_id:
            raise ValueError("Stale experiment event version")
        if needs_approval:
            registration = item["registration"]["payload"]
            matching = [
                e
                for e in read_decisions(review_root)
                if e["recommendation_bundle_id"] == registration["recommendation_bundle_id"]
                and e["proposal_id"] == registration["proposal_id"]
            ]
            if (
                not matching
                or matching[-1]["event_id"] != registration["approval_event"]["event_id"]
                or matching[-1]["to_status"] != "APPROVED"
            ):
                raise ValueError(
                    "Experiment approval was superseded; new enrollment or launch is blocked"
                )
        now = utc_now()
        protocol = item["protocol"]
        payload = dict(payload)
        if kind == "TRANSITION":
            target = payload.get("to_status")
            if target not in STATES.get(item["status"], set()):
                raise ValueError("Invalid experiment transition")
            if target == "RUNNING" and (
                payload.get("filter_id") != protocol["filter_id"]
                or payload.get("filter_config") != protocol["filter_config"]
                or not payload.get("evidence_reference")
            ):
                raise ValueError("Exact launch/resume configuration evidence required")
            if target == "RUNNING" and item["launch"] is None:
                launched = aware(payload.get("launched_at_utc"))
                if not aware(protocol["planned_start_utc"]) <= launched < aware(
                    protocol["planned_end_utc"]
                ) or not aware(item["registration"]["recorded_at"]) <= launched <= aware(now):
                    raise ValueError(
                        "Launch must follow registration and fall within declared window"
                    )
                if (
                    payload.get("filter_id") != protocol["filter_id"]
                    or payload.get("filter_config") != protocol["filter_config"]
                    or not payload.get("evidence_reference")
                ):
                    raise ValueError("Exact launch configuration evidence required")
            if target == "READY_FOR_REVIEW":
                from oddsjam_ev.io.experiment_reports import load_experiment_report

                reference = payload.get("review_assessment_reference")
                if not isinstance(reference, str) or not reference:
                    raise ValueError("Review readiness requires a pinned outcome report")
                report_path = Path(reference)
                assessment = load_experiment_report(report_path)
                manifest = json.loads((report_path / "manifest.json").read_text())
                recorded = [
                    e
                    for e in manifest["specification"]["events"]
                    if e["experiment_id"] == experiment_id
                ]
                audit = assessment["experiment_audit"].loc[
                    assessment["experiment_audit"].experiment_id.eq(experiment_id)
                ]
                if (
                    not recorded
                    or recorded[-1]["event_id"] != expected_event_id
                    or len(audit) != 1
                    or aware(manifest["specification"]["as_of"]) > aware(now)
                ):
                    raise ValueError("Stale or inconsistent readiness report")
                row = audit.iloc[0]
                if (
                    row.readiness != "eligible_for_human_readiness_review"
                    and not row.settlement_followup_elapsed
                ):
                    raise ValueError("Outcomes or declared follow-up are incomplete")
                payload["readiness_assessment_status"] = row.readiness
            if target == "CLOSED" and payload.get("review_outcome") not in {
                "continue_new_protocol",
                "modify_new_protocol",
                "stop",
                "inconclusive",
            }:
                raise ValueError("Explicit human review outcome required; no promotion")
        elif kind == "DEVIATION":
            if (
                item["status"] not in {"RUNNING", "PAUSED", "AWAITING_OUTCOMES"}
                or not payload.get("details")
                or not payload.get("evidence_reference")
            ):
                raise ValueError("Deviation details and evidence required")
        elif kind == "ENROLL":
            if item["status"] != "RUNNING" or aware(now) >= aware(protocol["planned_end_utc"]):
                raise ValueError("Enrollment is closed or paused")
            required = {
                "enrollment_id",
                "observation_key",
                "observed_at_utc",
                "event_start_utc",
                "feature_values",
                "evidence_reference",
            }
            if not required <= set(payload) or not all(
                isinstance(payload[k], str) and payload[k].strip()
                for k in ["enrollment_id", "observation_key", "evidence_reference"]
            ):
                raise ValueError("Enrollment evidence is incomplete")
            if not isinstance(payload["feature_values"], dict) or any(
                payload["feature_values"].get(k) != v for k, v in protocol["predicate"].items()
            ):
                raise ValueError("Decision-time predicate values required")
            if payload["enrollment_id"] in item["enrollments"] or payload["observation_key"] in {
                e["payload"]["observation_key"] for e in item["enrollments"].values()
            }:
                raise ValueError("Duplicate experiment enrollment")
            if payload["observed_at_utc"] is not None and aware(payload["observed_at_utc"]) > aware(
                now
            ):
                raise ValueError("Observation cannot be recorded from the future")
            quote = payload.get("scanner_quote")
            if not isinstance(quote, dict) or not all(
                quote.get(k) is not None for k in ["selection", "line", "odds"]
            ):
                raise ValueError("Decision-time scanner selection, line and odds are required")
            try:
                quoted_odds = float(quote["odds"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Finite scanner American odds required") from exc
            if (
                not isinstance(quote["selection"], str)
                or not quote["selection"].strip()
                or not math.isfinite(quoted_odds)
                or abs(quoted_odds) < 100
            ):
                raise ValueError("Valid scanner quote required")
            payload["timing_status"] = enrollment_timing(
                now,
                item["launch"]["payload"]["launched_at_utc"],
                payload["event_start_utc"],
                payload["observed_at_utc"],
            )
            payload["intended_mode"] = "paper"
            payload["availability_status"] = "not_checked"
        elif kind == "CHECK":
            if payload.get("enrollment_id") not in item["enrollments"]:
                raise ValueError("Unknown enrollment")
            status = payload.get("status")
            if (
                status not in {"checked_available", "checked_unavailable", "inconclusive"}
                or not payload.get("reason")
                or not payload.get("evidence_reference")
            ):
                raise ValueError("Verification status, failures and reasons must be retained")
            enrollment = item["enrollments"][payload["enrollment_id"]]["payload"]
            checked = aware(payload.get("checked_at_utc"))
            if checked > aware(now) or (
                enrollment["observed_at_utc"] is not None
                and checked < aware(enrollment["observed_at_utc"])
            ):
                raise ValueError("Invalid verification time")
            if protocol["experiment_type"] == "execution_verification" and not payload.get(
                "sample_selection_reference"
            ):
                raise ValueError("Sample selection evidence required, including failed checks")
            if status == "checked_available" and not all(
                payload.get(k) is not None
                for k in ["selection", "line", "odds", "available_stake_usd"]
            ):
                raise ValueError("Observed selection, line, odds and stake limit required")
            if status == "checked_available":
                try:
                    odds = float(payload["odds"])
                    limit = float(payload["available_stake_usd"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("Finite observed odds and stake limit required") from exc
                if (
                    not math.isfinite(odds)
                    or abs(odds) < 100
                    or not math.isfinite(limit)
                    or limit <= 0
                ):
                    raise ValueError("Valid American odds and positive observed capacity required")
            quote = enrollment["scanner_quote"]
            payload["original_quote_matched"] = (
                status == "checked_available"
                and payload["selection"] == quote["selection"]
                and str(payload["line"]) == str(quote["line"])
                and float(payload["odds"]) == float(quote["odds"])
            )
            payload["check_delay_seconds"] = (
                (checked - aware(enrollment["observed_at_utc"])).total_seconds()
                if enrollment["observed_at_utc"] is not None
                else None
            )
        elif kind == "LINK":
            if (
                payload.get("enrollment_id") not in item["enrollments"]
                or not payload.get("source_record_id")
                or not payload.get("evidence_reference")
            ):
                raise ValueError("Explicit enrollment/source identity evidence required")
            if any(
                e["payload"]["source_record_id"] == payload["source_record_id"]
                and key != payload["enrollment_id"]
                for key, e in item["links"].items()
            ):
                raise ValueError("Source identity already linked within experiment")
        else:
            raise ValueError("Unknown event command")
        return _append(root, events, experiment_id, kind, payload, actor, rationale, now)
