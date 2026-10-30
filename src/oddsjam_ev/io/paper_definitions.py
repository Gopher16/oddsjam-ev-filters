"""
paper_definitions.py
===============================================================================
Purpose
-------
Store immutable candidate-bound paper definitions and publish frozen review copies.

What this module provides
-------------------------
Append-only versions, exact source lineage, future-start gates and verified replay.

Design Notes
------------
No definition launches or approves an experiment. Existing approval controls remain.
Register tail deletion needs an external head anchor to detect, as with other logs.
===============================================================================
"""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.analysis.paper_definitions import KEY, validate_definition
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.signal_prioritization import load_prioritization
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery


def utc_now():
    """Return recording time; callers cannot backdate through the CLI."""
    return datetime.now(UTC).isoformat()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay(events):
    last = None
    stamp = None
    latest = {}
    for sequence, event in enumerate(events, 1):
        body = {k: v for k, v in event.items() if k != "version_id"}
        key = event["test_key"]
        prior = latest.get(key)
        recorded = aware(event["recorded_at"])
        validate_definition(event["protocol"])
        if (
            event["contract_version"] != 1
            or not KEY.fullmatch(key)
            or event["sequence"] != sequence
            or event["previous_event_id"] != last
            or event["previous_version_id"] != (prior["version_id"] if prior else None)
            or event["version"] != (prior["version"] + 1 if prior else 1)
            or fingerprint(body) != event["version_id"]
            or (stamp is not None and recorded < stamp)
            or aware(event["protocol"]["planned_start_utc"]) <= recorded
            or event["status"] != "definition_only_not_launched"
            or event["evaluation_status"] != "no_prospective_observations_linked"
            or event["external_execution_performed"] is not False
            or not event["actor"].strip()
            or not event["rationale"].strip()
        ):
            raise ValueError("Paper definition register integrity failure")
        latest[key] = event
        last, stamp = event["version_id"], recorded
    return latest


def read_definitions(root: Path, as_of: str | None = None) -> list[dict]:
    """Verify the complete private log before selecting system-known versions.

    Parameters
    ----------
    root : pathlib.Path
        Private definition register.
    as_of : str, optional
        Aware recording cutoff; future events are excluded only after verification.

    Returns
    -------
    list[dict]
        All versions known at the requested cutoff, including superseded definitions.
    """
    events = []
    for i, path in enumerate(sorted(root.glob("*.json")), 1):
        if path.name != f"{i:08d}.json":
            raise ValueError("Paper definition sequence gap")
        events.append(json.loads(path.read_text()))
    _replay(events)
    return [e for e in events if aware(e["recorded_at"]) <= aware(as_of)] if as_of else events


def register_definition(
    root: Path,
    discovery: Path,
    prioritization: Path,
    candidate_id: str,
    test_key: str,
    protocol: dict,
    actor: str,
    rationale: str,
    expected_version_id: str | None = None,
    decision_publication: Path | None = None,
) -> dict:
    """Freeze a new future definition or append a version with explicit concurrency.

    Parameters
    ----------
    root, discovery, prioritization : pathlib.Path
        Private register and explicit verified research publications.
    candidate_id, test_key : str
        Frozen source candidate and stable human-selected slug.
    protocol : dict
        Completed prospective intent; exact source rules cannot be overridden.
    actor, rationale : str
        Human attribution, including reasons for an exploratory test or revision.
    expected_version_id : str, optional
        Required current version when revising an existing test key.
    decision_publication : pathlib.Path, optional
        Exact verified configuration sheet for a configuration-bound definition.

    Returns
    -------
    dict
        Immutable definition version, never an experiment approval or launch event.
    """
    if not isinstance(test_key, str) or not KEY.fullmatch(test_key):
        raise ValueError("Stable lowercase test key required")
    validate_definition(protocol)
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    if (
        not isinstance(actor, str)
        or not actor.strip()
        or not isinstance(rationale, str)
        or not rationale.strip()
    ):
        raise ValueError("Actor and rationale required")
    if any(root.resolve().is_relative_to(p.resolve()) for p in (discovery, prioritization)):
        raise ValueError("Definition register must not modify source publications")
    family = load_within_filter_discovery(discovery)
    priorities = load_prioritization(prioritization)
    dm = json.loads((discovery / "manifest.json").read_text())
    pm = json.loads((prioritization / "manifest.json").read_text())
    ds, ps = dm["specification"], pm["specification"]
    if (
        ps["discovery_publication_id"] != dm["publication_id"]
        or ps["discovery_manifest_sha256"] != _digest(discovery / "manifest.json")
        or ps["diagnostics_id"] != ds["diagnostics_id"]
        or ps["diagnostics_manifest_sha256"] != ds["input_manifest_sha256"]["diagnostics"]
    ):
        raise ValueError("Definition source lineage mismatch")
    selected = family["signal_candidates"].loc[lambda f: f.candidate_id.eq(candidate_id)]
    stats = priorities["signal_statistics"].loc[lambda f: f.candidate_id.eq(candidate_id)]
    decisions = priorities["signal_priorities"].loc[lambda f: f.candidate_id.eq(candidate_id)]
    if len(selected) != 1 or len(stats) != 1 or len(decisions) != 1:
        raise ValueError("Exact source candidate required")
    candidate, stat, decision = selected.iloc[0], stats.iloc[0], decisions.iloc[0]
    if any(
        candidate[k] != stat[k] for k in ("parent_id", "filter_id", "execution_status", "product")
    ):
        raise ValueError("Definition source scope mismatch")
    source = {
        "candidate_id": candidate_id,
        "parent_id": candidate.parent_id,
        "filter_id": candidate.filter_id,
        "historical_funding": candidate.execution_status,
        "product": candidate["product"],
        "exact_predicate": json.loads(candidate.predicate_json),
        "discovery_publication_id": dm["publication_id"],
        "discovery_manifest_sha256": _digest(discovery / "manifest.json"),
        "prioritization_publication_id": pm["publication_id"],
        "prioritization_manifest_sha256": _digest(prioritization / "manifest.json"),
        "canonical_bundle_id": ds["canonical_bundle_id"],
        "atlas_id": ds["atlas_id"],
        "historical_category": decision.category,
        "historical_test_status": stat.test_status,
        "included_historical_shortlist": bool(decision.included_shortlist),
        "trial_arm": "subgroup"
        if protocol["design"] == "subgroup_tracking"
        else "parent_minus_subgroup",
        "baseline_arm": "unchanged_parent",
    }
    reference = protocol.get("configuration_reference")
    if reference is not None:
        from oddsjam_ev.io.decision_records import load_decisions

        if decision_publication is None:
            raise ValueError(
                "Configuration-bound definitions require the exact decision publication"
            )
        publication = load_decisions(decision_publication)
        match = [r for r in publication["records"] if r["decision_id"] == reference["decision_id"]]
        spec = publication["manifest"]["specification"]
        if (
            len(match) != 1
            or publication["manifest"]["publication_id"] != reference["publication_id"]
        ):
            raise ValueError("Configuration publication identity mismatch")
        record = match[0]
        if (
            record["record_sha256"] != reference["record_sha256"]
            or record["candidate_id"] != candidate_id
            or record["design"] != protocol["design"]
            or record["setup_status"] != "ready_for_operator_configuration_review"
            or spec["discovery_manifest_sha256"] != _digest(discovery / "manifest.json")
            or spec["prioritization_manifest_sha256"] != _digest(prioritization / "manifest.json")
            or aware(spec["as_of"]) > aware(utc_now())
        ):
            raise ValueError("Configuration binding is blocked or has different source evidence")
        baseline = record["baseline"]
        if (
            protocol["baseline"]["filter_config"]
            != {r["control"]: r["baseline_value"] for r in record["settings"]}
            or protocol["baseline"]["evidence_reference"] != baseline["evidence_reference"]
            or aware(protocol["planned_start_utc"]) < aware(baseline["effective_from"])
            or (
                baseline["effective_until"] is not None
                and aware(protocol["planned_end_utc"]) > aware(baseline["effective_until"])
            )
        ):
            raise ValueError("Protocol baseline differs from the pinned configuration window")
        sizing = next(
            r["final_value"] for r in record["settings"] if r["control"] == "paper_sizing"
        )
        if sizing != protocol["sizing"]:
            raise ValueError("Paper sizing differs from configuration evidence")
    elif decision_publication is not None:
        raise ValueError("Decision publication requires an explicit configuration reference")
    with writer_lock(root):
        events = read_definitions(root)
        previous = _replay(events).get(test_key)
        if expected_version_id != (previous["version_id"] if previous else None):
            raise ValueError(
                "Stale or missing definition version; revision needs current version ID"
            )
        now = utc_now()
        if aware(protocol["planned_start_utc"]) <= aware(now):
            raise ValueError("Definition must be recorded before its prospective start")
        if previous and previous["source"] == source and previous["protocol"] == protocol:
            raise ValueError("No material definition change")
        event = {
            "contract_version": 1,
            "sequence": len(events) + 1,
            "previous_event_id": events[-1]["version_id"] if events else None,
            "test_key": test_key,
            "version": previous["version"] + 1 if previous else 1,
            "previous_version_id": previous["version_id"] if previous else None,
            "recorded_at": now,
            "actor": actor.strip(),
            "rationale": rationale.strip(),
            "protocol": protocol,
            "source": source,
            "status": "definition_only_not_launched",
            "evaluation_status": "no_prospective_observations_linked",
            "external_execution_performed": False,
        }
        event["version_id"] = fingerprint(event)
        _replay([*events, event])
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as staging:
            file = Path(staging) / "definition.json"
            write_json(file, event)
            os.link(file, root / f"{len(events) + 1:08d}.json")
        return event


def load_definition_publication(path: Path) -> dict:
    """Verify a frozen definition publication; never consult a live register.

    Parameters
    ----------
    path : pathlib.Path
        Explicit content-addressed publication.

    Returns
    -------
    dict
        Verified specification and all definition versions at its knowledge cutoff.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    if spec["contract_version"] != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Paper definition publication integrity failure")
    events = json.loads(
        verify_bytes(path / "definitions.json", manifest["output_sha256"]["definitions.json"])
    )
    _replay(events)
    aware(spec["as_of"])
    if spec["register_head"] != (events[-1]["version_id"] if events else None):
        raise ValueError("Paper definition publication head mismatch")
    if events != spec["events"] or any(
        aware(e["recorded_at"]) > aware(spec["as_of"]) for e in events
    ):
        raise ValueError("Paper definition knowledge mismatch")
    return {"manifest": manifest, "events": events}


def publish_definitions(register: Path, output_root: Path, as_of: str) -> Path:
    """Publish immutable known versions without changing the register or old snapshots.

    Parameters
    ----------
    register, output_root : pathlib.Path
        Private live register and separate publication root.
    as_of : str
        Explicit knowledge cutoff no later than the recording clock.

    Returns
    -------
    pathlib.Path
        Immutable verified definition publication, including an empty register state.
    """
    if output_root.resolve().is_relative_to(register.resolve()):
        raise ValueError("Publication must not modify register")
    if aware(as_of) > aware(utc_now()):
        raise ValueError("Definition publication cutoff cannot be in the future")
    with writer_lock(register):
        events = read_definitions(register, as_of)
    files = [Path(__file__), Path(__file__).parents[1] / "analysis" / "paper_definitions.py"]
    spec = {
        "contract_version": 1,
        "as_of": aware(as_of).isoformat(),
        "events": events,
        "register_head": events[-1]["version_id"] if events else None,
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
    }
    ident = fingerprint(spec)
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            if load_definition_publication(destination)["manifest"]["specification"] != spec:
                raise ValueError("Definition replay mismatch")
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            write_json(stage / "definitions.json", events)
            write_json(
                stage / "manifest.json",
                {
                    "publication_id": ident,
                    "specification": spec,
                    "output_sha256": {"definitions.json": _digest(stage / "definitions.json")},
                },
            )
            load_definition_publication(stage)
            stage.rename(destination)
    return destination
