"""
prospective.py
===============================================================================
Purpose
-------
Publish exact definition-to-approval proposals and validate pinned review evidence.

What this module provides
-------------------------
Immutable recommendation-compatible bridges and version-specific review validation.

Design Notes
------------
Publishing a bridge does not approve, register or launch an experiment.
===============================================================================
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.analysis.prospective import validate_bound_protocol
from oddsjam_ev.io.manifests import write_json, writer_lock
from oddsjam_ev.io.paper_definitions import load_definition_publication
from oddsjam_ev.io.recommendations import load_recommendations
from oddsjam_ev.io.source_csv import fingerprint


def publish_definition_proposal(
    publication: Path,
    version_id: str,
    output_root: Path,
    filter_id: str,
    sampling_rule: str,
    experiment_type: str = "signal_screening",
) -> Path:
    """Create an immutable approval proposal from one explicit frozen definition.

    Parameters
    ----------
    publication : pathlib.Path
        Verified frozen definition publication.
    version_id : str
        Exact version, never an automatically selected latest definition.
    output_root : pathlib.Path
        Private output directory.
    filter_id, sampling_rule, experiment_type : str
        Explicit prospective scanner identity, parent capture policy and paper test type.

    Returns
    -------
    pathlib.Path
        Recommendation bundle usable by the existing human decision/register commands.
    """
    snapshot = load_definition_publication(publication)
    selected = [e for e in snapshot["events"] if e["version_id"] == version_id]
    if len(selected) != 1:
        raise ValueError("Exact published definition version required")
    definition = selected[0]
    if [e for e in snapshot["events"] if e["test_key"] == definition["test_key"]][-1][
        "version_id"
    ] != version_id:
        raise ValueError("Superseded definition in this snapshot; publish its replacement")
    plan = definition["protocol"]
    protocol = {
        "contract_version": 2,
        "definition": definition,
        "definition_publication_id": snapshot["manifest"]["publication_id"],
        "definition_manifest_sha256": hashlib.sha256(
            (publication / "manifest.json").read_bytes()
        ).hexdigest(),
        "experiment_type": experiment_type,
        "filter_id": filter_id,
        "sampling_rule": sampling_rule,
        "filter_config": plan["baseline"]["filter_config"],
        "predicate": definition["source"]["exact_predicate"],
        **{
            k: plan[k]
            for k in ("planned_start_utc", "planned_end_utc", "funding", "max_funded_stake_usd")
        },
    }
    validate_bound_protocol(protocol)
    from oddsjam_ev.analysis import prospective

    spec = {
        "contract_version": 2,
        "protocol": protocol,
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(prospective.__file__))
        },
    }
    ident = fingerprint(spec)
    proposal_id = "definition-proposal:" + fingerprint(protocol)
    tables = {
        "proposals": pd.DataFrame(
            [
                {
                    "proposal_id": proposal_id,
                    "source_report_id": definition["source"]["discovery_publication_id"],
                    "action": "CREATE_TEST",
                    "approval_scope": "draft_paper_protocol_only",
                    "approval_allowed": True,
                    "approval_blockers_json": "[]",
                    "funding": "paper_only",
                    "ui_controls_json": "[]",
                    "target_filter_id": None,
                    "predicate_json": json.dumps(protocol["predicate"], sort_keys=True),
                    "prospective_protocol_json": json.dumps(protocol, sort_keys=True),
                    "definition_version_id": version_id,
                }
            ]
        ),
        "candidate_evidence": pd.DataFrame(
            [
                {
                    "definition_version_id": version_id,
                    "definition_json": json.dumps(definition, sort_keys=True),
                }
            ]
        ),
        "coverage_audit": pd.DataFrame(
            [
                {
                    "definition_version_id": version_id,
                    "configuration_provenance": "pinned_parameter_sheet"
                    if "configuration_reference" in plan
                    else "operator_attested_baseline_unknown_scanner_completeness",
                    "external_execution_performed": False,
                }
            ]
        ),
    }
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            load_recommendations(destination)
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            hashes = {}
            for name, table in tables.items():
                file = name + ".parquet"
                table.to_parquet(stage / file, index=False)
                hashes[file] = hashlib.sha256((stage / file).read_bytes()).hexdigest()
            write_json(
                stage / "manifest.json",
                {
                    "bundle_id": ident,
                    "specification": spec,
                    "output_sha256": hashes,
                    "table_rows": {k: len(v) for k, v in tables.items()},
                },
            )
            stage.rename(destination)
    return destination


def validate_review(item: dict, payload: dict, now: str) -> dict:
    """Require a current report pinned to the same experiment and definition version.

    Parameters
    ----------
    item, payload : dict
        Current state and explicit scheduled review/report evidence.
    now : str
        System recording time.

    Returns
    -------
    dict
        Payload with immutable report identity and recorded assessment, not promotion.
    """
    from oddsjam_ev.io.experiment_reports import load_experiment_report

    reference = payload.get("review_assessment_reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError("Pinned prospective report required")
    path = Path(reference)
    tables = load_experiment_report(path)
    manifest = json.loads((path / "manifest.json").read_text())
    ident = item["registration"]["experiment_id"]
    pinned = [e for e in manifest["specification"]["events"] if e["experiment_id"] == ident]
    if (
        not pinned
        or pinned[-1]["event_id"] != item["events"][-1]["event_id"]
        or aware(manifest["specification"]["as_of"]) > aware(now)
    ):
        raise ValueError("Stale or future review report")
    rows = tables.get("prospective_reviews", pd.DataFrame())
    if rows.empty:
        raise ValueError("Definition-bound review evidence missing")
    selected = rows.loc[
        rows.experiment_id.eq(ident)
        & rows.definition_version_id.eq(item["protocol"]["definition"]["version_id"])
    ]
    if payload.get("scheduled_review_utc"):
        selected = selected.loc[selected.scheduled_review_utc.eq(payload["scheduled_review_utc"])]
    if selected.empty:
        raise ValueError("Review definition/date mismatch")
    if payload.get("scheduled_review_utc") and not selected.iloc[0].review_due:
        raise ValueError("Review report must cover the due scheduled date")
    row = selected.iloc[0]
    if (
        payload.get("to_status") == "READY_FOR_REVIEW"
        and row.readiness != "eligible_for_human_readiness_review"
        and not row.settlement_followup_elapsed
    ):
        raise ValueError("Outcomes or declared follow-up are incomplete")
    return {
        **payload,
        "review_report_id": manifest["report_id"],
        "readiness_assessment_status": row.readiness,
        "blockers_json": row.blockers_json,
    }
