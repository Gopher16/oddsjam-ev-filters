"""
decision_records.py
===============================================================================
Purpose
-------
Publish immutable candidate configuration sheets tied to verified historical evidence.

What this module provides
-------------------------
Pinned input contracts, complete records, integrity checking and identical replay.

Design Notes
------------
Builders run explicitly; dashboards never infer baselines, lifecycle or live defaults.
===============================================================================
"""

import hashlib
import json
import tempfile
from pathlib import Path

from oddsjam_ev.analysis.decision_config import validate_evidence
from oddsjam_ev.analysis.decision_records import resolve_decision
from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.signal_prioritization import load_prioritization
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_decisions(path: Path) -> dict:
    """Verify one explicit decision publication without consulting live evidence.

    Parameters
    ----------
    path : pathlib.Path
        Immutable private publication directory.

    Returns
    -------
    dict
        Manifest, complete configuration records and reconciliation audit.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    if spec["contract_version"] != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Decision publication integrity failure")
    aware(spec["as_of"])
    aware(spec["effective_at"])
    records = json.loads(
        verify_bytes(path / "decisions.json", manifest["output_sha256"]["decisions.json"])
    )
    audit = json.loads(verify_bytes(path / "audit.json", manifest["output_sha256"]["audit.json"]))
    if (
        len(records) != manifest["record_count"]
        or audit != manifest["audit"]
        or len({r["decision_id"] for r in records}) != len(records)
    ):
        raise ValueError("Decision count mismatch")
    for record in records:
        if (
            fingerprint({k: v for k, v in record.items() if k != "record_sha256"})
            != record["record_sha256"]
        ):
            raise ValueError("Decision record integrity failure")
    return {"manifest": manifest, "records": records, "audit": audit}


def publish_decisions(
    discovery: Path,
    prioritization: Path,
    evidence: dict,
    output_root: Path,
    as_of: str,
    effective_at: str,
) -> Path:
    """Resolve two explicitly labeled change sheets per frozen source candidate.

    Parameters
    ----------
    discovery, prioritization : pathlib.Path
        Exact verified source publications.
    evidence : dict
        Explicit baseline/lifecycle/translation snapshot; missing evidence is allowed
        and produces blocked sheets, never guessed settings.
    output_root : pathlib.Path
        Private destination outside input publications.
    as_of, effective_at : str
        Knowledge cutoff and effective configuration time.

    Returns
    -------
    pathlib.Path
        Immutable publication. No inference, approval or execution is performed.
    """
    if any(output_root.resolve().is_relative_to(p.resolve()) for p in (discovery, prioritization)):
        raise ValueError("Decision output must not modify source publications")
    if aware(effective_at) > aware(as_of):
        raise ValueError("Effective time cannot follow knowledge cutoff")
    json.dumps(evidence, allow_nan=False)
    normalized = validate_evidence(evidence)
    family = load_within_filter_discovery(discovery)
    priority = load_prioritization(prioritization)
    dm = json.loads((discovery / "manifest.json").read_text())
    pm = json.loads((prioritization / "manifest.json").read_text())
    ds, ps = dm["specification"], pm["specification"]
    if (
        ps["discovery_publication_id"] != dm["publication_id"]
        or ps["discovery_manifest_sha256"] != _digest(discovery / "manifest.json")
        or ps["diagnostics_id"] != ds["diagnostics_id"]
        or ps["diagnostics_manifest_sha256"] != ds["input_manifest_sha256"]["diagnostics"]
    ):
        raise ValueError("Decision source lineage mismatch")
    candidates = family["signal_candidates"]
    stats, ranks = priority["signal_statistics"], priority["signal_priorities"]
    if any(
        frame.candidate_id.duplicated().any()
        or set(frame.candidate_id) != set(candidates.candidate_id)
        for frame in (stats, ranks)
    ):
        raise ValueError("Complete source candidate family required")
    stats, ranks = stats.set_index("candidate_id"), ranks.set_index("candidate_id")
    files = [
        Path(__file__),
        Path(__file__).parents[1] / "analysis" / "decision_config.py",
        Path(__file__).parents[1] / "analysis" / "decision_records.py",
    ]
    spec = {
        "contract_version": 1,
        "as_of": aware(as_of).isoformat(),
        "effective_at": aware(effective_at).isoformat(),
        "discovery_publication_id": dm["publication_id"],
        "discovery_manifest_sha256": _digest(discovery / "manifest.json"),
        "prioritization_publication_id": pm["publication_id"],
        "prioritization_manifest_sha256": _digest(prioritization / "manifest.json"),
        "canonical_bundle_id": ds["canonical_bundle_id"],
        "atlas_id": ds["atlas_id"],
        "canonical_manifest_sha256": ds["input_manifest_sha256"]["canonical"],
        "atlas_manifest_sha256": ds["input_manifest_sha256"]["atlas"],
        "evidence": evidence,
        "evidence_sha256": fingerprint(evidence),
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
        "interpretation": "configuration_sheets_not_recommendation_ranking_or_production_authorization",
    }
    ident = fingerprint(spec)
    records = []
    for candidate in candidates.sort_values("candidate_id").to_dict("records"):
        stat, rank = stats.loc[candidate["candidate_id"]], ranks.loc[candidate["candidate_id"]]
        if any(
            candidate[k] != stat[k]
            for k in ("parent_id", "filter_id", "execution_status", "product")
        ):
            raise ValueError("Decision parent scope mismatch")
        for design in ("subgroup_tracking", "exclusion_comparison"):
            record = resolve_decision(candidate, normalized, design, as_of, effective_at)
            record.pop("record_sha256")
            record["decision_id"] = "decision:" + fingerprint(
                [ident, candidate["candidate_id"], design]
            )
            record["source_publication_id"] = ident
            record["historical_evidence"] = {
                "category": rank.category,
                "test_status": stat.test_status,
                "included_shortlist": bool(rank.included_shortlist),
                "statistics": json.loads(stat.to_json()),
                "interpretation": "unchanged_historical_evidence_not_performance_of_proposed_settings",
            }
            record["record_sha256"] = fingerprint(record)
            records.append(record)
    audit = {
        "candidate_count": len(candidates),
        "record_count": len(records),
        "setup_ready_count": sum(
            r["setup_status"] == "ready_for_operator_configuration_review" for r in records
        ),
        "blocked_count": sum(r["setup_status"] == "blocked" for r in records),
        "confirmed_prod_sheet_count": sum(
            r["lifecycle"]["status"] == "PROD"
            and r["lifecycle"]["confirmation_status"] == "confirmed"
            for r in records
        ),
        "historical_shortlist_count": len(priority["signal_shortlist"]),
        "statistical_policy_changed": False,
        "external_execution_performed": False,
    }
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old = load_decisions(destination)
            if old["records"] != records or old["audit"] != audit:
                raise ValueError("Decision replay mismatch")
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            write_json(stage / "decisions.json", records)
            write_json(stage / "audit.json", audit)
            write_json(
                stage / "manifest.json",
                {
                    "publication_id": ident,
                    "specification": spec,
                    "record_count": len(records),
                    "audit": audit,
                    "output_sha256": {
                        name: _digest(stage / name) for name in ("decisions.json", "audit.json")
                    },
                },
            )
            load_decisions(stage)
            stage.rename(destination)
    return destination
