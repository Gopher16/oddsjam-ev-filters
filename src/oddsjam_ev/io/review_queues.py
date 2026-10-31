"""
review_queues.py
===============================================================================
Purpose
-------
Publish immutable opportunity and production-change review records from pinned inputs.

What this module provides
-------------------------
Full lineage checks, unchanged correction bytes, bounded failure audits and replay.

Design Notes
------------
No latest discovery, inference rerun, dashboard default change or external execution.
===============================================================================
"""

import hashlib
import json
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from oddsjam_ev.analysis.review_catalogue import ReviewBudgetExceeded
from oddsjam_ev.analysis.review_config import ReviewConfig
from oddsjam_ev.analysis.review_queues import build_review_queues
from oddsjam_ev.io.decision_records import load_decisions
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.signal_prioritization import load_prioritization
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.visual_diagnostics import load_visual_diagnostics
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_review_queues(path: Path) -> dict:
    """Read one explicit verified queue publication; never infer a newest artifact.

    Parameters
    ----------
    path : pathlib.Path
        Immutable private publication directory.

    Returns
    -------
    dict
        Queue records, groups, overlap, coverage, original correction and manifest.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    ReviewConfig(**spec["policy"])
    if spec["contract_version"] != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Review specification integrity failure")
    payload = json.loads(
        verify_bytes(path / "review.json", manifest["output_sha256"]["review.json"])
    )
    audit = json.loads(verify_bytes(path / "audit.json", manifest["output_sha256"]["audit.json"]))
    hypothesis_bytes = verify_bytes(
        path / "source_hypotheses.parquet", manifest["output_sha256"]["source_hypotheses.parquet"]
    )
    if hashlib.sha256(hypothesis_bytes).hexdigest() != spec["source_hypotheses_sha256"]:
        raise ValueError("Review changed source correction bytes")
    if (
        audit != manifest["audit"]
        or len(payload["records"]) != audit["record_count"]
        or len({r["candidate_id"] for r in payload["records"]}) != audit["record_count"]
    ):
        raise ValueError("Review audit or record mismatch")
    hypotheses = read_parquet_file(path / "source_hypotheses.parquet")
    if len(hypotheses) != audit["source_hypothesis_count"]:
        raise ValueError("Review correction family count mismatch")
    return {**payload, "manifest": manifest, "audit": audit, "source_hypotheses": hypotheses}


def publish_review_queues(
    discovery: Path,
    prioritization: Path,
    decisions: Path,
    visuals: Path,
    output_root: Path,
    cfg: ReviewConfig,
) -> Path:
    """Join exact source publications and preserve their correction family unchanged.

    Parameters
    ----------
    discovery, prioritization, decisions, visuals : pathlib.Path
        Verified matching source publications, never inferred from timestamps.
    output_root : pathlib.Path
        Separate private destination, outside every source publication.
    cfg : ReviewConfig
        Frozen global catalogue, display and complete-publication budgets.

    Returns
    -------
    pathlib.Path
        Immutable verified output, identical replay or explicit whole-build failure.
    """
    paths = {
        k: Path(p)
        for k, p in (
            ("discovery", discovery),
            ("prioritization", prioritization),
            ("decisions", decisions),
            ("visuals", visuals),
        )
    }
    output_root = Path(output_root)
    if any(output_root.resolve().is_relative_to(path.resolve()) for path in paths.values()):
        raise ValueError("Review output must not modify source publications")
    family = load_within_filter_discovery(paths["discovery"])
    priority = load_prioritization(paths["prioritization"])
    sheets = load_decisions(paths["decisions"])
    visual = load_visual_diagnostics(paths["visuals"])
    manifests = {k: json.loads((p / "manifest.json").read_text()) for k, p in paths.items()}
    dm, pm, sm, vm = (manifests[k] for k in ("discovery", "prioritization", "decisions", "visuals"))
    ds, ps, ss, vs = (m["specification"] for m in (dm, pm, sm, vm))
    hashes = {k: _digest(p / "manifest.json") for k, p in paths.items()}
    if (
        ps["discovery_publication_id"] != dm["publication_id"]
        or ps["discovery_manifest_sha256"] != hashes["discovery"]
        or ss["discovery_publication_id"] != dm["publication_id"]
        or ss["discovery_manifest_sha256"] != hashes["discovery"]
        or ss["prioritization_publication_id"] != pm["publication_id"]
        or ss["prioritization_manifest_sha256"] != hashes["prioritization"]
        or any(vs[k] != ds[k] for k in ("canonical_bundle_id", "atlas_id", "diagnostics_id"))
        or any(
            vs["source_manifest_sha256"][k] != ds["input_manifest_sha256"][k]
            for k in ("canonical", "atlas", "diagnostics")
        )
        or ps["diagnostics_id"] != ds["diagnostics_id"]
        or ps["diagnostics_manifest_sha256"] != ds["input_manifest_sha256"]["diagnostics"]
    ):
        raise ValueError("Review source lineage mismatch")
    analysis = Path(__file__).parents[1] / "analysis"
    files = [
        Path(__file__),
        *[
            analysis / (n + ".py")
            for n in (
                "review_config",
                "review_catalogue",
                "review_ranking",
                "review_queues",
                "decision_config",
                "decision_records",
                "paired_comparison",
                "prioritization_config",
                "visual_features",
                "visual_config",
            )
        ],
    ]
    source_file = paths["prioritization"] / "signal_hypotheses.parquet"
    spec = {
        "contract_version": 1,
        "policy": asdict(cfg),
        "input_publication_ids": {k: v["publication_id"] for k, v in manifests.items()},
        "input_manifest_sha256": hashes,
        "source_hypotheses_sha256": _digest(source_file),
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "global_catalogue_inference": "descriptive_only_not_part_of_existing_correction_family",
        "units": {
            "returns": "fractions",
            "source_ev": "percentage_points",
            "money": "reported_USD_not_verified_fee_net",
        },
        "interpretation": "research_priority_not_validation_or_production_authorization",
    }
    spec = json.loads(json.dumps(spec))
    ident = fingerprint(spec)
    try:
        payload, audit = build_review_queues(family, priority, sheets, visual, cfg)
    except ReviewBudgetExceeded as exc:
        failure = {
            "status": "aborted_budget_not_truncated",
            "reason": str(exc),
            "specification": spec,
        }
        with writer_lock(output_root):
            folder = output_root / "failures"
            folder.mkdir(exist_ok=True)
            file = folder / (fingerprint(failure) + ".json")
            if file.exists():
                if json.loads(file.read_text()) != failure:
                    raise ValueError("Review failure replay mismatch") from exc
            else:
                write_json(file, failure)
        raise
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old = load_review_queues(destination)
            if any(old[k] != v for k, v in payload.items()) or old["audit"] != audit:
                raise ValueError("Review replay mismatch")
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as temporary:
            stage = Path(temporary)
            write_json(stage / "review.json", payload)
            write_json(stage / "audit.json", audit)
            (stage / "source_hypotheses.parquet").write_bytes(source_file.read_bytes())
            write_json(
                stage / "manifest.json",
                {
                    "publication_id": ident,
                    "specification": spec,
                    "audit": audit,
                    "output_sha256": {
                        name: _digest(stage / name)
                        for name in ("review.json", "audit.json", "source_hypotheses.parquet")
                    },
                },
            )
            load_review_queues(stage)
            stage.rename(destination)
    return destination
