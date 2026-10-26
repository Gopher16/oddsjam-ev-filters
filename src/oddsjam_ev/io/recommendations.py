"""
recommendations.py
===============================================================================
Purpose
-------
Publish immutable research proposals from verified, consistently linked inputs.

What this module provides
-------------------------
Source linkage verification, proposal publication, integrity checks and replay.

Design Notes
------------
Generated proposals contain no human approvals. Private evidence stays ignored.
===============================================================================
"""

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
from oddsjam_ev.analysis.recommendations import build_recommendations
from oddsjam_ev.io.atlas import load_atlas
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.discovery import load_discovery
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.source_csv import fingerprint

TABLES = ("proposals", "candidate_evidence", "coverage_audit")


def load_recommendations(path: Path) -> dict[str, pd.DataFrame]:
    """Verify and load a pinned proposal bundle.

    Parameters
    ----------
    path : pathlib.Path
        Explicit published directory.

    Returns
    -------
    dict[str,pandas.DataFrame]
        Verified proposals and original candidate evidence.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if fingerprint(manifest["specification"]) != manifest["bundle_id"]:
        raise ValueError("Recommendation specification integrity failure")
    result = {}
    for name in TABLES:
        file = name + ".parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        result[name] = read_parquet_file(path / file)
        if len(result[name]) != manifest["table_rows"][name]:
            raise ValueError("Recommendation count integrity failure")
    if result["proposals"].proposal_id.duplicated().any():
        raise ValueError("Duplicate proposal IDs")
    return result


def publish_recommendations(
    report: Path,
    atlas: Path,
    canonical: Path,
    output_root: Path,
    cfg: RecommendationConfig,
    registry: dict,
    capabilities: dict,
) -> Path:
    """Publish proposals, rejecting mismatched source bundles.

    Parameters
    ----------
    report, atlas, canonical : pathlib.Path
        Explicit verified discovery, atlas and canonical directories.
    output_root : pathlib.Path
        Private publication root.
    cfg : RecommendationConfig
        Research action policy.
    registry, capabilities : dict
        Versioned targets and reviewed UI evidence.

    Returns
    -------
    pathlib.Path
        Immutable proposal bundle, independently verified on replay.
    """
    discovery = load_discovery(report)
    atlas_tables = load_atlas(atlas)
    canonical_tables = load_canonical_bundle(canonical)
    manifests = {
        name: json.loads((p / "manifest.json").read_text())
        for name, p in [("report", report), ("atlas", atlas), ("canonical", canonical)]
    }
    if (
        manifests["report"]["specification"]["atlas_id"] != manifests["atlas"]["atlas_id"]
        or manifests["atlas"]["specification"]["canonical_bundle_id"]
        != manifests["canonical"]["bundle_id"]
    ):
        raise ValueError("Recommendation source identities do not match")
    if (
        manifests["report"]["specification"]["atlas_manifest_sha256"]
        != hashlib.sha256((atlas / "manifest.json").read_bytes()).hexdigest()
        or manifests["atlas"]["specification"]["canonical_manifest_sha256"]
        != hashlib.sha256((canonical / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Recommendation source manifest hashes do not match")
    analysis = Path(__file__).resolve().parents[1] / "analysis"
    files = [
        analysis / (n + ".py")
        for n in ["recommendation_config", "recommendations", "filter_actions"]
    ] + [analysis / "filter_attribution.py", Path(__file__)]
    files.append(Path(__file__).with_name("parquet.py"))
    spec = {
        "source_report_id": manifests["report"]["report_id"],
        "atlas_id": manifests["atlas"]["atlas_id"],
        "canonical_bundle_id": manifests["canonical"]["bundle_id"],
        "source_manifest_sha256": {
            name: hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest()
            for name, p in [("report", report), ("atlas", atlas), ("canonical", canonical)]
        },
        "policy": asdict(cfg),
        "filter_registry": registry,
        "capabilities": capabilities,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
    }
    ident = fingerprint(spec)
    tables = build_recommendations(
        discovery,
        atlas_tables["wager_dimensions"],
        canonical_tables["filter_attribution"],
        registry,
        capabilities,
        cfg,
        manifests["report"]["report_id"],
    )
    for frame in tables.values():
        for column in frame.select_dtypes(include="object"):
            frame[column] = frame[column].where(frame[column].notna(), None)
        frame["recommendation_bundle_id"] = ident
        frame["recommendation_contract_version"] = cfg.contract_version
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old = load_recommendations(destination)
            for name in TABLES:
                assert_frame_equal(tables[name], old[name])
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            hashes = {}
            for name, frame in tables.items():
                file = name + ".parquet"
                frame.to_parquet(stage / file, index=False)
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
