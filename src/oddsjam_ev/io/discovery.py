"""
discovery.py
===============================================================================
Purpose
-------
Publish pinned research reports without changing canonical data or notebook defaults.

What this module provides
-------------------------
Complete search ledger, conditional diagnostics, temporal gates and immutable replay.

Design Notes
------------
No ranking, promotion, acquisition, scheduling or filter edits occur.
===============================================================================
"""

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.multiple_testing import adjust_family
from oddsjam_ev.analysis.robustness import evaluate_candidate
from oddsjam_ev.analysis.segment_search import enumerate_segments
from oddsjam_ev.analysis.walk_forward import build_walk_forward
from oddsjam_ev.io.atlas import load_atlas
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.source_csv import fingerprint

TABLES = (
    "candidate_ledger",
    "candidate_memberships",
    "candidate_results",
    "temporal_gates",
    "temporal_partitions",
    "frozen_candidates",
    "temporal_metrics",
    "temporal_stability",
    "temporal_fold_audit",
)


def build_discovery(
    wagers: pd.DataFrame, cfg: DiscoveryConfig, evidence: dict | None = None
) -> dict[str, pd.DataFrame]:
    """Evaluate all declared candidates, keeping missing-evidence gates explicit.

    Parameters
    ----------
    wagers : pandas.DataFrame
        Verified atlas wager dimensions.
    cfg : DiscoveryConfig
        Predeclared research policy.
    evidence : dict, optional
        External chronological evidence; absent on the October source.

    Returns
    -------
    dict[str,pandas.DataFrame]
        Complete ledgers, conditional diagnostics and temporal audits.
    """
    ledger, members = enumerate_segments(wagers, cfg)
    by_id = wagers.set_index("bet_id", drop=False)
    records = []
    for candidate, group in members.groupby("candidate_id", sort=True):
        seed = int(fingerprint([cfg.seed, candidate])[:16], 16)
        records.append(
            {
                "candidate_id": candidate,
                **evaluate_candidate(by_id.loc[group.bet_id].reset_index(drop=True), cfg, seed),
            }
        )
    if records:
        results = ledger.merge(pd.DataFrame(records), on="candidate_id", validate="one_to_one")
    else:
        sample = wagers.iloc[:0].copy()
        results = ledger.copy()
        for key, value in evaluate_candidate(sample, cfg, cfg.seed).items():
            results[key] = pd.Series(dtype="object" if isinstance(value, str) else "float64")
    results = adjust_family(results, cfg.alpha)
    return {
        "candidate_ledger": ledger,
        "candidate_memberships": members,
        "candidate_results": results,
        **build_walk_forward(wagers, cfg, evidence),
    }


def load_discovery(path: Path) -> dict[str, pd.DataFrame]:
    """Verify and load one explicit report.

    Parameters
    ----------
    path : pathlib.Path
        Published discovery directory.

    Returns
    -------
    dict[str,pandas.DataFrame]
        Hash-verified tables.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if fingerprint(manifest["specification"]) != manifest["report_id"]:
        raise ValueError("Discovery specification integrity failure")
    tables = {}
    for name in TABLES:
        file = name + ".parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        tables[name] = pd.read_parquet(path / file)
        if len(tables[name]) != manifest["table_rows"][name]:
            raise ValueError("Discovery count integrity failure")
    return tables


def publish_discovery(
    atlas: Path, output_root: Path, cfg: DiscoveryConfig, evidence: dict | None = None
) -> Path:
    """Publish an immutable report pinned to source, policy, code and evidence.

    Parameters
    ----------
    atlas : pathlib.Path
        Explicit existing atlas bundle.
    output_root : pathlib.Path
        Private output directory.
    cfg : DiscoveryConfig
        Search/inference policy.
    evidence : dict, optional
        Supporting temporal attestations, recorded in report provenance.

    Returns
    -------
    pathlib.Path
        Published report. Repeating a specification checks logical equality.
    """
    source = load_atlas(atlas)
    manifest = json.loads((atlas / "manifest.json").read_text())
    analysis = Path(__file__).resolve().parents[1] / "analysis"
    files = [
        analysis / (name + ".py")
        for name in [
            "discovery_config",
            "segment_search",
            "robustness",
            "multiple_testing",
            "walk_forward",
        ]
    ] + [Path(__file__)]
    spec = {
        "atlas_id": manifest["atlas_id"],
        "atlas_manifest_sha256": hashlib.sha256((atlas / "manifest.json").read_bytes()).hexdigest(),
        "canonical_bundle_id": manifest["specification"]["canonical_bundle_id"],
        "policy": asdict(cfg),
        "temporal_evidence": evidence or {},
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "limitations": {
            "fees": "unverified",
            "events": "dependence_proxies",
            "coverage": "logged_records_only",
            "interpretation": "exploratory_not_strategy_validation",
        },
    }
    ident = fingerprint(spec)
    tables = build_discovery(source["wager_dimensions"], cfg, evidence)
    for frame in tables.values():
        for col in frame.select_dtypes(include="object"):
            frame[col] = frame[col].where(frame[col].notna(), None)
        frame["atlas_id"] = manifest["atlas_id"]
        frame["discovery_contract_version"] = cfg.contract_version
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            previous = load_discovery(destination)
            for name in TABLES:
                assert_frame_equal(tables[name], previous[name])
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
                    "report_id": ident,
                    "specification": spec,
                    "output_sha256": hashes,
                    "table_rows": {k: len(v) for k, v in tables.items()},
                },
            )
            stage.rename(destination)
    return destination
