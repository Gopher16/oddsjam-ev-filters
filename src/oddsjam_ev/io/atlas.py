"""
atlas.py
===============================================================================
Purpose
-------
Publish immutable descriptive atlas outputs from an explicit canonical bundle.

What this module provides
-------------------------
Hash-verified reading, atomic publication, and reproducible replay.

Design Notes
------------
Atlas output is private and never changes ingestion or legacy research defaults.
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

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.analysis.opportunity_atlas import build_atlas
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.source_csv import fingerprint

TABLES = (
    "wager_dimensions",
    "opportunity_dimensions",
    "wager_segments",
    "opportunity_segments",
    "filter_segment_diagnostics",
    "atlas_quality",
)


def load_atlas(path: Path) -> dict[str, pd.DataFrame]:
    """Load verified atlas tables.

    Parameters
    ----------
    path : pathlib.Path
        Explicit published atlas directory.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Integrity-checked tables.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    if fingerprint(manifest["specification"]) != manifest["atlas_id"]:
        raise ValueError("Atlas specification integrity failure")
    result = {}
    for name in TABLES:
        file = name + ".parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        result[name] = pd.read_parquet(path / file)
        if len(result[name]) != manifest["table_rows"][name]:
            raise ValueError("Atlas count integrity failure")
    return result


def publish_atlas(bundle: Path, output_root: Path, cfg: AtlasConfig) -> Path:
    """Build and atomically publish a pinned atlas; replay checks logical equality.

    Parameters
    ----------
    bundle : pathlib.Path
        Explicit canonical bundle; no latest selection.
    output_root : pathlib.Path
        Private output directory.
    cfg : AtlasConfig
        Display policy and market mapping.

    Returns
    -------
    pathlib.Path
        Immutable atlas directory with provenance manifest.
    """
    source = load_canonical_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    module = Path(__file__).resolve().parents[1] / "analysis"
    files = [module / "atlas_config.py", module / "opportunity_atlas.py", Path(__file__)]
    spec = {
        "canonical_manifest_sha256": hashlib.sha256(
            (bundle / "manifest.json").read_bytes()
        ).hexdigest(),
        "canonical_bundle_id": manifest["bundle_id"],
        "ingestion_snapshot_id": manifest["specification"]["ingestion_manifest"]["snapshot_id"],
        "policy": asdict(cfg),
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "limitations": {
            "coverage": "logged_records_only",
            "chronology": "unverified",
            "fees": "unverified",
            "opportunity_identity": "proxy",
            "uncertainty": "not_estimated_dependence_unverified",
        },
    }
    ident = fingerprint(spec)
    tables = build_atlas(source, cfg)
    for frame in tables.values():
        frame["canonical_bundle_id"] = manifest["bundle_id"]
        frame["ingestion_snapshot_id"] = spec["ingestion_snapshot_id"]
        frame["atlas_contract_version"] = cfg.contract_version
        for col in frame.select_dtypes(include="object"):
            frame[col] = frame[col].where(frame[col].notna(), None)
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            existing = load_atlas(destination)
            for name in TABLES:
                assert_frame_equal(tables[name], existing[name])
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
                    "atlas_id": ident,
                    "specification": spec,
                    "output_sha256": hashes,
                    "table_rows": {k: len(v) for k, v in tables.items()},
                },
            )
            stage.rename(destination)
    return destination
