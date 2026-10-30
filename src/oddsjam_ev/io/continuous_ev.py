"""
continuous_ev.py
===============================================================================
Purpose
-------
Publish optional continuous-EV diagnostics from explicit verified canonical inputs.

What this module provides
-------------------------
Versioned typed EV tables, provenance, coverage, hash verification and safe replay.

Design Notes
------------
Existing canonical/atlas schemas and historical metrics remain unchanged. This is
not a discovery search, calibration result, ticket EV model or dashboard rebuild.
===============================================================================
"""

import hashlib
import json
import platform
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.continuous_ev import CONTRACT, build_ev_tables
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.source_csv import fingerprint

TABLES = ("source_ev", "wager_ev")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_continuous_ev(path: Path) -> dict:
    """Verify the contract and table bytes of one explicit publication.

    Parameters
    ----------
    path : pathlib.Path
        Immutable EV publication directory.

    Returns
    -------
    dict
        Typed source and candidate tables with no economic amounts.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    if spec["ev_contract"].get("contract_version") != 1 or fingerprint(spec) != manifest["ev_id"]:
        raise ValueError("Continuous EV specification integrity failure")
    result = {}
    for name in TABLES:
        file = path / (name + ".parquet")
        verify_bytes(file, manifest["output_sha256"][file.name])
        result[name] = read_parquet_file(file)
        if len(result[name]) != manifest["summary"][name]["rows"]:
            raise ValueError("Continuous EV row-count mismatch")
    return result


def publish_continuous_ev(canonical: Path, output_root: Path) -> Path:
    """Publish additive exact-EV views without rebuilding historical metrics.

    Parameters
    ----------
    canonical : pathlib.Path
        Explicit verified canonical bundle with source strings and identity links.
    output_root : pathlib.Path
        Private destination outside the input bundle.

    Returns
    -------
    pathlib.Path
        Content-addressed publication, or verified identical replay.
    """
    canonical, output_root = Path(canonical), Path(output_root)
    if output_root.resolve().is_relative_to(canonical.resolve()):
        raise ValueError("EV output must not modify source bundle")
    source = load_canonical_bundle(canonical)
    manifest = json.loads((canonical / "manifest.json").read_text())
    tables = build_ev_tables(source)
    analysis = Path(__file__).parents[1] / "analysis" / "continuous_ev.py"
    spec = {
        "ev_contract": CONTRACT,
        "canonical_bundle_id": manifest["bundle_id"],
        "canonical_manifest_sha256": _digest(canonical / "manifest.json"),
        "snapshot_id": manifest["specification"]["ingestion_manifest"]["snapshot_id"],
        "source_table_sha256": {
            name: manifest["output_sha256"][name + ".parquet"]
            for name in ("source_features", "bet_facts")
        },
        "code_sha256": {
            "analysis/continuous_ev.py": _digest(analysis),
            "io/continuous_ev.py": _digest(Path(__file__)),
        },
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "python_version": platform.python_version(),
    }
    summary = {
        name: {
            "rows": len(frame),
            "status_counts": {str(k): int(v) for k, v in frame.ev_status.value_counts().items()},
        }
        for name, frame in tables.items()
    }
    ident = fingerprint(spec)
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            loaded = load_continuous_ev(destination)
            for name in TABLES:
                assert_frame_equal(tables[name], loaded[name])
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            for name, frame in tables.items():
                frame.to_parquet(stage / (name + ".parquet"), index=False)
            write_json(
                stage / "manifest.json",
                {
                    "ev_id": ident,
                    "specification": spec,
                    "summary": summary,
                    "output_sha256": {
                        name + ".parquet": _digest(stage / (name + ".parquet")) for name in TABLES
                    },
                },
            )
            loaded = load_continuous_ev(stage)
            for name in TABLES:
                assert_frame_equal(tables[name], loaded[name])
            stage.rename(destination)
        return destination
