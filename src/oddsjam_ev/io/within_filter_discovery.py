"""
within_filter_discovery.py
===============================================================================
Purpose
-------
Publish bounded descriptive within-filter families from verified pinned evidence.

What this module provides
-------------------------
Strict provenance, immutable tables, failed-budget audits and deterministic replay.

Design Notes
------------
No old report is replaced. Budget failures never publish a partial candidate family.
This optional stage does not rank, optimize, acquire data or run on dashboard load.
===============================================================================
"""

import hashlib
import json
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.multidimensional_config import MultidimensionalConfig
from oddsjam_ev.analysis.multidimensional_discovery import (
    DiscoveryBudgetExceeded,
    build_within_filter_discovery,
)
from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig
from oddsjam_ev.io.atlas import load_atlas
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.sizing_diagnostics import load_sizing_diagnostics
from oddsjam_ev.io.source_csv import fingerprint

TABLES = (
    "signal_candidates",
    "signal_evidence",
    "signal_comparisons",
    "signal_membership",
    "signal_exclusions",
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_within_filter_discovery(path: Path) -> dict:
    """Verify one complete descriptive family, not an inferential shortlist.

    Parameters
    ----------
    path : pathlib.Path
        Explicit immutable publication directory.

    Returns
    -------
    dict
        Candidate, descriptive evidence, comparison and membership tables.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    MultidimensionalConfig(**spec["policy"])
    if spec.get("contract_version") != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Within-filter specification integrity failure")
    audit = json.loads(verify_bytes(path / "audit.json", manifest["output_sha256"]["audit.json"]))
    if audit != manifest["audit"] or audit["status"] != "complete_descriptive_family":
        raise ValueError("Incomplete within-filter family")
    tables = {}
    for name in TABLES:
        file = name + ".parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        tables[name] = read_parquet_file(path / file)
        if len(tables[name]) != manifest["table_rows"][name]:
            raise ValueError("Within-filter table count mismatch")
    return tables


def publish_within_filter_discovery(
    canonical: Path, atlas: Path, diagnostics: Path, output_root: Path, cfg: MultidimensionalConfig
) -> Path:
    """Freeze, evaluate and publish a fully accounted descriptive candidate family.

    Parameters
    ----------
    canonical, atlas, diagnostics : pathlib.Path
        Verified publications pinned to the same exact canonical manifest.
    output_root : pathlib.Path
        Private root outside each source publication.
    cfg : MultidimensionalConfig
        Reviewed operational bounds and declared descriptive feature predicates.

    Returns
    -------
    pathlib.Path
        Immutable result, or verified replay. Budget overflow writes a separate
        private failure audit then raises; no partial result becomes consumable.
    """
    canonical, atlas, diagnostics, output_root = map(
        Path, (canonical, atlas, diagnostics, output_root)
    )
    if any(
        output_root.resolve().is_relative_to(p.resolve()) for p in (canonical, atlas, diagnostics)
    ):
        raise ValueError("Discovery output must not modify source publications")
    source = load_canonical_bundle(canonical)
    dimensions = load_atlas(atlas)
    sizing = load_sizing_diagnostics(diagnostics)
    manifests = {
        name: json.loads((p / "manifest.json").read_text())
        for name, p in (("canonical", canonical), ("atlas", atlas), ("diagnostics", diagnostics))
    }
    cm = manifests["canonical"]
    digest = _digest(canonical / "manifest.json")
    for name in ("atlas", "diagnostics"):
        spec = manifests[name]["specification"]
        if (
            spec["canonical_bundle_id"] != cm["bundle_id"]
            or spec["canonical_manifest_sha256"] != digest
        ):
            raise ValueError("Within-filter source provenance mismatch")
    analysis = Path(__file__).parents[1] / "analysis"
    files = [
        Path(__file__),
        *[
            analysis / (n + ".py")
            for n in (
                "multidimensional_config",
                "multidimensional_discovery",
                "sizing_diagnostics",
                "continuous_ev",
            )
        ],
    ]
    spec = {
        "contract_version": 1,
        "policy": asdict(cfg),
        "input_manifest_sha256": {
            name: _digest(p / "manifest.json")
            for name, p in (
                ("canonical", canonical),
                ("atlas", atlas),
                ("diagnostics", diagnostics),
            )
        },
        "canonical_bundle_id": cm["bundle_id"],
        "atlas_id": manifests["atlas"]["atlas_id"],
        "diagnostics_id": manifests["diagnostics"]["diagnostics_id"],
        "sizing_policy": manifests["diagnostics"]["specification"]["policy"],
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    ident = fingerprint(spec)
    try:
        tables, audit = build_within_filter_discovery(
            source,
            dimensions["wager_dimensions"],
            sizing["candidate_diagnostics"],
            cfg,
            SizingDiagnosticsConfig(**spec["sizing_policy"]),
            spec,
        )
    except DiscoveryBudgetExceeded as exc:
        failure = {**exc.audit, "input_specification": spec}
        with writer_lock(output_root):
            failed = output_root / "failures"
            failed.mkdir(exist_ok=True)
            target = failed / (fingerprint(failure) + ".json")
            if target.exists():
                if json.loads(target.read_text()) != json.loads(json.dumps(failure)):
                    raise ValueError("Budget audit replay mismatch") from exc
            else:
                write_json(target, failure)
        raise
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old = load_within_filter_discovery(destination)
            for name in TABLES:
                assert_frame_equal(tables[name], old[name])
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as temporary:
            stage = Path(temporary)
            for name, frame in tables.items():
                frame.to_parquet(stage / (name + ".parquet"), index=False)
            write_json(stage / "audit.json", audit)
            write_json(
                stage / "manifest.json",
                {
                    "publication_id": ident,
                    "specification": spec,
                    "audit": audit,
                    "table_rows": {name: len(frame) for name, frame in tables.items()},
                    "output_sha256": {
                        name: _digest(stage / name)
                        for name in (*[n + ".parquet" for n in TABLES], "audit.json")
                    },
                },
            )
            loaded = load_within_filter_discovery(stage)
            for name in TABLES:
                assert_frame_equal(tables[name], loaded[name])
            stage.rename(destination)
    return destination
