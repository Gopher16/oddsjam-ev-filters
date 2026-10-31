"""
visual_diagnostics.py
===============================================================================
Purpose
-------
Publish optional visual datasets from exact canonical, atlas and sizing publications.

What this module provides
-------------------------
Full lineage, immutable replay, verified tables, chart readiness and failed-work audits.

Design Notes
------------
No notebook execution, dashboard fitting, source mutation or current-default changes.
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
import pyarrow
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.visual_config import VisualConfig
from oddsjam_ev.analysis.visual_diagnostics import VisualBudgetExceeded, build_visual_diagnostics
from oddsjam_ev.io.atlas import load_atlas
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.sizing_diagnostics import load_sizing_diagnostics
from oddsjam_ev.io.source_csv import fingerprint

TABLES = (
    "visual_candidates",
    "visual_memberships",
    "visual_cohorts",
    "visual_periods",
    "visual_activity",
    "visual_coverage",
    "visual_readiness",
)


def _normalized(frame):
    # Parquet may decode nullable object cells as None or NaN. Compare their
    # semantic nulls consistently without suppressing dtype/value drift.
    result = frame.copy()
    for column in result.select_dtypes(include="object"):
        result[column] = result[column].where(result[column].notna(), None)
    return result


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_visual_diagnostics(path: Path) -> dict:
    """Verify an explicit complete visual publication; never find a latest snapshot.

    Parameters
    ----------
    path : pathlib.Path
        Immutable publication directory.

    Returns
    -------
    dict
        Persisted chart tables, verified against specification and audit hashes.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    VisualConfig(**spec["policy"])
    if spec["contract_version"] != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Visual specification integrity failure")
    audit = json.loads(verify_bytes(path / "audit.json", manifest["output_sha256"]["audit.json"]))
    if audit != manifest["audit"]:
        raise ValueError("Visual audit mismatch")
    result = {}
    for name in TABLES:
        file = name + ".parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        result[name] = read_parquet_file(path / file)
        if len(result[name]) != audit["table_rows"][name]:
            raise ValueError("Visual row count mismatch")
    return result


def publish_visual_diagnostics(
    canonical: Path, atlas: Path, diagnostics: Path, output_root: Path, cfg: VisualConfig
) -> Path:
    """Build immutable visual data only from mutually pinned verified inputs.

    Parameters
    ----------
    canonical, atlas, diagnostics : pathlib.Path
        Explicit source publications sharing exact canonical manifest bytes.
    output_root : pathlib.Path
        Separate private visual publication root.
    cfg : VisualConfig
        Versioned descriptive display/source-evidence policy and resource bounds.

    Returns
    -------
    pathlib.Path
        Verified publication, or identical replay. Failed budgets write only a
        failure audit, never incomplete chart tables/manifests.
    """
    canonical, atlas, diagnostics, output_root = map(
        Path, (canonical, atlas, diagnostics, output_root)
    )
    if any(
        output_root.resolve().is_relative_to(p.resolve()) for p in (canonical, atlas, diagnostics)
    ):
        raise ValueError("Visual output must not modify source publications")
    source = load_canonical_bundle(canonical)
    dimensions = load_atlas(atlas)
    sizing = load_sizing_diagnostics(diagnostics)
    manifests = {
        k: json.loads((p / "manifest.json").read_text())
        for k, p in (("canonical", canonical), ("atlas", atlas), ("diagnostics", diagnostics))
    }
    cm = manifests["canonical"]
    for name in ("atlas", "diagnostics"):
        spec = manifests[name]["specification"]
        if spec["canonical_bundle_id"] != cm["bundle_id"] or spec[
            "canonical_manifest_sha256"
        ] != _digest(canonical / "manifest.json"):
            raise ValueError("Visual source canonical provenance mismatch")
    analysis = Path(__file__).parents[1] / "analysis"
    files = [
        Path(__file__),
        *[
            analysis / (name + ".py")
            for name in ("visual_config", "visual_features", "visual_diagnostics", "continuous_ev")
        ],
    ]
    spec = {
        "contract_version": 1,
        "policy": asdict(cfg),
        "canonical_bundle_id": cm["bundle_id"],
        "atlas_id": manifests["atlas"]["atlas_id"],
        "diagnostics_id": manifests["diagnostics"]["diagnostics_id"],
        "source_manifest_sha256": {
            k: _digest(p / "manifest.json")
            for k, p in (("canonical", canonical), ("atlas", atlas), ("diagnostics", diagnostics))
        },
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "units": {
            "money": "source_reported_USD_or_declared_notional_USD",
            "return_and_ev": "fractions_not_percentage_points",
            "source_ev_percent": "percentage_points_preserved_with_exact_decimal_text",
            "time_to_event": "hours",
            "activity": "unique_logged_candidates_per_active_source_day",
        },
        "interval_policy": "descriptive_only_no_new_tests_or_intervals",
        "date_policy": "source_wall_calendar_attribution_not_verified_execution_or_settlement",
    }
    ident = fingerprint(spec)
    try:
        tables, audit = build_visual_diagnostics(
            source, dimensions["wager_dimensions"], sizing["candidate_diagnostics"], cfg
        )
    except VisualBudgetExceeded as exc:
        failure = {
            "status": "aborted_budget_not_truncated",
            "reason": str(exc),
            "specification": json.loads(json.dumps(spec)),
        }
        with writer_lock(output_root):
            root = output_root / "failures"
            root.mkdir(exist_ok=True)
            file = root / (fingerprint(failure) + ".json")
            if file.exists():
                if json.loads(file.read_text()) != failure:
                    raise ValueError("Visual failure replay mismatch") from exc
            else:
                write_json(file, failure)
        raise
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old = load_visual_diagnostics(destination)
            for name in TABLES:
                assert_frame_equal(_normalized(tables[name]), _normalized(old[name]))
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
                    "output_sha256": {
                        name: _digest(stage / name)
                        for name in [*[n + ".parquet" for n in TABLES], "audit.json"]
                    },
                },
            )
            loaded = load_visual_diagnostics(stage)
            for name in TABLES:
                assert_frame_equal(_normalized(tables[name]), _normalized(loaded[name]))
            stage.rename(destination)
    return destination
