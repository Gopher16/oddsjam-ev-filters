"""
sizing_diagnostics.py
===============================================================================
Purpose
-------
Publish immutable sizing-aware diagnostics from matched verified evidence bundles.

What this module provides
-------------------------
Explicit policy/input lineage, cohort reconciliation, verified loading and replay.

Design Notes
------------
Default research outputs and metric definitions are preserved. New arithmetic
comparisons are hypothetical; this optional stage never runs from dashboard UI.
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

from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig, build_sizing_diagnostics
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.sizing_history import load_sizing_history
from oddsjam_ev.io.source_csv import fingerprint

TABLES = ("candidate_diagnostics", "cohort_metrics")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sizing_diagnostics(path: Path) -> dict:
    """Verify one publication's policy, tables and audit bytes.

    Parameters
    ----------
    path : pathlib.Path
        Explicit private publication directory.

    Returns
    -------
    dict
        Persisted candidate diagnostics and paired cohort metrics.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    SizingDiagnosticsConfig(**spec["policy"])
    if fingerprint(spec) != manifest["diagnostics_id"]:
        raise ValueError("Sizing diagnostics specification integrity failure")
    result = {}
    for name in TABLES:
        file = path / (name + ".parquet")
        verify_bytes(file, manifest["output_sha256"][file.name])
        result[name] = read_parquet_file(file)
        if len(result[name]) != manifest["summary"]["table_rows"][name]:
            raise ValueError("Sizing diagnostics row mismatch")
    audit = json.loads(verify_bytes(path / "audit.json", manifest["output_sha256"]["audit.json"]))
    if audit != manifest["summary"]:
        raise ValueError("Sizing diagnostics audit mismatch")
    return result


def publish_sizing_diagnostics(
    canonical: Path, sizing: Path, output_root: Path, cfg: SizingDiagnosticsConfig
) -> Path:
    """Publish paired diagnostics with exact sizing/canonical provenance checks.

    Parameters
    ----------
    canonical, sizing : pathlib.Path
        Explicit verified canonical bundle and sizing evidence tied to that bundle.
    output_root : pathlib.Path
        Private output destination outside source publications.
    cfg : SizingDiagnosticsConfig
        Declared notional unit and event-proxy interval policy.

    Returns
    -------
    pathlib.Path
        New immutable publication, or verified identical replay.
    """
    canonical, sizing, output_root = map(Path, (canonical, sizing, output_root))
    if any(output_root.resolve().is_relative_to(p.resolve()) for p in (canonical, sizing)):
        raise ValueError("Sizing diagnostic output must not modify source publications")
    source = load_canonical_bundle(canonical)
    _, linkage, _ = load_sizing_history(sizing)
    cm = json.loads((canonical / "manifest.json").read_text())
    sm = json.loads((sizing / "manifest.json").read_text())
    if sm["specification"]["canonical_bundle_id"] != cm["bundle_id"] or sm["specification"][
        "canonical_manifest_sha256"
    ] != _digest(canonical / "manifest.json"):
        raise ValueError("Sizing history does not match canonical provenance")
    tables = build_sizing_diagnostics(source, linkage, cfg)
    data = tables["candidate_diagnostics"]
    metrics = tables["cohort_metrics"]
    overall = metrics.loc[metrics["view"].eq("overall")]
    regimes = metrics.loc[metrics["view"].eq("regime")]
    if (
        len(data) != len(source["bet_facts"])
        or int(overall.candidate_count.sum()) != len(data)
        or int(regimes.candidate_count.sum()) != len(data)
    ):
        raise ValueError("Sizing candidate partitions do not reconcile")
    policy = asdict(cfg)
    analysis = Path(__file__).parents[1] / "analysis"
    files = [
        Path(__file__),
        analysis / "sizing_diagnostics.py",
        analysis / "robustness.py",
        analysis / "discovery_config.py",
    ]
    spec = {
        "policy": policy,
        "canonical_bundle_id": cm["bundle_id"],
        "canonical_manifest_sha256": _digest(canonical / "manifest.json"),
        "sizing_publication_id": sm["publication_id"],
        "sizing_manifest_sha256": _digest(sizing / "manifest.json"),
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "interval_method": "paired_namespace_event_cluster_percentile_bootstrap_conditional_on_known_proxies_not_proven_independent",
        "normalization_assumptions": "proportional_profit_scaling_hypothetical_not_Kelly_or_bankroll_replay",
    }
    audit = {
        "table_rows": {name: len(frame) for name, frame in tables.items()},
        "canonical_candidates": len(data),
        "canonical_eligible_candidates": int(data.reported_roi_eligible.sum()),
        "diagnostic_eligible_candidates": int(data.diagnostic_eligible.sum()),
        "additional_excluded_candidates": int(
            (data.reported_roi_eligible & ~data.diagnostic_eligible).sum()
        ),
        "diagnostic_exclusion_combinations": {
            str(k): int(v) for k, v in data.diagnostic_exclusions_json.value_counts().items()
        },
        "unknown_regime_candidates": int(data.sizing_regime.eq("<unknown>").sum()),
        "bankroll_relative_available_candidates": int(data.bankroll_relative_stake.notna().sum()),
        "overall_and_regime_partitions_reconcile": True,
        "filter_views_non_additive": True,
        "sizing_replay_blocked": True,
        "pvalues_or_rankings_computed": False,
    }
    ident = fingerprint(spec)
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            existing = load_sizing_diagnostics(destination)
            for name in TABLES:
                assert_frame_equal(tables[name], existing[name])
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as temporary:
            stage = Path(temporary)
            for name, frame in tables.items():
                frame.to_parquet(stage / (name + ".parquet"), index=False)
            write_json(stage / "audit.json", audit)
            write_json(
                stage / "manifest.json",
                {
                    "diagnostics_id": ident,
                    "specification": spec,
                    "summary": audit,
                    "output_sha256": {
                        name: _digest(stage / name)
                        for name in (*[n + ".parquet" for n in TABLES], "audit.json")
                    },
                },
            )
            existing = load_sizing_diagnostics(stage)
            for name in TABLES:
                assert_frame_equal(tables[name], existing[name])
            stage.rename(destination)
    return destination
