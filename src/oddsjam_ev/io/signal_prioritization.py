"""
signal_prioritization.py
===============================================================================
Purpose
-------
Publish immutable exploratory statistics, overlap audits and transparent shortlists.

What this module provides
-------------------------
Pinned-family validation, approved policy lineage, failed-budget audits and replay.

Design Notes
------------
No default dashboard/pipeline behavior or historical outputs are changed. Statistical
flags remain conditional/exploratory; no production authorization is created.
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
import scipy
from pandas.testing import assert_frame_equal

from oddsjam_ev.analysis.paired_comparison import PrioritizationBudgetExceeded
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.analysis.signal_prioritization import build_prioritization
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.parquet import read_parquet_file
from oddsjam_ev.io.sizing_diagnostics import load_sizing_diagnostics
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery

TABLES = (
    "signal_statistics",
    "signal_hypotheses",
    "signal_overlap",
    "signal_priorities",
    "signal_shortlist",
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prioritization(path: Path) -> dict:
    """Verify an explicit complete prioritization publication and its correction audit.

    Parameters
    ----------
    path : pathlib.Path
        Private immutable publication directory.

    Returns
    -------
    dict
        Full statistics/family/overlap/priorities and presentation-only shortlist.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    PrioritizationConfig(**spec["policy"])
    if spec.get("contract_version") != 1 or fingerprint(spec) != manifest["publication_id"]:
        raise ValueError("Prioritization specification integrity failure")
    audit = json.loads(verify_bytes(path / "audit.json", manifest["output_sha256"]["audit.json"]))
    if audit != manifest["audit"] or not audit["family_preserved_after_suppression"]:
        raise ValueError("Incomplete prioritization family")
    result = {}
    for name in TABLES:
        file = name + ".parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        result[name] = read_parquet_file(path / file)
        if len(result[name]) != manifest["table_rows"][name]:
            raise ValueError("Prioritization row count mismatch")
    if len(result["signal_hypotheses"]) != 4 * audit["candidate_count"]:
        raise ValueError("Directional family count mismatch")
    return result


def publish_prioritization(
    discovery: Path, diagnostics: Path, output_root: Path, cfg: PrioritizationConfig
) -> Path:
    """Publish exploratory evidence with strict family/input integrity and safe replay.

    Parameters
    ----------
    discovery, diagnostics : pathlib.Path
        Verified complete within-filter family and its exact sizing-diagnostic input.
    output_root : pathlib.Path
        Private destination outside input publications.
    cfg : PrioritizationConfig
        User-reviewed gates, four-hypothesis correction and presentation limits.

    Returns
    -------
    pathlib.Path
        Immutable publication. Resource overflows write a separate failure audit
        and raise without making partial statistics or shortlists consumable.
    """
    discovery, diagnostics, output_root = map(Path, (discovery, diagnostics, output_root))
    if any(output_root.resolve().is_relative_to(p.resolve()) for p in (discovery, diagnostics)):
        raise ValueError("Prioritization output must not modify source publications")
    family = load_within_filter_discovery(discovery)
    data = load_sizing_diagnostics(diagnostics)
    dm = json.loads((discovery / "manifest.json").read_text())
    sm = json.loads((diagnostics / "manifest.json").read_text())
    if dm["specification"]["diagnostics_id"] != sm["diagnostics_id"] or dm["specification"][
        "input_manifest_sha256"
    ]["diagnostics"] != _digest(diagnostics / "manifest.json"):
        raise ValueError("Prioritization diagnostic provenance mismatch")
    analysis = Path(__file__).parents[1] / "analysis"
    files = [
        Path(__file__),
        *[
            analysis / (n + ".py")
            for n in (
                "signal_prioritization",
                "paired_comparison",
                "prioritization_config",
                "multiple_testing",
            )
        ],
    ]
    spec = {
        "contract_version": 1,
        "policy": asdict(cfg),
        "discovery_publication_id": dm["publication_id"],
        "discovery_manifest_sha256": _digest(discovery / "manifest.json"),
        "diagnostics_id": sm["diagnostics_id"],
        "diagnostics_manifest_sha256": _digest(diagnostics / "manifest.json"),
        "code_sha256": {p.parent.name + "/" + p.name: _digest(p) for p in files},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "correction_scope": "entire_frozen_family_x_weighted_and_equal_stake_x_positive_and_negative_including_blocked_tests",
        "overlap_policy": "audit_all_nonempty_overlap; suppress_only_exact_equivalence_within_same_parent_and_category_after_correction",
    }
    ident = fingerprint(spec)
    try:
        tables, audit = build_prioritization(family, data["candidate_diagnostics"], cfg)
    except PrioritizationBudgetExceeded as exc:
        failure = {
            **exc.audit,
            "status": "aborted_budget_exceeded_not_truncated",
            "input_specification": spec,
        }
        with writer_lock(output_root):
            root = output_root / "failures"
            root.mkdir(exist_ok=True)
            path = root / (fingerprint(failure) + ".json")
            if path.exists():
                if json.loads(path.read_text()) != failure:
                    raise ValueError("Prioritization budget replay mismatch") from exc
            else:
                write_json(path, failure)
        raise
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            old = load_prioritization(destination)
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
            loaded = load_prioritization(stage)
            for name in TABLES:
                assert_frame_equal(tables[name], loaded[name])
            stage.rename(destination)
    return destination
