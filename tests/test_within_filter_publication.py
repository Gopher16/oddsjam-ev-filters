"""Verified within-filter publication and explicit failure audits; no source mutation."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from test_atlas_bundle import canonical

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.analysis.multidimensional_config import MultidimensionalConfig
from oddsjam_ev.analysis.multidimensional_discovery import DiscoveryBudgetExceeded
from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig
from oddsjam_ev.io.atlas import publish_atlas
from oddsjam_ev.io.sizing_diagnostics import publish_sizing_diagnostics
from oddsjam_ev.io.sizing_history import publish_sizing_history
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.within_filter_discovery import (
    load_within_filter_discovery,
    publish_within_filter_discovery,
)


def sources(tmp_path):
    canonical_path = canonical(tmp_path)
    atlas = publish_atlas(canonical_path, tmp_path / "atlas", AtlasConfig())
    sizing = publish_sizing_history(
        canonical_path, tmp_path / "sizing", knowledge_as_of="2026-03-02T00:00:00Z"
    )
    diagnostics = publish_sizing_diagnostics(
        canonical_path, sizing, tmp_path / "diagnostics", SizingDiagnosticsConfig()
    )
    return canonical_path, atlas, diagnostics


def test_publication_replay_sources_and_cli(tmp_path):
    canonical_path, atlas, diagnostics = sources(tmp_path)
    before = {
        p: p.read_bytes()
        for base in (canonical_path, atlas, diagnostics)
        for p in base.iterdir()
        if p.is_file()
    }
    cfg = MultidimensionalConfig(views=[["sportsbook"]])
    root = tmp_path / "results"
    output = publish_within_filter_discovery(canonical_path, atlas, diagnostics, root, cfg)
    assert publish_within_filter_discovery(canonical_path, atlas, diagnostics, root, cfg) == output
    result = load_within_filter_discovery(output)
    assert len(result["signal_candidates"]) == 1
    assert result["signal_comparisons"].partition_reconciles.all()
    assert all(p.read_bytes() == value for p, value in before.items())
    config_path = tmp_path / "search.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg)))
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/build_within_filter_discovery.py"),
        "--canonical",
        str(canonical_path),
        "--atlas",
        str(atlas),
        "--diagnostics",
        str(diagnostics),
        "--config",
        str(config_path),
        "--output-root",
        str(tmp_path / "cli"),
    ]
    cli = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert cli.returncode == 0, cli.stderr
    with pytest.raises(ValueError, match="source publications"):
        publish_within_filter_discovery(canonical_path, atlas, diagnostics, atlas / "results", cfg)
    (output / "signal_comparisons.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_within_filter_discovery(output)


def test_budget_failure_publishes_audit_not_partial_family(tmp_path):
    canonical_path, atlas, diagnostics = sources(tmp_path)
    root = tmp_path / "results"
    cfg = MultidimensionalConfig(views=[["sportsbook"]], max_scan_rows=1)
    for _ in range(2):
        with pytest.raises(DiscoveryBudgetExceeded):
            publish_within_filter_discovery(canonical_path, atlas, diagnostics, root, cfg)
    failures = list((root / "failures").glob("*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "aborted_budget_exceeded_not_truncated"
    assert not list(root.glob("*/manifest.json"))


def test_wrong_input_lineage_rejected(tmp_path):
    canonical_path, atlas, diagnostics = sources(tmp_path)
    path = atlas / "manifest.json"
    data = json.loads(path.read_text())
    data["specification"]["canonical_bundle_id"] = "wrong"
    data["atlas_id"] = fingerprint(data["specification"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        publish_within_filter_discovery(
            canonical_path, atlas, diagnostics, tmp_path / "results", MultidimensionalConfig()
        )
