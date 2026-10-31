"""Pinned input publications, immutable replay, CLI, corruption and budget failures."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_within_filter_publication import sources

from oddsjam_ev.analysis.visual_config import VisualConfig
from oddsjam_ev.analysis.visual_diagnostics import VisualBudgetExceeded
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.visual_diagnostics import load_visual_diagnostics, publish_visual_diagnostics


def test_publication_replay_cli_and_source_preservation(tmp_path):
    canonical, atlas, diagnostics = sources(tmp_path)
    before = {
        p: p.read_bytes()
        for base in (canonical, atlas, diagnostics)
        for p in base.iterdir()
        if p.is_file()
    }
    root = tmp_path / "visuals"
    output = publish_visual_diagnostics(canonical, atlas, diagnostics, root, VisualConfig())
    assert publish_visual_diagnostics(canonical, atlas, diagnostics, root, VisualConfig()) == output
    tables = load_visual_diagnostics(output)
    assert len(tables["visual_candidates"]) == 1
    assert tables["visual_readiness"].chart_id.nunique() == 10
    assert all(p.read_bytes() == contents for p, contents in before.items())
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/build_visual_diagnostics.py"),
        "--canonical",
        str(canonical),
        "--atlas",
        str(atlas),
        "--diagnostics",
        str(diagnostics),
        "--output-root",
        str(tmp_path / "cli"),
    ]
    cli = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert cli.returncode == 0, cli.stderr
    assert Path(cli.stdout.strip()).name == output.name
    with pytest.raises(ValueError, match="source publications"):
        publish_visual_diagnostics(canonical, atlas, diagnostics, atlas / "visuals", VisualConfig())
    (output / "visual_cohorts.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_visual_diagnostics(output)


def test_budget_failure_audit_and_no_partial_publication(tmp_path):
    canonical, atlas, diagnostics = sources(tmp_path)
    root = tmp_path / "visuals"
    for _ in range(2):
        with pytest.raises(VisualBudgetExceeded):
            publish_visual_diagnostics(
                canonical, atlas, diagnostics, root, VisualConfig(max_rows=1)
            )
    failures = list((root / "failures").glob("*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "aborted_budget_not_truncated"
    assert not list(root.glob("*/manifest.json"))


def test_wrong_canonical_lineage_rejected(tmp_path):
    canonical, atlas, diagnostics = sources(tmp_path)
    path = diagnostics / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["specification"]["canonical_manifest_sha256"] = "wrong"
    manifest["diagnostics_id"] = fingerprint(manifest["specification"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        publish_visual_diagnostics(
            canonical, atlas, diagnostics, tmp_path / "visuals", VisualConfig()
        )
