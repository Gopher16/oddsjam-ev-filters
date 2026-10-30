"""Verified publication, cohort reconciliation, replay and private source preservation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal
from test_atlas_bundle import canonical

from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.sizing_diagnostics import load_sizing_diagnostics, publish_sizing_diagnostics
from oddsjam_ev.io.sizing_history import publish_sizing_history
from oddsjam_ev.io.source_csv import fingerprint


def sources(tmp_path):
    bundle = canonical(tmp_path)
    sizing = publish_sizing_history(
        bundle, tmp_path / "sizing", knowledge_as_of="2026-03-02T00:00:00Z"
    )
    return bundle, sizing


def test_publication_reconciles_original_metrics_and_replays(tmp_path):
    bundle, sizing = sources(tmp_path)
    before = {p: p.read_bytes() for base in (bundle, sizing) for p in base.iterdir() if p.is_file()}
    cfg = SizingDiagnosticsConfig(min_event_groups=2, bootstrap_resamples=100)
    output = publish_sizing_diagnostics(bundle, sizing, tmp_path / "diagnostics", cfg)
    assert publish_sizing_diagnostics(bundle, sizing, tmp_path / "diagnostics", cfg) == output
    tables = load_sizing_diagnostics(output)
    original = load_canonical_bundle(bundle)["bet_facts"]
    overall = tables["cohort_metrics"].query("view == 'overall'").iloc[0]
    eligible = original.loc[original.reported_roi_eligible]
    assert (
        overall.stake_weighted_roi
        == eligible.reported_profit_usd.sum() / eligible.reported_stake_usd.sum()
    )
    assert overall.additional_excluded_count == 0
    assert tables["candidate_diagnostics"].sizing_regime.eq("<unknown>").all()
    assert all(p.read_bytes() == value for p, value in before.items())
    for name in tables:
        assert_frame_equal(tables[name], load_sizing_diagnostics(output)[name])
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/build_sizing_diagnostics.py"),
            "--canonical",
            str(bundle),
            "--sizing",
            str(sizing),
            "--output-root",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "bankroll_amount" not in result.stdout
    with pytest.raises(ValueError, match="source publications"):
        publish_sizing_diagnostics(bundle, sizing, sizing / "output", cfg)
    (output / "cohort_metrics.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_sizing_diagnostics(output)


def test_mismatched_canonical_and_manifest_tampering(tmp_path):
    bundle, sizing = sources(tmp_path)
    path = sizing / "manifest.json"
    data = json.loads(path.read_text())
    data["specification"]["canonical_bundle_id"] = "wrong"
    data["publication_id"] = fingerprint(data["specification"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        publish_sizing_diagnostics(
            bundle, sizing, tmp_path / "diagnostics", SizingDiagnosticsConfig()
        )
