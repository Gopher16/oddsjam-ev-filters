"""Immutable full-family statistics, replay, source hashes and failed-work audits."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_within_filter_publication import sources

from oddsjam_ev.analysis.multidimensional_config import MultidimensionalConfig
from oddsjam_ev.analysis.paired_comparison import PrioritizationBudgetExceeded
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.io.signal_prioritization import load_prioritization, publish_prioritization
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.within_filter_discovery import publish_within_filter_discovery


def family_sources(tmp_path):
    canonical, atlas, diagnostics = sources(tmp_path)
    family = publish_within_filter_discovery(
        canonical,
        atlas,
        diagnostics,
        tmp_path / "family",
        MultidimensionalConfig(views=[["sportsbook"], ["advertised_ev_percent"]]),
    )
    return family, diagnostics


def test_immutable_full_family_and_cli_replay(tmp_path):
    family, diagnostics = family_sources(tmp_path)
    before = {
        p: p.read_bytes() for base in (family, diagnostics) for p in base.iterdir() if p.is_file()
    }
    root = tmp_path / "priorities"
    cfg = PrioritizationConfig()
    output = publish_prioritization(family, diagnostics, root, cfg)
    assert publish_prioritization(family, diagnostics, root, cfg) == output
    tables = load_prioritization(output)
    assert len(tables["signal_hypotheses"]) == 8
    assert tables["signal_hypotheses"].placeholder_in_correction.all()
    assert tables["signal_shortlist"].empty
    assert all(p.read_bytes() == v for p, v in before.items())
    cli = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/build_signal_prioritization.py"),
            "--discovery",
            str(family),
            "--diagnostics",
            str(diagnostics),
            "--output-root",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode == 0, cli.stderr
    with pytest.raises(ValueError, match="source publications"):
        publish_prioritization(family, diagnostics, family / "output", cfg)
    (output / "signal_hypotheses.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_prioritization(output)


def test_bad_provenance_and_failed_budget_audit(tmp_path):
    family, diagnostics = family_sources(tmp_path)
    path = family / "manifest.json"
    data = json.loads(path.read_text())
    data["specification"]["diagnostics_id"] = "different"
    data["publication_id"] = fingerprint(data["specification"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        publish_prioritization(family, diagnostics, tmp_path / "results", PrioritizationConfig())


def test_budget_failures_never_publish_partial_shortlists(tmp_path, monkeypatch):
    import oddsjam_ev.io.signal_prioritization as module

    family, diagnostics = family_sources(tmp_path)

    def fail(*args, **kwargs):
        raise PrioritizationBudgetExceeded(
            {"reason": "synthetic_work_limit", "requested": 2, "maximum": 1}
        )

    monkeypatch.setattr(module, "build_prioritization", fail)
    root = tmp_path / "results"
    for _ in range(2):
        with pytest.raises(PrioritizationBudgetExceeded):
            publish_prioritization(
                family, diagnostics, root, replace(PrioritizationConfig(), max_pair_comparisons=1)
            )
    assert len(list((root / "failures").glob("*.json"))) == 1
    assert not list(root.glob("*/manifest.json"))
