"""Pinned queue lineage, correction bytes, replay and safe failure on corrupt inputs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_prioritization_publication import family_sources

from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.analysis.review_catalogue import ReviewBudgetExceeded
from oddsjam_ev.analysis.review_config import ReviewConfig
from oddsjam_ev.analysis.visual_config import VisualConfig
from oddsjam_ev.io.decision_records import publish_decisions
from oddsjam_ev.io.review_queues import load_review_queues, publish_review_queues
from oddsjam_ev.io.signal_prioritization import publish_prioritization
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.visual_diagnostics import publish_visual_diagnostics


def queue_sources(tmp_path):
    family, diagnostics = family_sources(tmp_path)
    priorities = publish_prioritization(
        family, diagnostics, tmp_path / "priorities", PrioritizationConfig()
    )
    decisions = publish_decisions(
        family,
        priorities,
        {"contract_version": 1, "baselines": [], "lifecycle": [], "translations": []},
        tmp_path / "decisions",
        "2030-01-01T00:00:00Z",
        "2030-01-01T00:00:00Z",
    )
    canonical = next((tmp_path / "canonical").glob("*/manifest.json")).parent
    atlas = next((tmp_path / "atlas").glob("*/manifest.json")).parent
    visuals = publish_visual_diagnostics(
        canonical, atlas, diagnostics, tmp_path / "visuals", VisualConfig()
    )
    return family, priorities, decisions, visuals


def test_replay_original_correction_bytes_and_cli(tmp_path):
    sources = queue_sources(tmp_path)
    before = {p: p.read_bytes() for root in sources for p in root.iterdir() if p.is_file()}
    root = tmp_path / "queues"
    output = publish_review_queues(*sources, root, ReviewConfig())
    assert publish_review_queues(*sources, root, ReviewConfig()) == output
    tables = load_review_queues(output)
    assert tables["audit"]["source_candidate_count"] == 2
    assert tables["audit"]["source_hypothesis_count"] == 8
    assert not any(r["headline"] for r in tables["records"])
    assert all(p.read_bytes() == data for p, data in before.items())
    assert (output / "source_hypotheses.parquet").read_bytes() == (
        sources[1] / "signal_hypotheses.parquet"
    ).read_bytes()
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/build_review_queues.py"),
    ]
    for name, value in zip(
        ("discovery", "prioritization", "decisions", "visuals"), sources, strict=True
    ):
        command.extend(["--" + name, str(value)])
    command.extend(["--output-root", str(tmp_path / "cli")])
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert Path(run.stdout.strip()).name == output.name
    with pytest.raises(ValueError, match="source publications"):
        publish_review_queues(*sources, sources[0] / "queue", ReviewConfig())
    (output / "review.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_review_queues(output)


def test_budget_failure_audit_and_wrong_source_rejected(tmp_path):
    sources = queue_sources(tmp_path)
    root = tmp_path / "queues"
    for _ in range(2):
        with pytest.raises(ReviewBudgetExceeded):
            publish_review_queues(*sources, root, ReviewConfig(max_candidates=1))
    assert not list(root.glob("*/manifest.json"))
    failures = list((root / "failures").glob("*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "aborted_budget_not_truncated"
    path = sources[3] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["specification"]["diagnostics_id"] = "wrong"
    manifest["publication_id"] = fingerprint(manifest["specification"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lineage"):
        publish_review_queues(*sources, root, ReviewConfig())
