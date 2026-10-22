"""Verified source linkage, publication, replay and command-line integration."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal
from test_atlas_bundle import canonical

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
from oddsjam_ev.io.atlas import publish_atlas
from oddsjam_ev.io.discovery import publish_discovery
from oddsjam_ev.io.recommendations import load_recommendations, publish_recommendations


def proposal_bundle(tmp_path):
    c = canonical(tmp_path)
    a = publish_atlas(c, tmp_path / "atlas", AtlasConfig())
    d = publish_discovery(a, tmp_path / "discovery", DiscoveryConfig())
    registry = {
        "schema_version": 1,
        "snapshot_date": "2026-10-22",
        "filters": [{"id": "a", "name": "TEST - A", "status": "TEST", "lifecycle": "existing"}],
    }
    cap = {"schema_version": 1, "exact_values": {}}
    out = publish_recommendations(
        d, a, c, tmp_path / "recommendations", RecommendationConfig(), registry, cap
    )
    return out, d, a, c, registry, cap


def test_replay_corruption_and_versions(tmp_path):
    out, d, a, c, r, cap = proposal_bundle(tmp_path)
    tables = load_recommendations(out)
    assert set(tables["proposals"].action) == {"COLLECT_MORE_DATA", "REVIEW_MANUALLY"}
    assert tables["proposals"].initial_review_status.eq("PROPOSED").all()
    assert (
        publish_recommendations(
            d, a, c, tmp_path / "recommendations", RecommendationConfig(), r, cap
        )
        == out
    )
    for name, frame in tables.items():
        assert_frame_equal(frame, load_recommendations(out)[name])
    changed = publish_recommendations(
        d,
        a,
        c,
        tmp_path / "recommendations",
        RecommendationConfig(draft_paper_hypotheses=False),
        r,
        cap,
    )
    assert changed != out
    assert set(tables["proposals"].proposal_id).isdisjoint(
        load_recommendations(changed)["proposals"].proposal_id
    )
    (out / "proposals.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_recommendations(out)


def test_mismatched_sources_and_atomic_failure(tmp_path, monkeypatch):
    out, d, a, c, r, cap = proposal_bundle(tmp_path)
    other = publish_atlas(c, tmp_path / "atlas", AtlasConfig(ev_edges=(0, 1)))
    with pytest.raises(ValueError, match="identities"):
        publish_recommendations(d, other, c, tmp_path / "wrong", RecommendationConfig(), r, cap)
    import oddsjam_ev.io.recommendations as module

    def fail(*args, **kwargs):
        raise OSError("synthetic failure")

    monkeypatch.setattr(module, "write_json", fail)
    with pytest.raises(OSError):
        publish_recommendations(d, a, c, tmp_path / "failed", RecommendationConfig(), r, cap)
    assert not list((tmp_path / "failed").iterdir())


def test_cli_and_manifest_tampering(tmp_path):
    out, d, a, c, r, cap = proposal_bundle(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/build_recommendations.py"),
            "--report",
            str(d),
            "--atlas",
            str(a),
            "--canonical",
            str(c),
            "--output-root",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["specification"]["policy"]["draft_paper_hypotheses"] = False
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="specification"):
        load_recommendations(out)
