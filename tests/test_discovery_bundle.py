"""Actual canonical/atlas integration, replay, corruption, CLI and atomic failures."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal
from test_atlas_bundle import canonical

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.io.atlas import publish_atlas
from oddsjam_ev.io.discovery import load_discovery, publish_discovery


def atlas_fixture(tmp_path):
    return publish_atlas(canonical(tmp_path), tmp_path / "atlas", AtlasConfig())


def test_replay_and_corruption(tmp_path):
    atlas = atlas_fixture(tmp_path)
    out = publish_discovery(atlas, tmp_path / "reports", DiscoveryConfig())
    tables = load_discovery(out)
    assert len(tables["candidate_ledger"]) == 10
    assert tables["candidate_results"].raw_pvalue.isna().all()
    assert tables["temporal_gates"].iloc[0].status == "blocked"
    assert publish_discovery(atlas, tmp_path / "reports", DiscoveryConfig()) == out
    for key, frame in tables.items():
        assert_frame_equal(frame, load_discovery(out)[key])
    changed = publish_discovery(atlas, tmp_path / "reports", DiscoveryConfig(seed=99))
    assert changed != out
    (out / "candidate_results.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_discovery(out)


def test_atomic_failure(tmp_path, monkeypatch):
    import oddsjam_ev.io.discovery as discovery

    atlas = atlas_fixture(tmp_path)

    def fail(*args, **kw):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(discovery, "write_json", fail)
    with pytest.raises(OSError):
        publish_discovery(atlas, tmp_path / "reports", DiscoveryConfig())
    assert not list((tmp_path / "reports").iterdir())


def test_cli_manifest_tampering_and_input_integrity(tmp_path):
    atlas = atlas_fixture(tmp_path)
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_discovery_report.py"),
            "--atlas",
            str(atlas),
            "--config",
            str(root / "configs/discovery_policy.yaml"),
            "--output-root",
            str(tmp_path / "reports"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    path = next((tmp_path / "reports").glob("*/manifest.json"))
    data = json.loads(path.read_text())
    data["specification"]["policy"]["alpha"] = 0.99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="specification"):
        load_discovery(path.parent)
    (atlas / "wager_dimensions.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        publish_discovery(atlas, tmp_path / "other", DiscoveryConfig())
