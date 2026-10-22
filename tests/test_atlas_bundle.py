"""Immutable publication, actual canonical integration, integrity and failure tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal
from test_canonical_bundle import config, registry, source

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.io.atlas import load_atlas, publish_atlas
from oddsjam_ev.io.canonical import build_canonical_bundle
from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
from oddsjam_ev.io.snapshots import materialize_snapshot


def canonical(tmp_path):
    store = tmp_path / "store"
    import_csv(source(tmp_path), store, SourceMetadata("synthetic"))
    snapshot = materialize_snapshot(store)
    return build_canonical_bundle(snapshot, tmp_path / "canonical", config(), registry())


def test_real_bundle_replay_policy_and_corruption(tmp_path):
    bundle = canonical(tmp_path)
    output = publish_atlas(bundle, tmp_path / "atlas", AtlasConfig())
    tables = load_atlas(output)
    assert tables["wager_segments"].query("view == 'overall'").eligible_stake_usd.sum() == 10
    assert publish_atlas(bundle, tmp_path / "atlas", AtlasConfig()) == output
    for name, frame in tables.items():
        assert_frame_equal(frame, load_atlas(output)[name])
    changed = publish_atlas(bundle, tmp_path / "atlas", AtlasConfig(ev_edges=(0, 10)))
    assert changed != output
    (output / "wager_segments.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_atlas(output)


def test_publication_failure(tmp_path, monkeypatch):
    import oddsjam_ev.io.atlas as atlas

    bundle = canonical(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("synthetic failure")

    monkeypatch.setattr(atlas, "write_json", fail)
    with pytest.raises(OSError):
        publish_atlas(bundle, tmp_path / "atlas", AtlasConfig())
    assert not list((tmp_path / "atlas").iterdir())


def test_cli_and_manifest_tampering(tmp_path):
    bundle = canonical(tmp_path)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_opportunity_atlas.py"),
            "--bundle",
            str(bundle),
            "--config",
            str(root / "configs/opportunity_atlas.yaml"),
            "--output-root",
            str(tmp_path / "atlas"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    output = next((tmp_path / "atlas").glob("*/manifest.json"))
    data = json.loads(output.read_text())
    data["specification"]["policy"]["contract_version"] = 99
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="specification"):
        load_atlas(output.parent)
