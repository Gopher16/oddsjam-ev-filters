"""Complete lineage, confinement, version gates, cache invalidation and read-only IO."""

import hashlib
import json
from pathlib import Path

import pytest

from oddsjam_ev.dashboard.artifacts import (
    DashboardConfig,
    discover_runs,
    load_published_run,
    operational_attempts,
)
from oddsjam_ev.dashboard.queries import validate_display_contract
from oddsjam_ev.io.source_csv import fingerprint


def reseal_result(path, result):
    result["payload_sha256"] = fingerprint(
        {k: v for k, v in result.items() if k != "payload_sha256"}
    )
    path.write_text(json.dumps(result))


def test_load_all_stages_and_no_file_changes(dashboard_run, monkeypatch):
    import oddsjam_ev.io.ingestion as ingestion
    import oddsjam_ev.pipeline.runner as pipeline

    root, run_id = dashboard_run
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Dashboard attempted a write stage")

    monkeypatch.setattr(ingestion, "import_csv", forbidden)
    for name in (
        "build_canonical_bundle",
        "publish_atlas",
        "publish_discovery",
        "publish_recommendations",
        "publish_experiment_report",
    ):
        monkeypatch.setattr(pipeline, name, forbidden)
    run = load_published_run(root, run_id)
    validate_display_contract(run.tables)
    assert set(run.tables) == {"canonical", "atlas", "discovery", "recommendations", "experiments"}
    assert run.result["specification"]["freshness"]["export_status"] == "unverified"
    assert discover_runs(root) == [run_id]
    assert operational_attempts(root)[0]["status"] == "completed_historical"
    assert before == {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }


def test_cache_rehash_detects_corruption_even_with_preserved_mtime(dashboard_run):
    import os

    root, run_id = dashboard_run
    run = load_published_run(root, run_id)
    p = next(p for p in run.expected_files if p.name == "wager_segments.parquet")
    stat = p.stat()
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 1
    p.write_bytes(data)
    os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="integrity"):
        run.verify()
    with pytest.raises(ValueError):
        load_published_run(root, run_id)


def test_missing_table_and_invalid_selection_fail(dashboard_run):
    root, run_id = dashboard_run
    for value in ("../other", "latest", "a" * 64):
        with pytest.raises((ValueError, OSError)):
            load_published_run(root, value)
    run = load_published_run(root, run_id)
    next(p for p in run.expected_files if p.name == "wager_segments.parquet").unlink()
    with pytest.raises(OSError):
        load_published_run(root, run_id)


def test_mismatched_reference_and_unsupported_contract(dashboard_run):
    root, run_id = dashboard_run
    path = root / "results" / run_id / "manifest.json"
    result = json.loads(path.read_text())
    result["outputs"]["canonical"]["manifest_sha256"] = "0" * 64
    record = result["outputs"]["canonical"]
    record["checkpoint_sha256"] = fingerprint(
        {k: v for k, v in record.items() if k != "checkpoint_sha256"}
    )
    reseal_result(path, result)
    with pytest.raises(ValueError, match="integrity"):
        load_published_run(root, run_id)
    with pytest.raises(ValueError):
        DashboardConfig(contract_version=2)
    with pytest.raises(ValueError):
        DashboardConfig(artifact_root=".local/oddsjam")


def test_symlink_escape_never_reads_outside_root(dashboard_run, tmp_path):
    root, run_id = dashboard_run
    run = load_published_run(root, run_id)
    p = next(p for p in run.expected_files if p.name == "wager_segments.parquet")
    external = tmp_path / "outside.parquet"
    p.rename(external)
    p.symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        run.verify()
    with pytest.raises(ValueError, match="outside"):
        load_published_run(root, run_id)


def test_embedded_snapshot_mismatch_and_stage_lineage(dashboard_run):
    root, run_id = dashboard_run
    run = load_published_run(root, run_id)
    # Alter an atlas parent, then recompute all outer hashes. Cross-stage checks
    # must reject this despite valid individual manifest fingerprints.
    atlas_record = run.result["outputs"]["atlas"]
    path = Path(atlas_record["path"]) / "manifest.json"
    atlas = json.loads(path.read_text())
    atlas["specification"]["canonical_bundle_id"] = "0" * 64
    atlas["atlas_id"] = fingerprint(atlas["specification"])
    path.write_text(json.dumps(atlas))
    atlas_record["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atlas_record["checkpoint_sha256"] = fingerprint(
        {k: v for k, v in atlas_record.items() if k != "checkpoint_sha256"}
    )
    reseal_result(root / "results" / run_id / "manifest.json", run.result)
    with pytest.raises(ValueError, match="lineage"):
        load_published_run(root, run_id)


def test_operational_records_are_allowlisted_and_not_result_selection(dashboard_run):
    root, run_id = dashboard_run
    bad = root / "attempts" / ("f" * 32)
    bad.mkdir()
    (bad / "outcome.json").write_text(
        json.dumps({"status": "SECRET_MARKER", "error": "SECRET_MARKER", "run_id": "SECRET_MARKER"})
    )
    rows = operational_attempts(root)
    assert "SECRET_MARKER" not in str(rows)
    assert discover_runs(root) == [run_id]
    assert operational_attempts(root / "missing") == []


@pytest.mark.parametrize(
    "column,value", [("canonical_bundle_id", "0" * 64), ("atlas_contract_version", 99)]
)
def test_table_parent_and_contract_are_verified_beyond_byte_hashes(
    dashboard_run, monkeypatch, column, value
):
    import oddsjam_ev.dashboard.artifacts as adapter

    root, run_id = dashboard_run
    original = adapter.LOADERS["atlas"]

    def inconsistent(path):
        tables = original(path)
        tables["wager_segments"][column] = value
        return tables

    monkeypatch.setitem(adapter.LOADERS, "atlas", inconsistent)
    with pytest.raises(ValueError, match="lineage"):
        load_published_run(root, run_id)
