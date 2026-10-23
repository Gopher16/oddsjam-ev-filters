"""Local orchestration replay, freshness, failure isolation and publication gates."""

import errno
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from test_canonical_bundle import config, registry, source

from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
from oddsjam_ev.io.snapshots import materialize_snapshot
from oddsjam_ev.pipeline.config import PipelineConfig, read_config
from oddsjam_ev.pipeline.freshness import assess_freshness
from oddsjam_ev.pipeline.manifests import load_result
from oddsjam_ev.pipeline.runner import run_pipeline, utc_now


@pytest.fixture
def setup(tmp_path):
    """Use synthetic sources and isolated stores; never private local history."""
    files = {
        "source_config": {
            "store": str(tmp_path / "store"),
            "source": asdict(SourceMetadata("synthetic")),
        },
        "canonical_config": asdict(config()),
        "atlas_config": {"contract_version": 1},
        "discovery_config": {"bootstrap_resamples": 100},
        "recommendation_config": {"contract_version": 1},
        "registry": registry(),
        "capabilities": {"schema_version": 1, "exact_values": {}},
    }
    paths = {}
    for key, value in files.items():
        p = tmp_path / (key + ".yaml")
        p.write_text(yaml.safe_dump(value))
        paths[key] = str(p)
    cfg = PipelineConfig(output_root=str(tmp_path / "pipeline"), **paths)
    csv = source(tmp_path)
    import_csv(csv, tmp_path / "store", SourceMetadata("synthetic"))
    snapshot = materialize_snapshot(tmp_path / "store")
    return cfg, csv, snapshot, utc_now()


def result_path(cfg, record):
    return Path(cfg.output_root) / "results" / record["run_id"]


def test_historical_replay_and_optional_empty_register(setup, tmp_path):
    cfg, _, snapshot, cutoff = setup
    first = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff, register=tmp_path / "empty")
    assert first["status"] == "completed_historical", first
    before = (result_path(cfg, first) / "manifest.json").read_bytes()
    again = run_pipeline(cfg, resume=first["run_id"])
    assert again["run_id"] == first["run_id"] and again["attempt_id"] != first["attempt_id"]
    assert before == (result_path(cfg, again) / "manifest.json").read_bytes()
    assert sum(e["state"] == "reused" for e in again["stages"]) == 5
    result = load_result(result_path(cfg, first))
    assert result["specification"]["freshness"]["export_status"] == "unverified"
    assert not result["specification"]["freshness"]["complete_coverage_verified"]
    assert all(v == 0 for v in result["summaries"]["experiments"]["table_rows"].values())
    assert set(result["summaries"]["recommendations"]["research_actions"]) == {
        "COLLECT_MORE_DATA",
        "REVIEW_MANUALLY",
    }
    assert "historical_reanalysis" in result["report"]
    assert not list((Path(cfg.output_root) / "attempts").glob("*/notice.md"))


def test_missing_repeated_and_bad_source_never_fall_back(setup, tmp_path):
    cfg, csv, _, _ = setup
    assert run_pipeline(cfg)["status"] == "missing_input"
    assert run_pipeline(cfg, csv=tmp_path / "missing.csv")["status"] == "missing_input"
    assert run_pipeline(cfg, csv=csv)["status"] == "no_new_source_content"
    csv.write_text("bad,headers\nSECRET_MARKER,broken\n")
    out = run_pipeline(cfg, csv=csv)
    assert out["exit_code"] == 2
    assert not (Path(cfg.output_root) / "results").exists()
    for p in (Path(cfg.output_root) / "attempts").rglob("*"):
        if p.is_file():
            assert "SECRET_MARKER" not in p.read_text()


def test_recent_manual_import_is_not_complete_coverage(setup, tmp_path):
    cfg, _, _, _ = setup
    p = Path(cfg.source_config)
    raw = yaml.safe_load(p.read_text())
    raw["store"] = str(tmp_path / "new_store")
    raw["source"]["exported_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    p.write_text(yaml.safe_dump(raw))
    csv = source(tmp_path, profit="11", name="new.csv")
    outcome = run_pipeline(cfg, csv=csv)
    assert outcome["status"] == "completed", outcome
    data = load_result(result_path(cfg, outcome))
    assert data["specification"]["freshness"]["export_status"] == "recent"
    assert not data["specification"]["freshness"]["scope_verified"]
    assert run_pipeline(cfg, csv=csv)["status"] == "no_new_source_content"


@pytest.mark.parametrize(
    "export_age,expected", [(None, "freshness_unverified"), (10, "stale_input")]
)
def test_undated_or_stale_manual_import_emits_notice(setup, tmp_path, export_age, expected):
    cfg, _, _, _ = setup
    p = Path(cfg.source_config)
    raw = yaml.safe_load(p.read_text())
    raw["store"] = str(tmp_path / "new_store")
    if export_age:
        raw["source"]["exported_at"] = (datetime.now(UTC) - timedelta(days=export_age)).isoformat()
    p.write_text(yaml.safe_dump(raw))
    result = run_pipeline(cfg, csv=source(tmp_path, profit="11", name="new.csv"))
    assert result["status"] == expected
    assert result["exit_code"] == 3
    assert not (Path(cfg.output_root) / "results").exists()


def test_freshness_preserves_all_selected_sources_and_rejects_future():
    base = {
        "import_id": "one",
        "imported_at": "2026-10-22T12:00:00Z",
        "source": {"exported_at": None, "scope": "unknown"},
    }
    recent = {
        **base,
        "import_id": "two",
        "source": {"exported_at": "2026-10-22T11:00:00Z", "scope": "full"},
    }
    report = assess_freshness([base, recent], "2026-10-23T12:00:00Z", 8)
    assert report["export_status"] == "unverified"
    assert report["sources"][1]["export_age_days"] > 1
    with pytest.raises(ValueError):
        assess_freshness([recent], "2026-10-21T12:00:00Z", 8)
    recent["source"]["exported_at"] = "2026-10-22T13:00:00Z"
    with pytest.raises(ValueError):
        assess_freshness([recent], "2026-10-23T12:00:00Z", 8)


@pytest.mark.parametrize(
    "stage",
    [
        "build_canonical_bundle",
        "publish_atlas",
        "publish_discovery",
        "publish_recommendations",
        "render_report",
    ],
)
def test_partial_failure_resume_and_sanitized_errors(setup, monkeypatch, stage):
    import oddsjam_ev.pipeline.runner as runner

    cfg, _, snapshot, cutoff = setup
    original = getattr(runner, stage)

    def fail(*args, **kwargs):
        raise RuntimeError("SECRET_MARKER /private/account")

    monkeypatch.setattr(runner, stage, fail)
    failed = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    assert failed["exit_code"] == 2 and failed["run_id"]
    assert not result_path(cfg, failed).exists()
    monkeypatch.setattr(runner, stage, original)
    resumed = run_pipeline(cfg, resume=failed["run_id"])
    assert resumed["exit_code"] == 0, resumed
    for p in (Path(cfg.output_root) / "attempts").rglob("*.json"):
        assert "SECRET_MARKER" not in p.read_text()


def test_corrupt_checkpoint_and_changed_configuration_block_resume(setup):
    cfg, _, snapshot, cutoff = setup
    out = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    record = load_result(result_path(cfg, out))
    path = Path(record["outputs"]["atlas"]["path"])
    (path / "wager_segments.parquet").write_bytes(b"corrupt")
    failed = run_pipeline(cfg, resume=out["run_id"])
    assert failed["exit_code"] == 2
    assert any(e["stage"] == "atlas" and e["state"] == "failed" for e in failed["stages"])
    p = Path(cfg.discovery_config)
    p.write_text("bootstrap_resamples: 200\n")
    assert run_pipeline(cfg, resume=out["run_id"])["exit_code"] == 2


def test_transient_retry_is_bounded_and_permanent_errors_fail(setup, monkeypatch):
    import oddsjam_ev.pipeline.runner as runner

    cfg, _, snapshot, cutoff = setup
    original = runner.publish_atlas
    calls = []

    def transient(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.EINTR, "SECRET_MARKER")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "publish_atlas", transient)
    result = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    assert result["exit_code"] == 0 and len(calls) == 2
    assert any(e["state"] == "retry" for e in result["stages"])

    def permanent(*args, **kwargs):
        raise OSError(errno.EACCES, "SECRET_MARKER")

    monkeypatch.setattr(runner, "publish_atlas", permanent)
    other = replace(cfg, output_root=cfg.output_root + "_other")
    assert run_pipeline(other, snapshot=snapshot, as_of=cutoff)["exit_code"] == 2


def test_policy_cutoff_lock_and_cli(setup, tmp_path):
    cfg, _, snapshot, cutoff = setup
    with pytest.raises(ValueError):
        replace(cfg, scheduling_enabled=True)
    assert run_pipeline(cfg, snapshot=snapshot)["exit_code"] == 2
    assert run_pipeline(cfg, snapshot=snapshot, as_of="2020-01-01T00:00:00Z")["exit_code"] == 2
    assert run_pipeline(cfg, snapshot=snapshot, as_of="2099-01-01T00:00:00Z")["exit_code"] == 2
    lock = Path(cfg.output_root) / ".writer.lock"
    lock.write_text("operator_review_required")
    with pytest.raises(ValueError, match="locked"):
        run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    assert lock.read_text() == "operator_review_required"
    lock.unlink()
    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text(yaml.safe_dump(asdict(cfg)))
    assert read_config(config_file) == cfg
    script = Path(__file__).resolve().parents[1] / "scripts/run_weekly.py"
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config_file)], capture_output=True, text=True
    )
    assert result.returncode == 3 and "missing_input" in result.stdout
    assert result.stderr == ""


def test_interrupted_snapshot_reuses_original_receipt_cutoff(setup, tmp_path, monkeypatch):
    import oddsjam_ev.pipeline.runner as runner

    cfg, _, _, _ = setup
    p = Path(cfg.source_config)
    settings = yaml.safe_load(p.read_text())
    settings["store"] = str(tmp_path / "fresh_store")
    settings["source"]["exported_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    p.write_text(yaml.safe_dump(settings))
    csv = source(tmp_path, profit="11", name="new.csv")
    original = runner.materialize_snapshot

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "SECRET_MARKER")

    monkeypatch.setattr(runner, "materialize_snapshot", fail)
    failed = run_pipeline(cfg, csv=csv)
    assert failed["exit_code"] == 2 and failed["run_id"] is None
    imported = next((tmp_path / "fresh_store" / "imports").glob("*/manifest.json"))
    original_receipt = json.loads(imported.read_text())["imported_at"]
    # A later source must not leak into recovery of this import.
    import_csv(
        source(tmp_path, profit="9", name="later.csv"),
        tmp_path / "fresh_store",
        SourceMetadata(**settings["source"]),
    )
    monkeypatch.setattr(runner, "materialize_snapshot", original)
    recovered = run_pipeline(cfg, csv=csv)
    assert recovered["exit_code"] == 0, recovered
    result = load_result(result_path(cfg, recovered))
    assert result["specification"]["as_of"] == original_receipt
    assert len(result["specification"]["freshness"]["sources"]) == 1


def test_numeric_schema_gate_and_publication_failure(setup, tmp_path, monkeypatch):
    import oddsjam_ev.pipeline.runner as runner

    cfg, _, snapshot, cutoff = setup
    csv = source(tmp_path, profit="not_a_number", name="invalid.csv")
    assert run_pipeline(cfg, csv=csv)["exit_code"] == 2
    assert not (Path(cfg.output_root) / "results").exists()
    original = runner.load_result

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "SECRET_MARKER")

    monkeypatch.setattr(runner, "load_result", fail)
    failed = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    assert failed["exit_code"] == 2
    assert not list((Path(cfg.output_root) / "results").iterdir())
    monkeypatch.setattr(runner, "load_result", original)
    assert run_pipeline(cfg, resume=failed["run_id"])["exit_code"] == 0


def test_later_import_and_register_events_do_not_change_frozen_run(setup, tmp_path):
    from experiment_fixtures import approved

    from oddsjam_ev.io.experiment_register import read_experiment_events

    cfg, _, snapshot, cutoff = setup
    # An empty register is an explicit choice, not fabricated experiment evidence.
    register = tmp_path / "empty"
    first = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff, register=register)
    import_csv(
        source(tmp_path, profit="9", name="later.csv"),
        tmp_path / "store",
        SourceMetadata("synthetic"),
    )
    assert not read_experiment_events(register, cutoff)
    again = run_pipeline(cfg, resume=first["run_id"])
    assert again["run_id"] == first["run_id"] and again["exit_code"] == 0
    # Exercise event cutoff behavior with a real synthetic registration chain.
    (tmp_path / "synthetic_register").mkdir()
    with pytest.MonkeyPatch.context() as patch:
        event_root, *_ = approved(tmp_path / "synthetic_register", patch)
        assert read_experiment_events(event_root)
        assert not read_experiment_events(event_root, cutoff)
    later = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff, register=event_root)
    assert later["exit_code"] == 0
    result = load_result(result_path(cfg, later))
    assert result["specification"]["experiment_events"] == []


def test_result_and_plan_tampering_block_replay(setup):
    cfg, _, snapshot, cutoff = setup
    out = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    manifest = result_path(cfg, out) / "manifest.json"
    contents = json.loads(manifest.read_text())
    contents["summaries"]["canonical"]["table_rows"]["bet_facts"] = 999
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="payload integrity"):
        load_result(manifest.parent)
    assert run_pipeline(cfg, resume=out["run_id"])["exit_code"] == 2
    plan = Path(cfg.output_root) / "plans" / (out["run_id"] + ".json")
    contents = json.loads(plan.read_text())
    contents["specification"]["as_of"] = "2020-01-01T00:00:00Z"
    plan.write_text(json.dumps(contents))
    assert run_pipeline(cfg, resume=out["run_id"])["exit_code"] == 2


def test_configuration_change_during_stage_blocks_publication(setup, monkeypatch):
    import oddsjam_ev.pipeline.runner as runner

    cfg, _, snapshot, cutoff = setup
    original = runner.publish_recommendations

    def change_config(*args, **kwargs):
        out = original(*args, **kwargs)
        Path(cfg.discovery_config).write_text("bootstrap_resamples: 200\n")
        return out

    monkeypatch.setattr(runner, "publish_recommendations", change_config)
    failed = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff)
    assert failed["exit_code"] == 2
    assert failed["stages"][-1]["stage"] == "publication"
    assert not result_path(cfg, failed).exists()


def test_csv_change_between_validation_and_import_is_rejected(setup, tmp_path, monkeypatch):
    import oddsjam_ev.pipeline.runner as runner

    cfg, _, _, _ = setup
    original = runner.import_csv
    csv = source(tmp_path, profit="11", name="race.csv")

    def changed(path, *args, **kwargs):
        path.write_text(path.read_text().replace(",11,", ",9,"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runner, "import_csv", changed)
    failed = run_pipeline(cfg, csv=csv)
    assert failed["exit_code"] == 2
    assert not (Path(cfg.output_root) / "results").exists()


def test_retry_exhaustion_and_experiment_failure_are_not_success(setup, tmp_path, monkeypatch):
    import oddsjam_ev.pipeline.runner as runner

    cfg, _, snapshot, cutoff = setup
    original = runner.publish_atlas
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise OSError(errno.ETIMEDOUT, "SECRET_MARKER")

    monkeypatch.setattr(runner, "publish_atlas", fail)
    failed = run_pipeline(cfg, snapshot=snapshot, as_of=cutoff, register=tmp_path / "empty")
    assert failed["exit_code"] == 2 and len(calls) == 2
    monkeypatch.setattr(runner, "publish_atlas", original)
    monkeypatch.setattr(runner, "publish_experiment_report", fail)
    failed = run_pipeline(cfg, resume=failed["run_id"])
    assert failed["exit_code"] == 2
    assert any(e["stage"] == "experiments" and e["state"] == "failed" for e in failed["stages"])
    assert not result_path(cfg, failed).exists()
